//! Compaction-backed write pacing.
//!
//! The controller in this module deliberately knows nothing about WAL size. A WAL is on the
//! critical durability path even when flush and compaction are healthy, so its growth is a poor
//! proxy for whether writes have outrun the background pipeline. The signals here are the backlog
//! that compression/flush/compaction actually drains and, more importantly, whether that backlog
//! is growing or falling.
//!
//! Short peaks spend a burst budget instead of being paced. Once the budget is exhausted, the
//! allowed rate follows the measured pressure. Recovery is additive and follows real drainage,
//! avoiding both a stale fixed "half speed" state and an abrupt release that immediately rebuilds
//! the same L0 backlog.

use std::time::Duration;

/// Tuning for [`WritePressureController`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct WritePressureConfig {
	/// Amount of normalized pressure-seconds a burst may spend before pacing starts.
	///
	/// Pressure `1.2` spends `0.2` units per second above the healthy line. The default budget of
	/// `120` therefore tolerates roughly ten minutes at that mild overload before engaging. A
	/// severe `2x` backlog engages after two minutes.
	pub burst_budget_load_seconds: f64,

	/// Time constant used to smooth noisy L0/memtable readings.
	pub pressure_smoothing: Duration,

	/// Smallest rate fraction handed back to the writer path before hard stalls take over.
	pub min_rate: f64,

	/// Additive rate recovery per second.
	pub recovery_rate_per_second: f64,

	/// Rate at which an excessive target is approached.
	pub decrease_rate_per_second: f64,
}

impl Default for WritePressureConfig {
	fn default() -> Self {
		Self {
			burst_budget_load_seconds: 120.0,
			pressure_smoothing: Duration::from_secs(3),
			min_rate: 0.25,
			recovery_rate_per_second: 1.0 / 45.0,
			decrease_rate_per_second: 4.0,
		}
	}
}

/// The result of one pressure update.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct PressureDecision {
	/// Smoothed normalized pressure. `1.0` is the onset of soft backpressure.
	pub pressure: f64,

	/// Fraction of unthrottled writes that should be admitted, in `[min_rate, 1.0]`.
	pub allowed_rate: f64,

	/// Whether sustained pressure has exhausted the burst budget.
	pub engaged: bool,

	/// Remaining burst budget in normalized pressure-seconds.
	pub burst_remaining: f64,

	/// Smoothed backlog growth in bytes per second; negative means it is draining.
	pub backlog_growth_bytes_per_second: f64,
}

impl PressureDecision {
	/// Converts the rate fraction to a per-writer delay.
	///
	/// `base_delay` represents the spacing at full speed. The returned delay grows as
	/// `1/rate - 1`. This is a pacing hint rather than an exact global throughput contract; the
	/// hard memtable/L0 limits remain the safety bound.
	pub(crate) fn pacing_delay(self, base_delay: Duration, max_delay: Duration) -> Duration {
		if self.allowed_rate >= 1.0 {
			return Duration::ZERO;
		}

		let extra = 1.0 / self.allowed_rate.max(f64::MIN_POSITIVE) - 1.0;
		let delay = base_delay.as_secs_f64() * extra;
		Duration::from_secs_f64(delay.min(max_delay.as_secs_f64()))
	}
}

/// Learns the healthy commit throughput and paces bytes once pressure is engaged.
///
/// The gate is deliberately inactive while writes are healthy. It exists to turn the policy's
/// rate fraction into an actual byte deadline after sustained compaction backlog.
#[derive(Debug, Clone)]
pub(crate) struct ThroughputGate {
	window_start: Option<Duration>,
	window_bytes: u128,
	demand_bytes_per_second: f64,
	token_balance: f64,
	last_refill: Option<Duration>,
}

impl Default for ThroughputGate {
	fn default() -> Self {
		Self {
			window_start: None,
			window_bytes: 0,
			demand_bytes_per_second: 0.0,
			token_balance: f64::INFINITY,
			last_refill: None,
		}
	}
}

impl ThroughputGate {
	#[cfg(test)]
	pub(crate) fn demand_bytes_per_second(&self) -> f64 {
		self.demand_bytes_per_second
	}

	/// Accounts one writer's bytes and returns how long it should wait before committing them.
	pub(crate) fn reserve(
		&mut self,
		at: Duration,
		bytes: u64,
		allowed_rate: f64,
		engaged: bool,
	) -> Duration {
		self.observe_demand(at, bytes);

		if !engaged || allowed_rate >= 0.999 || self.demand_bytes_per_second < 1.0 {
			self.token_balance = f64::INFINITY;
			self.last_refill = Some(at);
			return Duration::ZERO;
		}

		let rate = self.demand_bytes_per_second * allowed_rate.clamp(0.05, 1.0);
		let burst_tokens = rate; // One second of target throughput.
		let last_refill = self.last_refill.unwrap_or(at);
		let elapsed = at.saturating_sub(last_refill).as_secs_f64().min(2.0);
		if self.token_balance.is_finite() {
			self.token_balance = (self.token_balance + rate * elapsed).min(burst_tokens);
		} else {
			self.token_balance = burst_tokens;
		}
		self.last_refill = Some(at);
		self.token_balance -= bytes as f64;

		if self.token_balance >= 0.0 {
			Duration::ZERO
		} else {
			Duration::from_secs_f64((-self.token_balance) / rate)
		}
	}

	fn observe_demand(&mut self, at: Duration, bytes: u64) {
		const WINDOW: Duration = Duration::from_secs(1);

		if self.window_start.is_none() {
			self.window_start = Some(at);
		}
		self.window_bytes = self.window_bytes.saturating_add(bytes as u128);

		let window_start = self.window_start.unwrap_or(at);
		let elapsed = at.saturating_sub(window_start);
		if elapsed < WINDOW {
			return;
		}

		let observed = self.window_bytes as f64 / elapsed.as_secs_f64();
		self.demand_bytes_per_second = if self.demand_bytes_per_second <= 0.0 {
			observed
		} else {
			self.demand_bytes_per_second * 0.5 + observed * 0.5
		};
		self.window_start = Some(at);
		self.window_bytes = 0;
	}
}

/// Stateful burst-tolerant, compaction-backed pacing policy.
#[derive(Debug, Clone)]
pub(crate) struct WritePressureController {
	config: WritePressureConfig,
	pressure: Option<f64>,
	pressure_slope_per_second: f64,
	burst_remaining: f64,
	engaged: bool,
	allowed_rate: f64,
	last_at: Option<Duration>,
	last_backlog_bytes: Option<u64>,
	backlog_growth_bytes_per_second: f64,
	throughput: ThroughputGate,
}

impl Default for WritePressureController {
	fn default() -> Self {
		Self::new(WritePressureConfig::default())
	}
}

impl WritePressureController {
	pub(crate) fn new(config: WritePressureConfig) -> Self {
		let burst = config.burst_budget_load_seconds.max(0.0);
		Self {
			config,
			pressure: None,
			pressure_slope_per_second: 0.0,
			burst_remaining: burst,
			engaged: false,
			allowed_rate: 1.0,
			last_at: None,
			last_backlog_bytes: None,
			backlog_growth_bytes_per_second: 0.0,
			throughput: ThroughputGate::default(),
		}
	}

	pub(crate) fn decision(&self) -> PressureDecision {
		PressureDecision {
			pressure: self.pressure.unwrap_or(0.0),
			allowed_rate: self.allowed_rate,
			engaged: self.engaged,
			burst_remaining: self.burst_remaining,
			backlog_growth_bytes_per_second: self.backlog_growth_bytes_per_second,
		}
	}

	/// Accounts commit bytes and returns a concrete throughput wait after pressure is engaged.
	pub(crate) fn reserve_write_bytes(
		&mut self,
		at: Duration,
		bytes: u64,
		decision: PressureDecision,
	) -> Duration {
		self.throughput.reserve(at, bytes, decision.allowed_rate, decision.engaged)
	}

	/// Feeds one live sample and advances the policy to `at`.
	///
	/// `raw_pressure` is the maximum of the normalized memtable, L0-file and L0-byte loads.
	/// `backlog_bytes` is an absolute pending-work estimate used to detect real drainage.
	pub(crate) fn observe(
		&mut self,
		at: Duration,
		raw_pressure: f64,
		backlog_bytes: u64,
	) -> PressureDecision {
		const MAX_STEP: Duration = Duration::from_secs(2);
		const SLOPE_TAU: Duration = Duration::from_secs(5);
		const BACKLOG_TAU: Duration = Duration::from_secs(10);
		const RISING_SLOPE: f64 = 0.02;
		const FALLING_SLOPE: f64 = -0.02;

		let raw_pressure = if raw_pressure.is_finite() {
			raw_pressure.clamp(0.0, 100.0)
		} else {
			0.0
		};
		let elapsed = self.last_at.map(|last| at.saturating_sub(last)).unwrap_or(Duration::ZERO);
		let dt = elapsed.min(MAX_STEP);
		let dt_seconds = dt.as_secs_f64();

		match self.pressure {
			None => self.pressure = Some(raw_pressure),
			Some(previous) if dt_seconds > 0.0 => {
				let tau = self.config.pressure_smoothing.as_secs_f64().max(f64::MIN_POSITIVE);
				let alpha = 1.0 - (-dt_seconds / tau).exp();
				let next = previous + alpha * (raw_pressure - previous);
				let instant_slope = (next - previous) / dt_seconds;
				let slope_alpha = 1.0 - (-dt_seconds / SLOPE_TAU.as_secs_f64()).exp();
				self.pressure_slope_per_second +=
					slope_alpha * (instant_slope - self.pressure_slope_per_second);
				self.pressure = Some(next);
			}
			_ => {}
		}

		if let Some(previous_bytes) = self.last_backlog_bytes {
			let elapsed_seconds = elapsed.as_secs_f64();
			if elapsed_seconds > 0.0 {
				let instant_growth =
					(backlog_bytes as f64 - previous_bytes as f64) / elapsed_seconds;
				let alpha = 1.0 - (-elapsed_seconds / BACKLOG_TAU.as_secs_f64()).exp();
				self.backlog_growth_bytes_per_second +=
					alpha * (instant_growth - self.backlog_growth_bytes_per_second);
			}
		}
		self.last_backlog_bytes = Some(backlog_bytes);
		self.last_at = Some(at);

		let pressure = self.pressure.unwrap_or(raw_pressure);
		if pressure > 1.0 {
			self.burst_remaining = (self.burst_remaining - (pressure - 1.0) * dt_seconds).max(0.0);
		} else if pressure < 1.0 {
			// Idle compaction capacity recharges twice as quickly as pressure spends budget.
			let recharge = (1.0 - pressure).clamp(0.0, 1.0) * 2.0 * dt_seconds;
			self.burst_remaining =
				(self.burst_remaining + recharge).min(self.config.burst_budget_load_seconds);
		}

		if !self.engaged && self.burst_remaining == 0.0 && pressure >= 1.0 {
			self.engaged = true;
		}

		if self.engaged {
			let min_rate = self.config.min_rate.clamp(0.05, 1.0);
			let mut target = (1.0 / pressure.max(1.0)).clamp(min_rate, 1.0);

			// A backlog that is still building deserves earlier protection. One that is visibly
			// draining can release a little sooner, but never above the pressure-derived target.
			if self.pressure_slope_per_second > RISING_SLOPE {
				target = (target * 0.85).max(min_rate);
			} else if self.pressure_slope_per_second < FALLING_SLOPE {
				target = (target + 0.15).min(1.0);
			}

			if target < self.allowed_rate {
				let step = self.config.decrease_rate_per_second.max(0.0) * dt_seconds;
				self.allowed_rate = (self.allowed_rate - step).max(target);
			} else if target > self.allowed_rate {
				let drainage_boost = if self.pressure_slope_per_second < FALLING_SLOPE {
					2.0
				} else {
					1.0
				};
				let step =
					self.config.recovery_rate_per_second.max(0.0) * drainage_boost * dt_seconds;
				self.allowed_rate = (self.allowed_rate + step).min(target);
			}

			if self.allowed_rate >= 0.999 && pressure < 0.95 {
				self.engaged = false;
				self.allowed_rate = 1.0;
				self.burst_remaining = self.config.burst_budget_load_seconds;
				self.pressure_slope_per_second = 0.0;
			}
		} else {
			self.allowed_rate = 1.0;
		}

		self.decision()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	fn minute(minute: u64) -> Duration {
		Duration::from_secs(minute * 60)
	}

	#[test]
	fn short_and_medium_bursts_are_not_paced() {
		let mut pacer = WritePressureController::default();

		// Two minutes at 1.5x soft pressure spends 60 of the 120-unit budget.
		for second in 0..120 {
			let decision = pacer.observe(Duration::from_secs(second), 1.5, 64 * 1024 * 1024);
			assert_eq!(decision.allowed_rate, 1.0);
			assert!(!decision.engaged);
		}

		assert!(pacer.decision().burst_remaining > 55.0);
		assert!(pacer.decision().burst_remaining < 65.0);
	}

	#[test]
	fn a_ten_minute_mild_event_peak_is_tolerated_before_gradual_pacing() {
		let mut pacer = WritePressureController::default();

		// The event-stream peak lasts more than ten minutes at a mild 1.2x load. The excess is
		// 0.2 pressure/second, so the default budget is exhausted after about ten minutes.
		for second in 0..600 {
			let decision = pacer.observe(Duration::from_secs(second), 1.2, 128 * 1024 * 1024);
			assert_eq!(decision.allowed_rate, 1.0, "second {second} was paced early");
		}

		let decision = pacer.observe(Duration::from_secs(601), 1.2, 129 * 1024 * 1024);
		assert!(decision.engaged);
		assert!(decision.allowed_rate > 0.80);
		assert!(decision.allowed_rate < 0.87);

		// Pacing stays gradual rather than jumping to a fixed half rate.
		for second in 602..900 {
			let decision = pacer.observe(Duration::from_secs(second), 1.2, 160 * 1024 * 1024);
			assert!(decision.allowed_rate >= WritePressureConfig::default().min_rate - 0.001);
			assert!(decision.allowed_rate <= 0.84);
		}
	}

	#[test]
	fn real_drainage_releases_a_half_speed_limit_promptly() {
		let mut pacer = WritePressureController::default();

		// Severe pressure engages after two minutes and reaches the pressure-derived half rate.
		for second in 0..130 {
			pacer.observe(Duration::from_secs(second), 2.0, 512 * 1024 * 1024);
		}
		let saturated = pacer.decision();
		assert!(saturated.engaged);
		assert!((saturated.allowed_rate - 0.5).abs() < 0.02);

		// Compaction catches up. The absolute reading falls and the backlog shrinks. The old
		// WAL-sized gate could remain at half speed for minutes; this policy must ramp back within
		// tens of seconds.
		let mut recovered_at = None;
		let mut backlog = 512 * 1024 * 1024u64;
		for second in 130..240 {
			backlog = backlog.saturating_sub(16 * 1024 * 1024);
			let decision = pacer.observe(Duration::from_secs(second), 0.35, backlog);
			if !decision.engaged && decision.allowed_rate == 1.0 {
				recovered_at = Some(second);
				break;
			}
		}

		let recovered_at = recovered_at.expect("pacer did not return to full speed");
		assert!(recovered_at - 130 < 60, "recovery took {}s", recovered_at - 130);
	}

	#[test]
	fn growing_backlog_is_paced_harder_than_a_draining_one() {
		fn final_rate(slope_samples: &[(u64, f64, u64)]) -> f64 {
			let mut pacer = WritePressureController::new(WritePressureConfig {
				burst_budget_load_seconds: 0.0,
				..WritePressureConfig::default()
			});
			for (second, pressure, backlog) in slope_samples {
				pacer.observe(Duration::from_secs(*second), *pressure, *backlog);
			}
			pacer.decision().allowed_rate
		}

		let growing = (0..30)
			.map(|second| (second, 1.0 + second as f64 * 0.05, second * 10_000_000))
			.collect::<Vec<_>>();
		let draining = (0..30)
			.map(|second| {
				(
					second,
					2.4 - second as f64 * 0.05,
					300_000_000u64.saturating_sub(second * 10_000_000),
				)
			})
			.collect::<Vec<_>>();

		assert!(final_rate(&growing) < final_rate(&draining));
	}

	#[test]
	fn long_peak_followed_by_slow_traffic_decline_recovers() {
		let mut pacer = WritePressureController::default();
		let mut backlog = 256 * 1024 * 1024u64;

		// Fifteen minutes of peak traffic. Pressure varies but stays above target often enough to
		// engage the adaptive limiter.
		for second in 0..900u64 {
			let pressure = 1.05 + (second % 60) as f64 / 180.0;
			backlog = backlog.saturating_add(64 * 1024);
			pacer.observe(Duration::from_secs(second), pressure, backlog);
		}
		assert!(pacer.decision().engaged, "the sustained peak should engage pacing");

		// Traffic and the residual backlog decline gradually over five minutes.
		for second in 900..1200u64 {
			let decline = (second - 900) as f64 / 300.0;
			let pressure = 1.15 - decline * 0.8;
			backlog = backlog.saturating_sub(256 * 1024);
			let decision = pacer.observe(Duration::from_secs(second), pressure, backlog);
			assert!(decision.allowed_rate >= 0.25);
		}

		assert!(
			!pacer.decision().engaged,
			"the pacer stayed engaged after the event-stream peak declined: {:?}",
			pacer.decision()
		);
		assert_eq!(pacer.decision().allowed_rate, 1.0);
		assert!(minute(20) > Duration::ZERO);
	}

	#[test]
	fn pacing_delay_scales_with_rate_and_is_bounded() {
		let base = Duration::from_micros(200);
		let max = Duration::from_millis(2);

		let full = PressureDecision {
			pressure: 0.5,
			allowed_rate: 1.0,
			engaged: false,
			burst_remaining: 120.0,
			backlog_growth_bytes_per_second: -1.0,
		};
		assert_eq!(full.pacing_delay(base, max), Duration::ZERO);

		let half = PressureDecision {
			allowed_rate: 0.5,
			..full
		};
		assert_eq!(half.pacing_delay(base, max), Duration::from_micros(200));

		let quarter = PressureDecision {
			allowed_rate: 0.25,
			..full
		};
		assert_eq!(quarter.pacing_delay(base, max), Duration::from_micros(600));

		let tiny = PressureDecision {
			allowed_rate: 0.001,
			..full
		};
		assert_eq!(tiny.pacing_delay(base, max), max);
	}

	#[test]
	fn throughput_gate_is_inactive_during_healthy_bursts() {
		let mut gate = ThroughputGate::default();
		assert_eq!(gate.reserve(Duration::ZERO, 4096, 1.0, false), Duration::ZERO);
		assert_eq!(gate.reserve(Duration::from_millis(1), 4096, 1.0, false), Duration::ZERO);
	}

	#[test]
	fn throughput_gate_spaces_excess_bytes_at_the_adaptive_rate() {
		let mut gate = ThroughputGate::default();

		// Learn a stable 1 MiB/s healthy commit throughput over a one-second window.
		gate.reserve(Duration::ZERO, 0, 1.0, false);
		gate.reserve(Duration::from_secs(1), 1024 * 1024, 1.0, false);
		assert!((gate.demand_bytes_per_second() - 1024.0 * 1024.0).abs() < 1.0);

		let at = Duration::from_secs(1);
		let half_rate = 0.5;

		// One second of target throughput is available as the engagement burst.
		assert_eq!(gate.reserve(at, 512 * 1024, half_rate, true), Duration::ZERO);

		// Another half MiB at the same instant needs one second of service at 512 KiB/s.
		let wait = gate.reserve(at, 512 * 1024, half_rate, true);
		assert!(
			(wait.as_secs_f64() - 1.0).abs() < 0.01,
			"expected a one-second spacing delay, got {wait:?}"
		);
	}
}
