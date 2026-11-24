//! Write stall controller for backpressure management.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::Notify;

use crate::error::{Error, Result, WriteStallReason};
use crate::rt::{pace, Instant};
use crate::write_pressure::WritePressureController;

/// Asks the background tasks for the work that ends a stall of the given reason.
type Rearm = Box<dyn Fn(WriteStallReason) + Send + Sync>;

/// Current counts of resources that can trigger write stalls.
#[derive(Debug, Clone, Copy)]
pub struct StallCounts {
	/// Number of immutable memtables queued for flush
	pub immutable_memtables: usize,
	/// Number of L0 SSTable files awaiting compaction
	pub l0_files: usize,
}

/// Live measurements of the background work that write pacing should track.
///
/// WAL bytes are intentionally absent. A large healthy WAL is expected during a sustained event
/// stream peak; the relevant question is whether flush/compression/compaction is draining the
/// memtable and level-zero backlog.
#[derive(Debug, Clone, Copy, Default)]
pub struct CompactionPressure {
	/// Total logical bytes held by L0 tables.
	pub l0_bytes: u64,
	/// Byte load at which L0 is considered softly pressured.
	pub l0_byte_target: u64,
	/// Bytes held by immutable memtables waiting to be flushed.
	pub immutable_bytes: u64,
	/// Total pending bytes, when a provider has a better compaction-level estimate than L0 plus
	/// immutable memtables.
	pub backlog_bytes: u64,
}

/// Thresholds that trigger write stalls when exceeded.
#[derive(Debug, Clone, Copy)]
pub struct StallThresholds {
	/// Maximum immutable memtable count before hard stalling writes
	pub memtable_limit: usize,
	/// Soft threshold where proactive pacing delays start for memtables
	pub memtable_soft_limit: usize,
	/// Maximum L0 file count before hard stalling writes
	pub l0_file_limit: usize,
	/// Soft threshold where proactive pacing delays start for L0 files
	pub l0_file_soft_limit: usize,
}

impl StallThresholds {
	pub fn new(memtable_limit: usize, l0_file_limit: usize) -> Self {
		Self {
			memtable_limit,
			memtable_soft_limit: memtable_limit.saturating_sub(2).max(1),
			l0_file_limit,
			l0_file_soft_limit: l0_file_limit.saturating_sub(4).max(1),
		}
	}
}

/// Trait for getting current stall condition counts.
/// Implementors provide live resource counts that the controller
/// checks against its configured thresholds.
pub trait WriteStallCountProvider: Send + Sync + 'static {
	/// Get current immutable memtable count and L0 file count.
	fn get_stall_counts(&self) -> StallCounts;

	/// Get byte-level compaction pressure. The default keeps lightweight test providers focused
	/// on the hard-stall counts.
	fn get_compaction_pressure(&self) -> CompactionPressure {
		CompactionPressure::default()
	}
}

/// Information about a write stall event.
#[derive(Debug, Clone)]
pub struct WriteStallInfo {
	/// The reason for the stall
	pub reason: WriteStallReason,
	/// The current value that triggered the stall
	pub current_value: usize,
	/// The threshold that was exceeded
	pub threshold: usize,
	/// Duration spent stalled
	pub duration: Duration,
}

/// Controller that manages write stall state and signaling.
///
/// Owns its count provider and thresholds, providing a self-contained
/// API for the commit path to check backpressure conditions.
pub struct WriteStallController {
	/// Notification for when stall conditions clear
	stall_cleared: Notify,

	/// Current stall state (for fast-path check)
	is_stalled: AtomicBool,

	/// Shutdown flag - checked in stall loop for graceful exit
	shutdown: AtomicBool,

	/// Counts `signal_failure` calls: a stalled writer gives up when it changes while it waits
	failures: AtomicU64,

	/// Provider for live stall count readings
	provider: Arc<dyn WriteStallCountProvider>,

	/// Static thresholds configured at startup
	thresholds: StallThresholds,

	/// Burst-tolerant adaptive pacer driven by compaction backlog rather than WAL size.
	pacer: Mutex<WritePressureController>,

	/// Clock base for the pure pacer's testable `Duration` timeline.
	started_at: Instant,

	/// Asks for a flush or a compaction for a stalled writer, once the tasks exist
	rearm: OnceLock<Rearm>,
}

impl WriteStallController {
	pub fn new(provider: Arc<dyn WriteStallCountProvider>, thresholds: StallThresholds) -> Self {
		Self::with_config(
			provider,
			thresholds,
			crate::write_pressure::WritePressureConfig::default(),
		)
	}

	pub(crate) fn with_config(
		provider: Arc<dyn WriteStallCountProvider>,
		thresholds: StallThresholds,
		config: crate::write_pressure::WritePressureConfig,
	) -> Self {
		Self {
			stall_cleared: Notify::new(),
			is_stalled: AtomicBool::new(false),
			shutdown: AtomicBool::new(false),
			failures: AtomicU64::new(0),
			provider,
			thresholds,
			pacer: Mutex::new(WritePressureController::new(config)),
			started_at: Instant::now(),
			rearm: OnceLock::new(),
		}
	}

	/// Current adaptive pressure decision, for metrics and focused tests.
	#[cfg(test)]
	pub(crate) fn pressure_decision(&self) -> crate::write_pressure::PressureDecision {
		self.pacer.lock().decision()
	}

	fn adaptive_delay(&self, counts: &StallCounts, write_bytes: u64) -> Duration {
		const ADAPTIVE_BASE_DELAY: Duration = Duration::from_micros(250);
		const ADAPTIVE_MAX_DELAY: Duration = Duration::from_millis(2);

		let pressure = self.provider.get_compaction_pressure();
		let memtable_load =
			counts.immutable_memtables as f64 / self.thresholds.memtable_soft_limit.max(1) as f64;
		let l0_file_load =
			counts.l0_files as f64 / self.thresholds.l0_file_soft_limit.max(1) as f64;
		let l0_byte_load = if pressure.l0_byte_target > 0 {
			pressure.l0_bytes as f64 / pressure.l0_byte_target as f64
		} else {
			0.0
		};
		let raw_load = memtable_load.max(l0_file_load).max(l0_byte_load);
		let measured_backlog =
			pressure.backlog_bytes.max(pressure.l0_bytes.saturating_add(pressure.immutable_bytes));

		let at = self.started_at.elapsed();
		let mut pacer = self.pacer.lock();
		let decision = pacer.observe(at, raw_load, measured_backlog);
		let throughput_delay = pacer.reserve_write_bytes(at, write_bytes, decision);
		let curve_delay = decision.pacing_delay(ADAPTIVE_BASE_DELAY, ADAPTIVE_MAX_DELAY);
		throughput_delay.max(curve_delay)
	}

	/// Sets what a stalled writer calls to ask for the work that ends its stall, such as the
	/// compaction of level 0. A writer asks each time it begins to wait, so a stall never depends
	/// on a wakeup that came, or was lost, before it.
	pub fn set_rearm(&self, rearm: impl Fn(WriteStallReason) + Send + Sync + 'static) {
		let _ = self.rearm.set(Box::new(rearm));
	}

	/// Check stall conditions and wait if stalled. Called before each write.
	///
	/// Returns `Ok(Some(WriteStallInfo))` if was stalled and then cleared,
	/// `Ok(None)` if not stalled, `Err(Error::PipelineStall)` on shutdown or when a failure is
	/// signalled while waiting.
	///
	/// Re-reads counts each iteration because background work (flushes,
	/// compactions) may complete while waiting.
	///
	/// IMPORTANT: The `Notified` future is created at the start of each loop
	/// iteration, BEFORE checking conditions. This ensures we receive any
	/// `notify_waiters()` calls that happen after we register but before we
	/// check. Per tokio docs: "The Notified future is guaranteed to receive
	/// wakeups from notify_waiters() as soon as it has been created."
	#[cfg(test)]
	pub async fn check(&self) -> Result<Option<WriteStallInfo>> {
		self.check_bytes(0).await
	}

	/// Checks backpressure and accounts the approximate bytes of the incoming commit.
	pub(crate) async fn check_bytes(&self, write_bytes: u64) -> Result<Option<WriteStallInfo>> {
		let mut stall_start: Option<Instant> = None;
		let mut stall_reason: Option<WriteStallReason> = None;
		let mut stall_value: usize = 0;
		let mut stall_threshold: usize = 0;
		let failures = self.failures.load(Ordering::Acquire);

		loop {
			// Create Notified FIRST to register for wakeups.
			// Any notify_waiters() call after this point will wake us.
			let notified = self.stall_cleared.notified();

			// Check shutdown, and a failure signalled since this writer began to wait
			if self.shutdown.load(Ordering::Acquire)
				|| self.failures.load(Ordering::Acquire) != failures
			{
				if stall_reason.is_some() {
					self.is_stalled.store(false, Ordering::Release);
				}
				return Err(Error::PipelineStall);
			}

			// Re-read counts (now any notify_waiters() after notified creation will wake us)
			let counts = self.provider.get_stall_counts();

			// Check if NOT hard-stalled
			if counts.immutable_memtables < self.thresholds.memtable_limit
				&& counts.l0_files < self.thresholds.l0_file_limit
			{
				// If we were stalled, clean up and return
				if let Some(reason) = stall_reason {
					self.is_stalled.store(false, Ordering::Release);
					let duration = stall_start.map(|s| s.elapsed()).unwrap_or(Duration::ZERO);
					tracing::debug!("Write stall cleared after {:?}", duration);
					return Ok(Some(WriteStallInfo {
						reason,
						current_value: stall_value,
						threshold: stall_threshold,
						duration,
					}));
				}

				// Proactive Write Pacing: apply microsecond delays if above soft limits
				let mut pacing_micros: u64 = 0;
				if counts.immutable_memtables > self.thresholds.memtable_soft_limit {
					let excess = counts.immutable_memtables - self.thresholds.memtable_soft_limit;
					pacing_micros = pacing_micros.max((excess as u64) * 200);
				}
				if counts.l0_files > self.thresholds.l0_file_soft_limit {
					let excess = counts.l0_files - self.thresholds.l0_file_soft_limit;
					pacing_micros = pacing_micros.max((excess as u64) * 100);
				}

				let legacy_delay = Duration::from_micros(pacing_micros);
				let pacing_delay = legacy_delay.max(self.adaptive_delay(&counts, write_bytes));
				let mut remaining_delay = pacing_delay;
				while remaining_delay > Duration::ZERO {
					let chunk = remaining_delay.min(Duration::from_millis(2));
					pace(chunk).await;
					remaining_delay = remaining_delay.saturating_sub(chunk);
				}

				return Ok(None);
			}

			// Stalled - determine which condition triggered it
			let (reason, value, threshold) =
				if counts.immutable_memtables >= self.thresholds.memtable_limit {
					(
						WriteStallReason::MemtableLimit,
						counts.immutable_memtables,
						self.thresholds.memtable_limit,
					)
				} else {
					(WriteStallReason::L0FileLimit, counts.l0_files, self.thresholds.l0_file_limit)
				};

			// Record stall reason if first time
			if stall_reason.is_none() {
				stall_reason = Some(reason);
				stall_value = value;
				stall_threshold = threshold;
				stall_start = Some(Instant::now());
				self.is_stalled.store(true, Ordering::Release);
				tracing::warn!("Write stall: {:?} ({} >= {})", reason, value, threshold);
			}

			// Ask for the work that ends the stall, then wait for a sign of it. Every flush and
			// compaction gives one, after which the writer looks again and, still stalled, asks
			// again. The wait needs no timer, so that a runtime built without the time driver does
			// not panic in the commit path.
			if let Some(rearm) = self.rearm.get() {
				rearm(reason);
			}
			notified.await;
		}
	}

	/// Signal that stall conditions may have changed.
	/// Called after flush or compaction completes.
	pub fn signal_work_done(&self) {
		self.stall_cleared.notify_waiters();
	}

	/// Signal shutdown - wakes all stalled writers to exit.
	pub fn signal_shutdown(&self) {
		self.shutdown.store(true, Ordering::Release);
		self.stall_cleared.notify_waiters();
	}

	/// Signal a failure that may pass - wakes the stalled writers to exit like `signal_shutdown`,
	/// but does not refuse the writers that come after.
	pub fn signal_failure(&self) {
		self.failures.fetch_add(1, Ordering::Release);
		self.stall_cleared.notify_waiters();
	}

	/// Fast check if currently stalled (for metrics).
	#[cfg(test)]
	pub fn is_stalled(&self) -> bool {
		self.is_stalled.load(Ordering::Acquire)
	}
}
