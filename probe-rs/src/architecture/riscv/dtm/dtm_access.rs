use crate::architecture::riscv::communication_interface::RiscvError;
use crate::probe::{CommandResult, DebugProbeError, DeferredResultIndex};
use std::fmt;
use std::time::Duration;

pub trait DtmAccess: Send + fmt::Debug {
    /// Perform interface-specific initialisation upon attaching.
    fn init(&mut self) -> Result<(), RiscvError> {
        Ok(())
    }

    /// Asserts a reset of the physical pins
    fn target_reset_assert(&mut self) -> Result<(), DebugProbeError>;

    /// Deasserts a reset of the physical pins
    fn target_reset_deassert(&mut self) -> Result<(), DebugProbeError>;

    /// Clear the sticky error state, if applicable
    fn clear_error_state(&mut self) -> Result<(), RiscvError>;

    /// Read previously scheduled `dmi` register accesses
    fn read_deferred_result(
        &mut self,
        index: DeferredResultIndex,
    ) -> Result<CommandResult, RiscvError>;

    /// Execute scheduled dmi accesses
    fn execute(&mut self) -> Result<(), RiscvError>;

    /// Schedule a write to an address on the `dmi` bus.
    ///
    /// Returns None if the underlying transport protocol does
    /// not return the value at the address on write
    fn schedule_write(
        &mut self,
        address: u64,
        value: u32,
    ) -> Result<Option<DeferredResultIndex>, RiscvError>;

    /// Schedule a read from an address on the `dmi` bus.
    fn schedule_read(&mut self, address: u64) -> Result<DeferredResultIndex, RiscvError>;

    /// Read an address on the `dmi` bus. If a busy value is returned, the access is
    /// retried until the transfer either succeeds, or the timeout expires.
    fn read_with_timeout(&mut self, address: u64, timeout: Duration) -> Result<u32, RiscvError>;

    /// Write an address to the `dmi` bus. If a busy value is returned, the access is
    /// retried until the transfer either succeeds, or the timeout expires.
    ///
    /// Returns None if the underlying protocol does not return the value on write
    fn write_with_timeout(
        &mut self,
        address: u64,
        value: u32,
        timeout: Duration,
    ) -> Result<Option<u32>, RiscvError>;

    /// Returns an idcode used for chip detection
    fn read_idcode(&mut self) -> Result<Option<u32>, DebugProbeError>;

    /// Schedule a DMI read whose response bypasses the standard
    /// status-check transform. The deferred result carries the raw scan
    /// capture as bytes; the caller extracts the op field (bits [1:0]) and
    /// the data field (bits [33:2]) itself. The returned index is captured,
    /// so the response of this scan is the result of the *previous* DMI
    /// operation, not of this read request.
    ///
    /// Intended for sequences that manage their own recovery protocol
    /// (e.g. the system-bus read pipeline experiments, where a busy
    /// status requires re-arming the address register rather than a
    /// DTM reset).
    #[doc(hidden)]
    fn schedule_raw_read(&mut self, _address: u64) -> Result<DeferredResultIndex, RiscvError> {
        Err(RiscvError::DebugProbe(DebugProbeError::NotImplemented {
            function_name: "schedule_raw_read",
        }))
    }

    /// Schedule a DMI no-op with a captured raw response, same contract as
    /// [`DtmAccess::schedule_raw_read`].
    #[doc(hidden)]
    fn schedule_raw_noop(&mut self) -> Result<DeferredResultIndex, RiscvError> {
        Err(RiscvError::DebugProbe(DebugProbeError::NotImplemented {
            function_name: "schedule_raw_noop",
        }))
    }

    /// Schedule a DMI read whose response is discarded (not captured). Used
    /// to push a request into the DMI pipeline when only its side effect
    /// (e.g. triggering a system-bus access) matters.
    #[doc(hidden)]
    fn schedule_raw_read_uncaptured(&mut self, _address: u64) -> Result<(), RiscvError> {
        Err(RiscvError::DebugProbe(DebugProbeError::NotImplemented {
            function_name: "schedule_raw_read_uncaptured",
        }))
    }

    /// Override the idle cycles injected between JTAG DR scans for the
    /// duration of a diagnostic sequence. Returns the previous value so
    /// callers restore it.
    #[doc(hidden)]
    fn set_debug_idle_cycles(&mut self, _idle: u8) -> Result<u8, RiscvError> {
        Err(RiscvError::DebugProbe(DebugProbeError::NotImplemented {
            function_name: "set_debug_idle_cycles",
        }))
    }
}
