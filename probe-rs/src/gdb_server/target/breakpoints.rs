use super::{GdbErrorExt, RuntimeTarget};

use crate::architecture::riscv::assembly::{C_EBREAK, EBREAK};
use crate::memory::MemoryInterface as _;

use gdbstub::target::ext::breakpoints::{
    Breakpoints, HwBreakpoint, HwBreakpointOps, HwWatchpointOps, SwBreakpoint, SwBreakpointOps,
};

/// An active software breakpoint: the original instruction bytes at
/// `addr` (`len` of them, little-endian) that the breakpoint encoding
/// replaced. Byte granularity keeps a compressed breakpoint confined to
/// its own two bytes: a raw word access at a 2-mod-4 address is a hard
/// bus error on this class of targets, and patching whole words would
/// clobber the neighboring halfword.
pub(super) struct SwBreak {
    pub(super) addr: u64,
    pub(super) saved: [u8; 4],
    /// Size of the replaced instruction: 2 for compressed, 4 otherwise.
    pub(super) len: usize,
}

/// Low two bits of an uncompressed RISC-V instruction are always 0b11;
/// any other value marks a compressed halfword.
const UNCOMPRESSED_MARKER: u16 = 0b11;

impl Breakpoints for RuntimeTarget<'_> {
    fn support_sw_breakpoint(&mut self) -> Option<SwBreakpointOps<'_, Self>> {
        // The ebreak encodings this stub patches in are RISC-V specific;
        // offering Z0 to other architectures would corrupt their
        // instructions instead of trapping.
        let mut session = self.session.lock();
        match session.core(self.cores[0]).map(|core| core.architecture()) {
            Ok(crate::Architecture::Riscv) => Some(self),
            _ => None,
        }
    }

    fn support_hw_breakpoint(&mut self) -> Option<HwBreakpointOps<'_, Self>> {
        Some(self)
    }

    fn support_hw_watchpoint(&mut self) -> Option<HwWatchpointOps<'_, Self>> {
        None
    }
}

// The stub manages software breakpoints itself: the breakpoint
// instruction is patched into target memory here and restored around
// stepping/resuming, and reporting a SwBreak stop reason requires this
// extension to be implemented in the first place.
impl SwBreakpoint for RuntimeTarget<'_> {
    fn add_sw_breakpoint(
        &mut self,
        addr: u64,
        _kind: <Self::Arch as gdbstub::arch::Arch>::BreakpointKind,
    ) -> gdbstub::target::TargetResult<bool, Self> {
        if self.sw_breakpoint_at(addr).is_some() {
            // Already inserted; GDB re-inserting the same address must
            // not clobber the saved original bytes.
            return Ok(true);
        }

        // Any failure to read or patch the instruction means the
        // breakpoint cannot land (unwritable memory, bus error): report
        // that to GDB rather than tearing down the session.
        let mut session = self.session.lock();
        let Ok(mut core) = session.core(self.cores[0]) else {
            return Ok(false);
        };

        // Read the halfword at the address to decide whether the
        // instruction is compressed; the low two bits of an
        // uncompressed instruction are always 0b11.
        let mut half = [0u8; 2];
        if core.read(addr, &mut half).is_err() {
            return Ok(false);
        }
        let compressed = u16::from_le_bytes(half) & UNCOMPRESSED_MARKER != UNCOMPRESSED_MARKER;

        let (len, breakpoint_bytes) = if compressed {
            (2, C_EBREAK.to_le_bytes().to_vec())
        } else {
            (4, EBREAK.to_le_bytes().to_vec())
        };

        let mut saved = [0u8; 4];
        if core.read(addr, &mut saved[..len]).is_err() || core.write(addr, &breakpoint_bytes).is_err() {
            return Ok(false);
        }

        self.sw_breakpoints.push(SwBreak { addr, saved, len });

        Ok(true)
    }

    fn remove_sw_breakpoint(
        &mut self,
        addr: u64,
        _kind: <Self::Arch as gdbstub::arch::Arch>::BreakpointKind,
    ) -> gdbstub::target::TargetResult<bool, Self> {
        let Some(index) = self.sw_breakpoints.iter().position(|bp| bp.addr == addr) else {
            // Removing an unknown breakpoint leaves the target memory
            // untouched rather than failing the whole GDB transaction.
            return Ok(true);
        };
        let bp = self.sw_breakpoints.remove(index);

        let mut session = self.session.lock();
        let mut core = session.core(self.cores[0]).into_target_result()?;
        // A failing restore must not tear down the connection: the
        // breakpoint stays registered so the address remains known and
        // a later removal or new connection can retry.
        if restore_breakpoint(&mut core, &bp).is_err() {
            self.sw_breakpoints.push(bp);
            return Ok(false);
        }

        Ok(true)
    }
}

/// Write the original instruction bytes of `bp` back to target memory.
fn restore_breakpoint(core: &mut crate::Core, bp: &SwBreak) -> Result<(), crate::Error> {
    core.write(bp.addr, &bp.saved[..bp.len])
}

impl RuntimeTarget<'_> {
    /// The breakpoint active at `addr`, if any. The returned copy keeps
    /// the lookup cheap for the resume path, which runs on every step.
    pub(super) fn sw_breakpoint_at(&self, addr: u64) -> Option<SwBreak> {
        self.sw_breakpoints
            .iter()
            .find(|bp| bp.addr == addr)
            .map(|bp| SwBreak {
                addr: bp.addr,
                saved: bp.saved,
                len: bp.len,
            })
    }

    /// The breakpoint encoding for `bp`, to re-patch after stepping.
    pub(super) fn breakpoint_bytes(bp: &SwBreak) -> Vec<u8> {
        if bp.len == 2 {
            C_EBREAK.to_le_bytes().to_vec()
        } else {
            EBREAK.to_le_bytes().to_vec()
        }
    }

    /// Restore every active software breakpoint's original instruction
    /// and drop the table. Entries whose restore write fails are kept
    /// so a later connection can retry instead of leaving an ebreak
    /// patched in memory with its original bytes lost.
    pub(super) fn clear_sw_breakpoints(&mut self) {
        if self.sw_breakpoints.is_empty() {
            return;
        }
        let mut breakpoints = std::mem::take(&mut self.sw_breakpoints);
        if let Ok(mut core) = self.session.lock().core(self.cores[0]) {
            breakpoints.retain(|bp| restore_breakpoint(&mut core, bp).is_err());
        }
        self.sw_breakpoints = breakpoints;
    }
}

impl HwBreakpoint for RuntimeTarget<'_> {
    fn add_hw_breakpoint(
        &mut self,
        addr: u64,
        _kind: <Self::Arch as gdbstub::arch::Arch>::BreakpointKind,
    ) -> gdbstub::target::TargetResult<bool, Self> {
        let mut session = self.session.lock();
        for core_id in &self.cores {
            let mut core = session.core(*core_id).into_target_result()?;

            core.set_hw_breakpoint(addr).into_target_result()?;
        }

        Ok(true)
    }

    fn remove_hw_breakpoint(
        &mut self,
        addr: u64,
        _kind: <Self::Arch as gdbstub::arch::Arch>::BreakpointKind,
    ) -> gdbstub::target::TargetResult<bool, Self> {
        let mut session = self.session.lock();
        for core_id in &self.cores {
            let mut core = session.core(*core_id).into_target_result()?;

            core.clear_hw_breakpoint(addr).into_target_result()?;
        }

        Ok(true)
    }
}
