use super::{ResumeAction, RuntimeTarget};
use crate::memory::MemoryInterface as _;

use gdbstub::target::ext::base::multithread::{
    MultiThreadResume, MultiThreadSchedulerLocking, MultiThreadSchedulerLockingOps,
    MultiThreadSingleStep, MultiThreadSingleStepOps,
};

impl MultiThreadResume for RuntimeTarget<'_> {
    fn resume(&mut self) -> Result<(), Self::Error> {
        let mut session = self.session.lock();

        match self.resume_action {
            (_, ResumeAction::Resume) => {
                for core_id in self.cores.iter() {
                    let mut core = session.core(*core_id)?;
                    if let Some(pc) = self.step_off_sw_breakpoint(&mut core)? {
                        // The step off the breakpoint already advanced the
                        // core; park dpc at the new address so run() does
                        // not single-step a second, extra instruction
                        // before resuming.
                        core.write_core_reg(core.program_counter(), pc)?;
                    }
                    core.run()?;
                }
            }
            (core_id, ResumeAction::Step) => {
                let mut core = session.core(core_id)?;
                if self.step_off_sw_breakpoint(&mut core)?.is_none() {
                    core.step()?;
                }
            }
            (_, ResumeAction::Unchanged) => {}
        }

        Ok(())
    }

    fn clear_resume_actions(&mut self) -> Result<(), Self::Error> {
        self.resume_action = (0, ResumeAction::Resume);

        Ok(())
    }

    fn set_resume_action_continue(
        &mut self,
        tid: gdbstub::common::Tid,
        _signal: Option<gdbstub::common::Signal>,
    ) -> Result<(), Self::Error> {
        let core_id = tid.get() - 1;
        self.resume_action = (core_id, ResumeAction::Resume);

        Ok(())
    }

    fn support_scheduler_locking(&mut self) -> Option<MultiThreadSchedulerLockingOps<'_, Self>> {
        Some(self)
    }

    fn support_single_step(&mut self) -> Option<MultiThreadSingleStepOps<'_, Self>> {
        Some(self)
    }
}

impl RuntimeTarget<'_> {
    /// Execute the real instruction under a software breakpoint the
    /// core is parked on: restore the original bytes, hardware-step
    /// once (a jump lands on its target), re-patch the breakpoint, and
    /// return the stepped-to PC. Returns None when no stub breakpoint
    /// sits at the core's PC - the caller steps or resumes directly.
    /// Skipping the instruction instead of executing it is only
    /// correct for non-branching instructions, which is why the
    /// breakpoint must actually run.
    fn step_off_sw_breakpoint(
        &self,
        core: &mut crate::Core,
    ) -> Result<Option<u64>, super::Error> {
        if self.sw_breakpoints.is_empty() {
            return Ok(None);
        }
        let pc: u64 = core.read_core_reg(core.program_counter())?;
        let Some(bp) = self.sw_breakpoint_at(pc) else {
            return Ok(None);
        };
        core.write(pc, &bp.saved[..bp.len])?;
        core.step()?;
        let new_pc: u64 = core.read_core_reg(core.program_counter())?;
        core.write(pc, &Self::breakpoint_bytes(&bp))?;
        Ok(Some(new_pc))
    }
}

impl MultiThreadSingleStep for RuntimeTarget<'_> {
    fn set_resume_action_step(
        &mut self,
        tid: gdbstub::common::Tid,
        _signal: Option<gdbstub::common::Signal>,
    ) -> Result<(), Self::Error> {
        let core_id = tid.get() - 1;
        self.resume_action = (core_id, ResumeAction::Step);

        Ok(())
    }
}

/// The stub groups all same-architecture cores into one target and its
/// resume path runs every core, so a locked "other thread" cannot be
/// held back. gdbstub 0.7.9 requires this IDET to answer GDB's
/// scheduler-locking request at all (stepping would otherwise tear down
/// the connection), hence the accept-and-ignore implementation; the
/// semantics of `set scheduler-locking on` are not honored on
/// multi-core targets.
impl MultiThreadSchedulerLocking for RuntimeTarget<'_> {
    fn set_resume_action_scheduler_lock(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}
