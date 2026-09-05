use super::{ResumeAction, RuntimeTarget};

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
                    core.run()?;
                }
            }
            (core_id, ResumeAction::Step) => {
                let mut core = session.core(core_id)?;
                core.step()?;
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
