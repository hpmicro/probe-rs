mod base;
mod breakpoints;
mod desc;
mod flash;
mod monitor;
mod resume;
mod thread;
mod traits;
mod utils;

use super::arch::RuntimeArch;
use crate::flashing::FlashLoader;
use crate::{BreakpointCause, CoreStatus, Error, HaltReason, Session};
use gdbstub::stub::state_machine::GdbStubStateMachine;
use parking_lot::FairMutex;

use std::net::{SocketAddr, TcpListener, TcpStream};
use std::num::NonZeroUsize;
use std::time::Duration;

use gdbstub::common::Signal;
use gdbstub::conn::ConnectionExt;
use gdbstub::stub::{GdbStub, MultiThreadStopReason};
use gdbstub::target::ext::base::BaseOps;
use gdbstub::target::ext::breakpoints::BreakpointsOps;
use gdbstub::target::ext::flash::FlashOps;
use gdbstub::target::ext::memory_map::MemoryMapOps;
use gdbstub::target::ext::monitor_cmd::MonitorCmdOps;
use gdbstub::target::ext::target_description_xml_override::TargetDescriptionXmlOverrideOps;
use gdbstub::target::Target;

pub(crate) use traits::{GdbErrorExt, ProbeRsErrorExt};

use desc::TargetDescription;

/// Actions for resuming a core
#[derive(Debug, Copy, Clone)]
pub(crate) enum ResumeAction {
    /// Don't change the state
    Unchanged,
    /// Resume core
    Resume,
    /// Single step core
    Step,
}

/// The top level gdbstub target for a probe-rs debug session
pub(crate) struct RuntimeTarget<'a> {
    /// The probe-rs session object
    session: &'a FairMutex<Session>,
    /// A list of core IDs for this stub
    cores: Vec<usize>,
    /// Buffered flash data of an in-flight GDB `load` (the vFlashWrite
    /// packets); committed and cleared on vFlashDone. RuntimeTarget is
    /// reused across GDB connections, so a new connection also clears it.
    flash_loader: Option<FlashLoader>,
    /// Whether a vFlashErase was seen without any buffered write data —
    /// used to fail erase-only transactions (GDB `flash-erase`) loudly.
    saw_flash_erase: bool,
    /// Highest end address of the data buffered for the current GDB
    /// `load`; a subsequent write below it marks the start of a new
    /// load whose predecessor aborted without a vFlashDone.
    flash_high_water: Option<u64>,
    /// Software breakpoints this stub patched into target memory
    /// (original word per address). The stub owns them so it can
    /// restore the real instruction around stepping and resuming.
    sw_breakpoints: Vec<breakpoints::SwBreak>,

    /// TCP listener accepting incoming connections
    listener: TcpListener,
    /// The current GDB stub state machine
    gdb: Option<GdbStubStateMachine<'a, RuntimeTarget<'a>, TcpStream>>,
    /// Bytes read from the connection but not yet fed to the stub state
    /// machine (bulk-read ahead of per-byte processing).
    rx_buf: std::collections::VecDeque<u8>,
    /// Resume action to be used upon a continue request
    resume_action: (usize, ResumeAction),

    /// Description of target's architecture and registers
    target_desc: TargetDescription,
}

impl<'a> RuntimeTarget<'a> {
    /// Create a new RuntimeTarget and get ready to start processing GDB input
    pub fn new(
        session: &'a FairMutex<Session>,
        cores: Vec<usize>,
        addrs: &[SocketAddr],
    ) -> Result<Self, Error> {
        let listener = TcpListener::bind(addrs).into_error()?;
        listener.set_nonblocking(true).into_error()?;

        Ok(Self {
            session,
            cores,
            flash_loader: None,
            saw_flash_erase: false,
            flash_high_water: None,
            sw_breakpoints: Vec::new(),
            listener,
            gdb: None,
            rx_buf: std::collections::VecDeque::new(),
            resume_action: (0, ResumeAction::Unchanged),
            target_desc: TargetDescription::default(),
        })
    }

    /// Process any pending work for this target
    ///
    /// Returns: Duration to wait before processing this target again
    pub fn process(&mut self) -> Result<Duration, Error> {
        // State 1 - unconnected
        if self.gdb.is_none() {
            // See if we have a connection
            match self.listener.accept() {
                Ok((s, addr)) => {
                    tracing::info!("New connection from {:#?}", addr);

                    // A new GDB connection must not inherit uncommitted
                    // flash data buffered by an aborted load, nor the
                    // software breakpoints an aborted session patched
                    // into memory.
                    self.flash_loader = None;
                    self.saw_flash_erase = false;
                    self.flash_high_water = None;
                    self.clear_sw_breakpoints();

                    for i in 0..self.cores.len() {
                        let core_id = self.cores[i];
                        // When we first attach to the core, GDB expects us to halt the core, so we do this here when a new client connects.
                        // If the core is already halted, nothing happens if we issue a halt command again, so we always do this no matter of core state.
                        self.session
                            .lock()
                            .core(core_id)?
                            .halt(Duration::from_millis(100))?;

                        self.load_target_desc()?;
                    }

                    // Start the GDB Stub state machine. A large packet
                    // buffer lets GDB send bulk memory writes (the `load`
                    // X packets) in few large transfers instead of one
                    // round trip per few kilobytes.
                    let stub = match GdbStub::<RuntimeTarget, _>::builder(s)
                        .packet_buffer_size(64 * 1024)
                        .build()
                    {
                        Ok(stub) => stub,
                        Err(e) => return Err(anyhow::Error::from(e).into()),
                    };
                    match stub.run_state_machine(self) {
                        Ok(gdbstub) => {
                            self.gdb = Some(gdbstub);
                        }
                        Err(e) => {
                            // Any errors at this state are either IO errors or fatal config errors
                            return Err(anyhow::Error::from(e).into());
                        }
                    };
                }
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    // No connection yet
                    return Ok(Duration::from_millis(10));
                }
                Err(e) => {
                    // Fatal error
                    return Err(anyhow::Error::from(e).into());
                }
            };
        }

        // Stage 2 - connected
        if self.gdb.is_some() {
            let mut wait_time = Duration::ZERO;
            let gdb = self.gdb.take().unwrap();

            self.gdb = match gdb {
                GdbStubStateMachine::Idle(mut state) => {
                    // Bulk-read new data into the local buffer, then feed
                    // the state machine byte by byte from it: one socket
                    // read per chunk instead of one per byte, which is
                    // what large packets (GDB load writes) would
                    // otherwise pay tens of thousands of syscalls for.
                    if self.rx_buf.is_empty() {
                        let mut chunk = [0u8; 16384];
                        let n = read_chunk_if_available(state.borrow_conn(), &mut chunk)?;
                        self.rx_buf.extend(&chunk[..n]);
                    }

                    if self.rx_buf.is_empty() {
                        // Short wait: every round trip to GDB (an OK for a
                        // write packet) pays half this on average, and a
                        // load is a burst of such round trips.
                        wait_time = Duration::from_millis(2);
                        Some(state.into())
                    } else {
                        let mut current = GdbStubStateMachine::Idle(state);
                        while matches!(current, GdbStubStateMachine::Idle(_))
                            && !self.rx_buf.is_empty()
                        {
                            let b = self.rx_buf.pop_front().unwrap();
                            let GdbStubStateMachine::Idle(s) = current else {
                                unreachable!("matches! guarded above");
                            };
                            current = s.incoming_data(self, b).into_error()?;
                        }
                        Some(current)
                    }
                }
                GdbStubStateMachine::Running(mut state) => {
                    // Read data if available
                    let next_byte = {
                        let conn = state.borrow_conn();

                        read_if_available(conn)?
                    };

                    if let Some(b) = next_byte {
                        Some(state.incoming_data(self, b).into_error()?)
                    } else {
                        // Check for break
                        let mut stop_reason: Option<MultiThreadStopReason<u64>> = None;
                        {
                            let mut session = self.session.lock();

                            for i in &self.cores {
                                let mut core = session.core(*i)?;
                                let status = core.status()?;

                                if let CoreStatus::Halted(reason) = status {
                                    let tid = NonZeroUsize::new(i + 1).unwrap();
                                    stop_reason = Some(match reason {
                                        HaltReason::Breakpoint(BreakpointCause::Hardware)
                                        | HaltReason::Breakpoint(BreakpointCause::Unknown) => {
                                            // Some architectures do not allow us to distinguish between hardware and software breakpoints, so we just treat `Unknown` as hardware breakpoints.
                                            MultiThreadStopReason::HwBreak(tid)
                                        }
                                        HaltReason::Breakpoint(BreakpointCause::Software) => {
                                            // The architecture layer reports Software for
                                            // every ebreak halt, including ones in the
                                            // firmware itself; only an address this stub
                                            // patched is a GDB software breakpoint.
                                            let pc: u64 = core
                                                .read_core_reg(core.program_counter())
                                                .unwrap_or_default();
                                            if self.sw_breakpoint_at(pc).is_some() {
                                                MultiThreadStopReason::SwBreak(tid)
                                            } else {
                                                MultiThreadStopReason::SignalWithThread {
                                                    tid,
                                                    signal: Signal::SIGINT,
                                                }
                                            }
                                        }
                                        HaltReason::Step => MultiThreadStopReason::DoneStep,
                                        _ => MultiThreadStopReason::SignalWithThread {
                                            tid,
                                            signal: Signal::SIGINT,
                                        },
                                    });
                                    break;
                                }
                            }

                            // halt all remaining cores that are still running
                            // GDB expects all or nothing stops
                            if stop_reason.is_some() {
                                for i in &self.cores {
                                    let mut core = session.core(*i)?;
                                    if !core.core_halted()? {
                                        core.halt(Duration::from_millis(100))?;
                                    }
                                }
                            }
                        }

                        if let Some(reason) = stop_reason {
                            Some(state.report_stop(self, reason).into_error()?)
                        } else {
                            wait_time = Duration::from_millis(10);
                            Some(state.into())
                        }
                    }
                }
                GdbStubStateMachine::CtrlCInterrupt(state) => {
                    // Break core, handle interrupt
                    {
                        let mut session = self.session.lock();
                        for i in &self.cores {
                            let mut core = session.core(*i)?;

                            core.halt(Duration::from_millis(100))?;
                        }
                    }

                    Some(
                        state
                            .interrupt_handled(
                                self,
                                Some(MultiThreadStopReason::Signal(Signal::SIGINT)),
                            )
                            .into_error()?,
                    )
                }
                GdbStubStateMachine::Disconnected(state) => {
                    tracing::info!("GDB client disconnected: {:?}", state.get_reason());

                    // Restore the instructions the session's software
                    // breakpoints patched in, so the free-running target
                    // cannot trap on a stray ebreak.
                    self.clear_sw_breakpoints();

                    None
                }
            };

            return Ok(wait_time);
        }

        Ok(Duration::ZERO)
    }
}

impl Target for RuntimeTarget<'_> {
    type Arch = RuntimeArch;
    type Error = Error;

    fn base_ops(&mut self) -> BaseOps<'_, Self::Arch, Self::Error> {
        BaseOps::MultiThread(self)
    }

    fn support_target_description_xml_override(
        &mut self,
    ) -> Option<TargetDescriptionXmlOverrideOps<'_, Self>> {
        Some(self)
    }

    fn support_breakpoints(&mut self) -> Option<BreakpointsOps<'_, Self>> {
        Some(self)
    }

    fn support_memory_map(&mut self) -> Option<MemoryMapOps<'_, Self>> {
        Some(self)
    }

    fn support_monitor_cmd(&mut self) -> Option<MonitorCmdOps<'_, Self>> {
        Some(self)
    }

    fn support_flash_operations(&mut self) -> Option<FlashOps<'_, Self>> {
        Some(self)
    }

    fn guard_rail_implicit_sw_breakpoints(&self) -> bool {
        true
    }
}

/// Read a byte from a stream if available, otherwise return None
fn read_if_available(conn: &mut TcpStream) -> Result<Option<u8>, Error> {
    match conn.peek() {
        Ok(p) => {
            // Unwrap is safe because peek already showed
            // there's data in the buffer
            match p {
                Some(_) => conn.read().map(Some).into_error(),
                None => Ok(None),
            }
        }
        Err(e) => Err(anyhow::Error::from(e).into()),
    }
}

/// Read whatever the socket has buffered into `buf` without blocking.
/// Returns 0 when nothing is pending yet. One read per chunk instead of
/// per byte is what keeps large packets (GDB load writes) from paying
/// thousands of syscalls each.
///
/// A zero-byte read is the peer closing the connection: surfaced as an
/// error, not as "nothing pending", so the state machine tears the dead
/// connection down and the listener can accept a new one.
fn read_chunk_if_available(conn: &mut TcpStream, buf: &mut [u8]) -> Result<usize, Error> {
    match std::io::Read::read(conn, buf) {
        Ok(0) => Err(anyhow::Error::from(std::io::Error::new(
            std::io::ErrorKind::UnexpectedEof,
            "GDB client closed the connection",
        ))
        .into()),
        Ok(n) => Ok(n),
        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => Ok(0),
        Err(e) => Err(anyhow::Error::from(e).into()),
    }
}

/// A teardown (process exit, dropped stub) must not leave breakpoint
/// ebreaks patched in target memory: a later bare run of the target
/// would trap into debug mode with nothing attached to handle it.
impl Drop for RuntimeTarget<'_> {
    fn drop(&mut self) {
        self.clear_sw_breakpoints();
    }
}
