//! Debug Module Communication
//!
//! This module implements communication with a
//! Debug Module, as described in the RISC-V debug
//! specification v0.13.2 .

use crate::architecture::riscv::dtm::dtm_access::DtmAccess;
use crate::{
    architecture::riscv::*, memory_mapped_bitfield_register, probe::CommandResult,
    probe::DeferredResultIndex, Error as ProbeRsError,
};
use std::any::Any;
use std::collections::HashMap;

/// Some error occurred when working with the RISC-V core.
#[derive(thiserror::Error, Debug)]
pub enum RiscvError {
    /// An error occurred during transport
    #[error("Error during transport")]
    DtmOperationFailed,
    /// DMI operation is in progress
    #[error("Transport operation in progress")]
    DtmOperationInProcess,
    /// An error with operating the debug probe occurred.
    #[error("Debug Probe Error")]
    DebugProbe(#[from] DebugProbeError),
    /// A timeout occurred during DMI access.
    #[error("Timeout during DMI access.")]
    Timeout,
    /// An error occurred during the execution of an abstract command.
    #[error("Error occurred during execution of an abstract command: {0:?}")]
    AbstractCommand(AbstractCommandErrorKind),
    /// The request for reset, resume or halt was not acknowledged.
    #[error("The core did not acknowledge a request for reset, resume or halt")]
    RequestNotAcknowledged,
    /// This debug transport module (DTM) version is currently not supported.
    #[error("The version '{0}' of the debug transport module (DTM) is currently not supported.")]
    UnsupportedDebugTransportModuleVersion(u8),
    /// This version of the debug module is not supported.
    #[error("The version '{0:?}' of the debug module is currently not supported.")]
    UnsupportedDebugModuleVersion(DebugModuleVersion),
    /// The provided csr address was invalid/unsupported
    #[error("CSR at address '{0:x}' is unsupported.")]
    UnsupportedCsrAddress(u16),
    /// The given program buffer register is not supported.
    #[error("Program buffer register '{0}' is currently not supported.")]
    UnsupportedProgramBufferRegister(usize),
    /// The program buffer is too small for the supplied program.
    #[error("Program buffer is too small for supplied program.")]
    ProgramBufferTooSmall,
    /// Memory width larger than 32 bits is not supported yet.
    #[error("Memory width larger than 32 bits is not supported yet.")]
    UnsupportedBusAccessWidth(RiscvBusAccess),
    /// An error during system bus access occurred.
    #[error("Error using system bus")]
    SystemBusAccess,
    /// The given trigger type is not available for the address breakpoint.
    #[error("Unexpected trigger type {0} for address breakpoint.")]
    UnexpectedTriggerType(u32),
    /// The connected target is not a RISC-V device.
    #[error("Connected target is not a RISC-V device.")]
    NoRiscvTarget,
    /// The target does not support halt after reset.
    #[error("The target does not support halt after reset.")]
    ResetHaltRequestNotSupported,
    /// The result index of a batched command is not available.
    #[error("The requested data is not available due to a previous error.")]
    BatchedResultNotAvailable,
    /// The hart is unavailable
    #[error("The requested hart is unavailable.")]
    HartUnavailable,
}

impl From<RiscvError> for ProbeRsError {
    fn from(err: RiscvError) -> Self {
        match err {
            RiscvError::DebugProbe(e) => e.into(),
            other => ProbeRsError::Riscv(other),
        }
    }
}

/// One raw DMI scan response, decoded from the capture: the 32-bit data
/// field (bits [33:2]) and the 2-bit op status (bits [1:0]) of the DMI
/// operation that the scan's predecessor requested. `raw` carries the
/// unmodified capture bytes for bit-level diagnosis.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct RawDmiResponse {
    pub data: u32,
    pub op: u8,
    pub raw: Vec<u8>,
}

/// Role of a captured scan within a system-bus read sequence: which
/// system-bus word its response carries, or why it carries none.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub enum ScanRole {
    /// Response holds `Sbdata0` for system-bus word #0 (the word at the
    /// sequence address), word #1, and so on.
    Data(usize),
    /// Response is the result of a filler no-op or a priming read and
    /// carries no meaningful data.
    Ignored,
}

/// A captured scan: its decoded response and its role in the sequence.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct ScanOutcome {
    pub response: RawDmiResponse,
    pub role: ScanRole,
}

/// Decode a raw DMI scan capture into its op and data fields. The 41-bit
/// DMI response places the op status in bits [1:0] and the data in
/// bits [33:2].
#[doc(hidden)]
pub fn decode_raw_response(result: CommandResult) -> Result<RawDmiResponse, RiscvError> {
    match result {
        CommandResult::VecU8(bytes) => {
            let value: u128 = bytes
                .iter()
                .enumerate()
                .fold(0u128, |acc, (i, b)| acc | ((*b as u128) << (8 * i)));
            Ok(RawDmiResponse {
                data: (value >> 2) as u32,
                op: (value & 0x3) as u8,
                raw: bytes,
            })
        }
        _ => Err(RiscvError::BatchedResultNotAvailable),
    }
}

/// Errors which can occur while executing an abstract command.
#[derive(Debug)]
pub enum AbstractCommandErrorKind {
    /// No error happened.
    None = 0,
    /// An abstract command was executing
    /// while command, `abstractcs`, or `abstractauto`
    /// was written, or when one of the `data` or `progbuf`
    /// registers was read or written. This status is only
    /// written if `cmderr` contains 0.
    Busy = 1,
    /// The requested command is not supported
    NotSupported = 2,
    /// An exception occurred while executing the command (e.g. while executing the Program Buffer).
    Exception = 3,
    /// The abstract command couldn’t
    /// execute because the hart wasn’t in the required
    /// state (running/halted), or unavailable.
    HaltResume = 4,
    /// The abstract command failed due to a
    /// bus error (e.g. alignment, access size, or timeout).
    Bus = 5,
    /// A reserved code. Should not occur.
    _Reserved = 6,
    /// The command failed for another reason.
    Other = 7,
}

impl AbstractCommandErrorKind {
    fn parse(value: u8) -> Self {
        use AbstractCommandErrorKind::*;

        match value {
            0 => None,
            1 => Busy,
            2 => NotSupported,
            3 => Exception,
            4 => HaltResume,
            5 => Bus,
            6 => _Reserved,
            7 => Other,
            _ => unreachable!("cmderr is a 3 bit value, values higher than 7 should not occur."),
        }
    }
}

/// List of all debug module versions.
///
/// The version of the debug module can be read from the version field of the `dmstatus`
/// register.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum DebugModuleVersion {
    /// There is no debug module present.
    NoModule,
    /// The debug module conforms to the version 0.11 of the RISC-V Debug Specification.
    Version0_11,
    /// The debug module conforms to the version 0.13 of the RISC-V Debug Specification.
    Version0_13,
    /// The debug module is present, but does not conform to any available version of the RISC-V Debug Specification.
    NonConforming,
    /// Unknown debug module version.
    Unknown(u8),
}

impl From<u8> for DebugModuleVersion {
    fn from(raw: u8) -> Self {
        match raw {
            0 => DebugModuleVersion::NoModule,
            1 => DebugModuleVersion::Version0_11,
            2 => DebugModuleVersion::Version0_13,
            15 => DebugModuleVersion::NonConforming,
            other => DebugModuleVersion::Unknown(other),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct CoreRegisterAbstractCmdSupport(u8);

impl CoreRegisterAbstractCmdSupport {
    const READ: Self = Self(1 << 0);
    const WRITE: Self = Self(1 << 1);
    const BOTH: Self = Self(Self::READ.0 | Self::WRITE.0);

    fn supports(&self, o: Self) -> bool {
        self.0 & o.0 == o.0
    }

    fn unset(&mut self, o: Self) {
        self.0 &= !(o.0);
    }
}

/// Save stack of a scratch register.
// TODO: we probably only need an Option, we don't seem to use scratch registers is nested situations.
#[derive(Debug, Default)]
struct ScratchState {
    stack: Vec<u32>,
}

impl ScratchState {
    fn push(&mut self, value: u32) {
        self.stack.push(value);
    }

    fn pop(&mut self) -> Option<u32> {
        self.stack.pop()
    }
}

/// A state to carry all the state data across multiple core switches in a session.
#[derive(Debug)]
pub struct RiscvCommunicationInterfaceState {
    /// Debug specification version
    debug_version: DebugModuleVersion,

    /// Size of the program buffer, in 32-bit words
    progbuf_size: u8,

    /// Cache for the program buffer.
    progbuf_cache: [u32; 16],

    /// Implicit `ebreak` instruction is present after the
    /// the program buffer.
    implicit_ebreak: bool,

    /// Number of data registers for abstract commands
    data_register_count: u8,

    /// Number of scratch registers
    nscratch: u8,

    /// Whether the target supports autoexecuting the program buffer
    supports_autoexec: bool,

    /// Pointer to the configuration string
    confstrptr: Option<u128>,

    /// Width of the hartsel register
    hartsellen: u8,

    /// Number of harts
    num_harts: u32,

    /// Describes, which memory access method should be used for a given access width
    memory_access_info: HashMap<RiscvBusAccess, MemoryAccessMethod>,

    /// describes, if the given register can be read / written with an
    /// abstract command
    abstract_cmd_register_info: HashMap<RegisterId, CoreRegisterAbstractCmdSupport>,

    /// First scratch register's state
    s0: ScratchState,

    /// Second scratch register's state
    s1: ScratchState,

    /// Bitfield of enabled harts
    enabled_harts: u32,

    /// The index of the last selected hart
    last_selected_hart: u32,

    /// Store the value of the `hasresethaltreq` bit of the `dmcstatus` register.
    hasresethaltreq: Option<bool>,
}

/// Timeout for RISC-V operations.
const RISCV_TIMEOUT: Duration = Duration::from_secs(5);

/// RiscV only supports 12bit CSRs. See
/// [Zicsr](https://riscv.org/wp-content/uploads/2019/06/riscv-spec.pdf#chapter.9) extension
const RISCV_MAX_CSR_ADDR: u16 = 0xFFF;

/// Abstract access-memory command (cmdtype 2) for a 32-bit read with
/// post-increment: cmdtype in bits [31:24], aamsize 2 (32-bit) in bits
/// [22:20], physical addressing (aamvirtual, bit 23, clear), and
/// aampostincrement at bit 19. The layout is pinned against the
/// [`AccessMemoryCommand`] field definitions by a unit test.
const ACCESS_MEMORY_READ: u32 = 2 << 24 | 2 << 20 | 1 << 19;

/// The same access-memory command with the write bit (bit 16) set.
const ACCESS_MEMORY_WRITE: u32 = ACCESS_MEMORY_READ | 1 << 16;

/// Words per program-buffer read batch. Bounding the batch keeps the
/// deferred queue depth compatible with the slowest load the program
/// buffer reads from: a flash load takes many more cycles than a RAM
/// load, and one giant queue overruns the debug module.
const WORDS_PER_BATCH: usize = 32;

/// Words per system-bus read batch (one DMI queue flush). Each word
/// costs one 41-bit scan; the driver splits a flush across USB
/// transfers and the per-sequence capture slicing keeps responses
/// aligned across the split, so this bound only trades memory against
/// recovery granularity.
const SYSBUS_BATCH_WORDS: usize = 64;

/// Polls of `sbcs.sbbusy` before a batch tail is declared stuck.
const SBBUSY_POLL_LIMIT: usize = 20;

/// Word layout of one system-bus read batch: how many words it covers
/// and where it starts, in output slots and in bytes from the range
/// start.
#[derive(Debug, PartialEq, Eq)]
struct SysbusBatchPlan {
    word_count: usize,
    word_offset: usize,
    address_offset_bytes: u32,
}

/// Split a system-bus read of `word_count` words into batches of at most
/// [`SYSBUS_BATCH_WORDS`] words, each with its output offset and its
/// byte distance from the range start.
fn sysbus_batch_partition(word_count: usize, byte_width: u32) -> Vec<SysbusBatchPlan> {
    let mut plans = Vec::new();
    let mut offset = 0usize;
    while offset < word_count {
        let take = (word_count - offset).min(SYSBUS_BATCH_WORDS);
        plans.push(SysbusBatchPlan {
            word_count: take,
            word_offset: offset,
            address_offset_bytes: (offset as u32) * byte_width,
        });
        offset += take;
    }
    plans
}

impl RiscvCommunicationInterfaceState {
    /// Create a new interface state.
    pub fn new() -> Self {
        RiscvCommunicationInterfaceState {
            // Set to the minimum here, will be set to the correct value below
            progbuf_size: 0,
            progbuf_cache: [0u32; 16],

            debug_version: DebugModuleVersion::NonConforming,

            // Assume the implicit ebreak is not present
            implicit_ebreak: false,

            // Set to the minimum here, will be set to the correct value below
            data_register_count: 1,

            nscratch: 0,

            supports_autoexec: false,

            confstrptr: None,

            // Assume maximum value, will be determined exactly alter.
            hartsellen: 20,

            // We assume only a singe hart exisits initially
            num_harts: 1,

            memory_access_info: HashMap::new(),

            abstract_cmd_register_info: HashMap::new(),

            s0: ScratchState::default(),
            s1: ScratchState::default(),
            enabled_harts: 0,
            last_selected_hart: 0,
            hasresethaltreq: None,
        }
    }

    /// Get the memory access method which should be used for an
    /// access with the specified width.
    fn memory_access_method(&mut self, access_width: RiscvBusAccess) -> MemoryAccessMethod {
        *self
            .memory_access_info
            .entry(access_width)
            .or_insert(MemoryAccessMethod::ProgramBuffer)
    }
}

impl Default for RiscvCommunicationInterfaceState {
    fn default() -> Self {
        Self::new()
    }
}

/// The combined state of a RISC-V debug module and its transport interface.
pub struct RiscvDebugInterfaceState {
    pub(super) interface_state: RiscvCommunicationInterfaceState,
    pub(super) dtm_state: Box<dyn Any + Send>,
}

impl RiscvDebugInterfaceState {
    pub(super) fn new(dtm_state: Box<dyn Any + Send>) -> Self {
        Self {
            interface_state: RiscvCommunicationInterfaceState::new(),
            dtm_state,
        }
    }
}

/// A single-use factory for creating RISC-V communication interfaces and their states.
pub trait RiscvInterfaceBuilder<'probe> {
    /// Creates a new RISC-V communication interface state object.
    ///
    /// The state object needs to be stored separately from the communication interface
    /// and can be used to restore the state of the interface at a later time.
    fn create_state(&self) -> RiscvDebugInterfaceState;

    /// Consumes the factory and creates a communication interface
    /// object initialised with the given state.
    fn attach<'state>(
        self: Box<Self>,
        state: &'state mut RiscvDebugInterfaceState,
    ) -> Result<RiscvCommunicationInterface<'state>, DebugProbeError>
    where
        'probe: 'state;
}

/// A interface that implements controls for RISC-V cores.
#[derive(Debug)]
pub struct RiscvCommunicationInterface<'state> {
    /// The Debug Transport Module (DTM) is used to
    /// communicate with the Debug Module on the target chip.
    dtm: Box<dyn DtmAccess + 'state>,
    state: &'state mut RiscvCommunicationInterfaceState,
}

impl<'state> RiscvCommunicationInterface<'state> {
    /// Creates a new RISC-V communication interface with a given probe driver.
    pub fn new(
        dtm: Box<dyn DtmAccess + 'state>,
        state: &'state mut RiscvCommunicationInterfaceState,
    ) -> Self {
        Self { dtm, state }
    }

    /// Select current hart
    pub fn select_hart(&mut self, hart: u32) -> Result<(), RiscvError> {
        if self.state.enabled_harts & (1 << hart) == 0 {
            return Err(RiscvError::HartUnavailable);
        }

        if self.state.last_selected_hart == hart {
            return Ok(());
        }

        let mut control: Dmcontrol = self.read_dm_register()?;
        control.set_dmactive(true);
        control.set_hartsel(hart);
        self.write_dm_register(control)?;
        self.state.last_selected_hart = hart;
        Ok(())
    }

    /// Overrides the memory access method used for every access width.
    ///
    /// The method is normally chosen from the debug module's reported
    /// capabilities. Forcing it is meant for diagnostics and performance
    /// tooling that needs to compare the available transport methods on
    /// the same target. Passing `None` clears any previously forced
    /// choice; the default (program buffer) then applies until the next
    /// attach re-probes the debug module's capabilities.
    pub fn force_memory_access_method(&mut self, method: Option<MemoryAccessMethod>) {
        match method {
            Some(method) => {
                for width in [
                    RiscvBusAccess::A8,
                    RiscvBusAccess::A16,
                    RiscvBusAccess::A32,
                    RiscvBusAccess::A64,
                    RiscvBusAccess::A128,
                ] {
                    self.state.memory_access_info.insert(width, method);
                }
            }
            None => self.state.memory_access_info.clear(),
        }
    }

    /// Check if the given hart is enabled
    pub fn hart_enabled(&self, hart: u32) -> bool {
        self.state.enabled_harts & (1 << hart) != 0
    }

    /// Assert the target reset
    pub fn target_reset_assert(&mut self) -> Result<(), DebugProbeError> {
        self.dtm.target_reset_assert()
    }

    /// Deassert the target reset.
    pub fn target_reset_deassert(&mut self) -> Result<(), DebugProbeError> {
        self.dtm.target_reset_deassert()
    }

    /// Read the targets idcode used as hint for chip detection
    pub fn read_idcode(&mut self) -> Result<Option<u32>, DebugProbeError> {
        self.dtm.read_idcode()
    }

    fn save_s0(&mut self) -> Result<bool, RiscvError> {
        let s0 = self.abstract_cmd_register_read(&registers::S0)?;

        self.state.s0.push(s0);

        Ok(true)
    }

    fn restore_s0(&mut self, saved: bool) -> Result<(), RiscvError> {
        if saved {
            let s0 = self.state.s0.pop().unwrap();

            self.abstract_cmd_register_write(&registers::S0, s0)?;
        }

        Ok(())
    }

    fn save_s1(&mut self) -> Result<bool, RiscvError> {
        let s1 = self.abstract_cmd_register_read(&registers::S1)?;

        self.state.s1.push(s1);

        Ok(true)
    }

    fn restore_s1(&mut self, saved: bool) -> Result<(), RiscvError> {
        if saved {
            let s1 = self.state.s1.pop().unwrap();

            self.abstract_cmd_register_write(&registers::S1, s1)?;
        }

        Ok(())
    }

    pub(crate) fn enter_debug_mode(&mut self) -> Result<(), RiscvError> {
        tracing::debug!("Building RISC-V interface");
        self.dtm.init()?;

        // Reset error bits from previous connections
        self.dtm.clear_error_state()?;

        // enable the debug module
        let mut control = Dmcontrol(0);
        control.set_dmactive(true);
        self.write_dm_register(control)?;

        // read the  version of the debug module
        let status: Dmstatus = self.read_dm_register()?;

        self.state.progbuf_cache.fill(0);
        self.state.debug_version = DebugModuleVersion::from(status.version() as u8);

        // Only version of 0.13 of the debug specification is currently supported.
        if self.state.debug_version != DebugModuleVersion::Version0_13 {
            return Err(RiscvError::UnsupportedDebugModuleVersion(
                self.state.debug_version,
            ));
        }

        self.state.implicit_ebreak = status.impebreak();

        // check if the configuration string pointer is valid, and retrieve it, if valid
        self.state.confstrptr = if status.confstrptrvalid() {
            let confstrptr_0: Confstrptr0 = self.read_dm_register()?;
            let confstrptr_1: Confstrptr1 = self.read_dm_register()?;
            let confstrptr_2: Confstrptr2 = self.read_dm_register()?;
            let confstrptr_3: Confstrptr3 = self.read_dm_register()?;
            let confstrptr = (u32::from(confstrptr_0) as u128)
                | (u32::from(confstrptr_1) as u128) << 8
                | (u32::from(confstrptr_2) as u128) << 16
                | (u32::from(confstrptr_3) as u128) << 32;
            Some(confstrptr)
        } else {
            None
        };

        tracing::debug!("dmstatus: {:?}", status);

        // Select all harts to determine the width
        // of the hartsel register.
        let mut control = Dmcontrol(0);
        control.set_dmactive(true);
        control.set_hartsel(0xffff_ffff);

        self.write_dm_register(control)?;

        let control: Dmcontrol = self.read_dm_register()?;

        self.state.hartsellen = control.hartsel().count_ones() as u8;

        tracing::debug!("HARTSELLEN: {}", self.state.hartsellen);

        // Determine number of harts

        let max_hart_index = 2u32.pow(self.state.hartsellen as u32);

        // Hart 0 exists on every chip
        let mut num_harts = 1;
        self.state.enabled_harts = 1;

        // Check if anynonexistent is avaliable.
        // Some chips that have only one hart do not implement anynonexistent and allnonexistent.
        // So let's check max hart index to see if we can use it reliably,
        // or else we will assume only one hart exists.
        let mut control = Dmcontrol(0);
        control.set_dmactive(true);
        control.set_hartsel(max_hart_index - 1);
        self.write_dm_register(control)?;

        // Check if the anynonexistent works
        let status: Dmstatus = self.read_dm_register()?;

        if status.anynonexistent() {
            for hart_index in 1..max_hart_index {
                let mut control = Dmcontrol(0);
                control.set_dmactive(true);
                control.set_hartsel(hart_index);

                self.write_dm_register(control)?;

                // Check if the current hart exists
                let status: Dmstatus = self.read_dm_register()?;

                if status.anynonexistent() {
                    break;
                }

                if !status.allunavail() {
                    self.state.enabled_harts |= 1 << num_harts;
                }

                num_harts += 1;
            }
        } else {
            tracing::debug!("anynonexistent not supported, assuming only one hart exists")
        }

        tracing::debug!("Number of harts: {}", num_harts);

        self.state.num_harts = num_harts;

        // Select hart 0 again - assuming all harts are same in regards of discovered features
        let mut control = Dmcontrol(0);
        control.set_dmactive(true);
        control.set_hartsel(0);

        self.write_dm_register(control)?;

        // determine size of the program buffer, and number of data
        // registers for abstract commands
        let abstractcs: Abstractcs = self.read_dm_register()?;

        self.state.progbuf_size = abstractcs.progbufsize() as u8;
        tracing::debug!("Program buffer size: {}", self.state.progbuf_size);

        self.state.data_register_count = abstractcs.datacount() as u8;
        tracing::debug!(
            "Number of data registers: {}",
            self.state.data_register_count
        );

        // determine more information about hart
        let hartinfo: Hartinfo = self.read_dm_register()?;

        self.state.nscratch = hartinfo.nscratch() as u8;
        tracing::debug!("Number of dscratch registers: {}", self.state.nscratch);

        // determine if autoexec works
        let mut abstractauto = Abstractauto(0);
        abstractauto.set_autoexecprogbuf(2u32.pow(self.state.progbuf_size as u32) - 1);
        abstractauto.set_autoexecdata(2u32.pow(self.state.data_register_count as u32) - 1);

        self.write_dm_register(abstractauto)?;

        let abstractauto_readback: Abstractauto = self.read_dm_register()?;

        self.state.supports_autoexec = abstractauto_readback == abstractauto;
        tracing::debug!("Support for autoexec: {}", self.state.supports_autoexec);

        // clear abstractauto
        abstractauto = Abstractauto(0);
        self.write_dm_register(abstractauto)?;

        // determine support system bus access
        let sbcs = self.read_dm_register::<Sbcs>()?;

        // Only version 1 is supported, this means that
        // the system bus access conforms to the debug
        // specification 13.2.
        if sbcs.sbversion() == 1 {
            self.clear_system_bus_errors()?;

            // When possible, we use system bus access for memory access

            if sbcs.sbaccess8() {
                self.state
                    .memory_access_info
                    .insert(RiscvBusAccess::A8, MemoryAccessMethod::SystemBus);
            }

            if sbcs.sbaccess16() {
                self.state
                    .memory_access_info
                    .insert(RiscvBusAccess::A16, MemoryAccessMethod::SystemBus);
            }

            if sbcs.sbaccess32() {
                self.state
                    .memory_access_info
                    .insert(RiscvBusAccess::A32, MemoryAccessMethod::SystemBus);
            }

            if sbcs.sbaccess64() {
                self.state
                    .memory_access_info
                    .insert(RiscvBusAccess::A64, MemoryAccessMethod::SystemBus);
            }

            if sbcs.sbaccess128() {
                self.state
                    .memory_access_info
                    .insert(RiscvBusAccess::A128, MemoryAccessMethod::SystemBus);
            }
        } else {
            tracing::debug!(
                "System bus interface version {} is not supported.",
                sbcs.sbversion()
            );
        }

        Ok(())
    }

    pub(crate) fn halt(&mut self, timeout: Duration) -> Result<CoreInformation, RiscvError> {
        // write 1 to the haltreq register, which is part
        // of the dmcontrol register

        let mut dmcontrol: Dmcontrol = self.read_dm_register()?;
        tracing::debug!(
            "Before requesting halt, the Dmcontrol register value was: {:?}",
            dmcontrol
        );

        dmcontrol.set_dmactive(true);
        dmcontrol.set_haltreq(true);

        self.write_dm_register(dmcontrol)?;

        self.wait_for_core_halted(timeout)?;

        // clear the halt request
        dmcontrol.set_haltreq(false);

        self.write_dm_register(dmcontrol)?;

        let pc: u64 = self
            .read_csr(super::registers::PC.id().0)
            .map(|v| v.into())?;

        Ok(CoreInformation { pc })
    }

    pub(crate) fn wait_for_core_halted(&mut self, timeout: Duration) -> Result<(), RiscvError> {
        let start = Instant::now();

        while start.elapsed() < timeout {
            let dmstatus: Dmstatus = self.read_dm_register()?;

            tracing::trace!("{:?}", dmstatus);

            if dmstatus.allhalted() {
                return Ok(());
            }
        }

        Err(RiscvError::Timeout)
    }

    pub(super) fn read_csr(&mut self, address: u16) -> Result<u32, RiscvError> {
        // We need to use the "Access Register Command",
        // which has cmdtype 0

        // write needs to be clear
        // transfer has to be set

        tracing::debug!("Reading CSR {:#x}", address);

        // always try to read register with abstract command, fallback to program buffer,
        // if not supported
        match self.abstract_cmd_register_read(address) {
            Err(RiscvError::AbstractCommand(AbstractCommandErrorKind::NotSupported)) => {
                tracing::debug!("Could not read core register {:#x} with abstract command, falling back to program buffer", address);
                self.read_csr_progbuf(address)
            }
            other => other,
        }
    }

    /// Schedules a DM register read, flushes the queue and returns the result.
    pub(crate) fn read_dm_register<R: MemoryMappedRegister<u32>>(
        &mut self,
    ) -> Result<R, RiscvError> {
        tracing::debug!(
            "Reading DM register '{}' at {:#010x}",
            R::NAME,
            R::get_mmio_address()
        );

        let register_value = self.read_dm_register_untyped(R::get_mmio_address())?.into();

        tracing::debug!(
            "Read DM register '{}' at {:#010x} = {:x?}",
            R::NAME,
            R::get_mmio_address(),
            register_value
        );

        Ok(register_value)
    }

    /// Schedules a DM register read, flushes the queue and returns the untyped result.
    ///
    /// Use the [`read_dm_register`] function if possible.
    fn read_dm_register_untyped(&mut self, address: u64) -> Result<u32, RiscvError> {
        self.dtm.read_with_timeout(address, Duration::from_millis(1000))
    }

    pub(crate) fn write_dm_register<R: MemoryMappedRegister<u32>>(
        &mut self,
        register: R,
    ) -> Result<(), RiscvError> {
        // write write command to dmi register

        tracing::debug!(
            "Write DM register '{}' at {:#010x} = {:x?}",
            R::NAME,
            R::get_mmio_address(),
            register
        );

        self.write_dm_register_untyped(R::get_mmio_address(), register.into())
    }

    /// Write to a DM register
    ///
    /// Use the [`write_dm_register`] function if possible.
    fn write_dm_register_untyped(&mut self, address: u64, value: u32) -> Result<(), RiscvError> {
        self.dtm.write_with_timeout(address, value, RISCV_TIMEOUT)?;

        Ok(())
    }

    fn schedule_write_progbuf(&mut self, index: usize, value: u32) -> Result<(), RiscvError> {
        match index {
            0 => self.schedule_write_dm_register(Progbuf0(value)),
            1 => self.schedule_write_dm_register(Progbuf1(value)),
            2 => self.schedule_write_dm_register(Progbuf2(value)),
            3 => self.schedule_write_dm_register(Progbuf3(value)),
            4 => self.schedule_write_dm_register(Progbuf4(value)),
            5 => self.schedule_write_dm_register(Progbuf5(value)),
            6 => self.schedule_write_dm_register(Progbuf6(value)),
            7 => self.schedule_write_dm_register(Progbuf7(value)),
            8 => self.schedule_write_dm_register(Progbuf8(value)),
            9 => self.schedule_write_dm_register(Progbuf9(value)),
            10 => self.schedule_write_dm_register(Progbuf10(value)),
            11 => self.schedule_write_dm_register(Progbuf11(value)),
            12 => self.schedule_write_dm_register(Progbuf12(value)),
            13 => self.schedule_write_dm_register(Progbuf13(value)),
            14 => self.schedule_write_dm_register(Progbuf14(value)),
            15 => self.schedule_write_dm_register(Progbuf15(value)),
            e => Err(RiscvError::UnsupportedProgramBufferRegister(e)),
        }
    }

    pub(crate) fn schedule_setup_program_buffer(&mut self, data: &[u32]) -> Result<(), RiscvError> {
        let required_len = if self.state.implicit_ebreak {
            data.len()
        } else {
            data.len() + 1
        };

        if required_len > self.state.progbuf_size as usize {
            return Err(RiscvError::ProgramBufferTooSmall);
        }

        if data == &self.state.progbuf_cache[..data.len()] {
            // Check if we actually have to write the program buffer
            tracing::debug!("Program buffer is up-to-date, skipping write.");
            return Ok(());
        }

        for (index, word) in data.iter().enumerate() {
            self.schedule_write_progbuf(index, *word)?;
        }

        // Add manual ebreak if necessary.
        //
        // This is necessary when we either don't need the full program buffer,
        // or if there is no implict ebreak after the last program buffer word.
        if !self.state.implicit_ebreak || data.len() < self.state.progbuf_size as usize {
            self.schedule_write_progbuf(data.len(), assembly::EBREAK)?;
        }

        // Update the cache
        self.state.progbuf_cache[..data.len()].copy_from_slice(data);

        Ok(())
    }

    /// Perform a single read from a memory location, using system bus access.

    /// Clear the sticky system-bus error bits. Both `sberror` and
    /// `sbbusyerror` survive until written back with ones, and with
    /// them set every further system bus access fails before it starts
    /// - so errors are cleared both before first use and before they
    /// are reported.
    fn clear_system_bus_errors(&mut self) -> Result<(), RiscvError> {
        let mut clear = Sbcs(0);
        clear.set_sberror(0b111);
        clear.set_sbbusyerror(true);
        self.write_dm_register(clear)
    }

    fn perform_memory_read_sysbus<V: RiscvValue32>(
        &mut self,
        address: u32,
    ) -> Result<V, RiscvError> {
        let mut sbcs = Sbcs(0);

        sbcs.set_sbaccess(V::WIDTH as u32);
        sbcs.set_sbreadonaddr(true);

        self.schedule_write_dm_register(sbcs)?;
        self.schedule_write_dm_register(Sbaddress0(address))?;

        let data = self.read_large_dtm_register::<V, Sbdata>()?;

        // Check that the read was succesful
        let sbcs = self.read_dm_register::<Sbcs>()?;

        if sbcs.sberror() != 0 {
            self.clear_system_bus_errors()?;
            Err(RiscvError::SystemBusAccess)
        } else {
            Ok(data)
        }
    }

    /// Number of DMI scans a system-bus batch read of `word_count` words
    /// queues: one priming read, `word_count - 2` captured reads, and one
    /// captured no-op tail. The last word of the range is collected after
    /// the batch (engine stopped), so a single word needs no scan at all.
    fn sysbus_batch_scan_count(word_count: usize) -> usize {
        match word_count {
            0 | 1 => 0,
            n => n,
        }
    }

    /// Fill `data` from captured scan outcomes by scan role. Any non-Ok
    /// op status invalidates the pipeline from that scan on (a stalled
    /// scan shifts every later response), so it is an error rather than
    /// skipped data.
    fn sysbus_batch_data_from_scans<V: RiscvValue32>(
        scans: &[ScanOutcome],
        data: &mut [V],
    ) -> Result<(), RiscvError> {
        for scan in scans {
            if let ScanRole::Data(index) = scan.role {
                if scan.response.op != 0 {
                    return Err(RiscvError::SystemBusAccess);
                }
                data[index] = V::from_register_value(scan.response.data);
            }
        }
        Ok(())
    }

    /// Perform multiple reads from consecutive memory locations
    /// using system bus access.
    /// Only reads up to a width of 32 bits are currently supported.
    ///
    /// The words are read as one `sbreadondata` pipeline per DMI queue
    /// flush: an uncaptured priming read pushes the first access result
    /// into the DMI pipeline, captured read scans carry consecutive words,
    /// and a captured no-op tail ends the batch without triggering an
    /// access past the range. The final word is collected after the
    /// engine is stopped, so a range read performs exactly one bus access
    /// per word.
    fn perform_memory_read_multiple_sysbus<V: RiscvValue32>(
        &mut self,
        address: u32,
        data: &mut [V],
    ) -> Result<(), RiscvError> {
        if data.is_empty() {
            return Ok(());
        }

        let byte_width = V::WIDTH.byte_width() as u32;
        for plan in sysbus_batch_partition(data.len(), byte_width) {
            let batch = &mut data[plan.word_offset..plan.word_offset + plan.word_count];

            // Re-arm the pipeline for every batch: the previous batch's
            // tail collection left readondata and autoincrement off.
            let mut sbcs = Sbcs(0);
            sbcs.set_sbaccess(V::WIDTH as u32);
            sbcs.set_sbreadonaddr(true);
            sbcs.set_sbreadondata(true);
            sbcs.set_sbautoincrement(true);
            self.write_dm_register(sbcs)?;

            // The address write triggers access #1 of this batch.
            self.write_dm_register(Sbaddress0(address + plan.address_offset_bytes))?;

            let scans_needed = Self::sysbus_batch_scan_count(batch.len());
            if scans_needed > 0 {
                // Priming read: its own response is the address-write
                // ack; it pushes the first word into the DMI pipeline.
                self.dtm
                    .schedule_raw_read_uncaptured(Sbdata::R0_ADDRESS as u64)?;

                // Scan k (1-based) carries data[k - 1]. Scans before the
                // tail are reads (each triggers the next access); the
                // final scan is a captured no-op so the batch ends with-
                // out triggering an access beyond the range.
                let tail = scans_needed - 1;
                let mut indices = Vec::with_capacity(tail);
                for k in 1..=tail {
                    let index = if k < tail {
                        self.dtm.schedule_raw_read(Sbdata::R0_ADDRESS as u64)?
                    } else {
                        self.dtm.schedule_raw_noop()?
                    };
                    indices.push((index, ScanRole::Data(k - 1)));
                }

                self.dtm.execute()?;

                let fill_len = batch.len() - 1;
                if let Err(e) = self.sysbus_collect_scans::<V>(indices, &mut batch[..fill_len]) {
                    // Leave the engine safe for a caller retry: wait out
                    // sbbusy, then turn the access-triggering bits off.
                    // A retry writing sbcs while the engine is busy is
                    // undefined behavior per the SBCS contract.
                    self.sysbus_abandon_batch(V::WIDTH as u32)?;
                    return Err(e);
                }
            }

            // Two-phase tail: the last access of the batch was triggered
            // by the final read scan; wait it out, stop the engine so
            // reading SBDATA0 triggers nothing, then collect the word.
            let mut status = self.read_dm_register::<Sbcs>()?;
            let mut polls = 0;
            while status.sbbusy() && polls < SBBUSY_POLL_LIMIT {
                status = self.read_dm_register::<Sbcs>()?;
                polls += 1;
            }
            if status.sbbusy() {
                return Err(RiscvError::SystemBusAccess);
            }

            let mut stop = Sbcs(0);
            stop.set_sbaccess(V::WIDTH as u32);
            stop.set_sbreadondata(false);
            stop.set_sbautoincrement(false);
            self.write_dm_register(stop)?;

            batch[batch.len() - 1] = self.read_large_dtm_register::<V, Sbdata>()?;
        }

        // Final status check; both sticky bits are cleared on error so a
        // failed read does not poison later system-bus work.
        let final_status: Sbcs = self.read_dm_register()?;
        if final_status.sberror() != 0 || final_status.sbbusyerror() {
            let mut clear = Sbcs(0);
            clear.set_sberror(final_status.sberror());
            clear.set_sbbusyerror(final_status.sbbusyerror());
            self.write_dm_register(clear)?;
            return Err(RiscvError::SystemBusAccess);
        }
        Ok(())
    }

    /// Collect a batch's deferred scan responses and lay the decoded
    /// words into `batch`. Errors surface a stalled or failed pipeline
    /// member instead of shifted data.
    fn sysbus_collect_scans<V: RiscvValue32>(
        &mut self,
        indices: Vec<(DeferredResultIndex, ScanRole)>,
        batch: &mut [V],
    ) -> Result<(), RiscvError> {
        let mut scans = Vec::with_capacity(indices.len());
        for (index, role) in indices {
            let response = decode_raw_response(self.dtm.read_deferred_result(index)?)?;
            scans.push(ScanOutcome { response, role });
        }
        Self::sysbus_batch_data_from_scans(&scans, batch)
    }

    /// Leave the system-bus engine in a safe state after a failed batch:
    /// wait (bounded) for sbbusy to clear, then turn the access-triggering
    /// bits off. Writing sbcs while the engine is busy is undefined
    /// behavior per the SBCS contract, so a wedged engine is reported
    /// instead of being written against.
    fn sysbus_abandon_batch(&mut self, width: u32) -> Result<(), RiscvError> {
        let mut status = self.read_dm_register::<Sbcs>()?;
        let mut polls = 0;
        while status.sbbusy() && polls < SBBUSY_POLL_LIMIT {
            status = self.read_dm_register()?;
            polls += 1;
        }
        if status.sbbusy() {
            return Err(RiscvError::SystemBusAccess);
        }

        let mut stop = Sbcs(0);
        stop.set_sbaccess(width);
        stop.set_sbreadondata(false);
        stop.set_sbautoincrement(false);
        self.write_dm_register(stop)
    }

    /// Perform memory reads using the abstract access-memory command
    /// (command type 2) with auto-increment. Only 32-bit accesses are
    /// implemented; narrower widths fail loudly here rather than
    /// degrading into a wrong-width access. The address is loaded into
    /// arg1 once and each command reads the next word into data0.
    fn perform_memory_read_multiple_abstract<V: RiscvValue32>(
        &mut self,
        address: u32,
        data: &mut [V],
    ) -> Result<(), RiscvError> {
        if V::WIDTH != RiscvBusAccess::A32 {
            return Err(RiscvError::UnsupportedBusAccessWidth(V::WIDTH));
        }

        self.write_dm_register(Data1(address))?;

        for slot in data.iter_mut() {
            self.execute_abstract_command(ACCESS_MEMORY_READ)?;
            let word: Data0 = self.read_dm_register()?;
            *slot = V::from_register_value(word.0);
        }

        Ok(())
    }

    /// Perform memory writes using the abstract access-memory command
    /// with auto-increment: data goes through data0, the address through
    /// arg1. As with reads, only 32-bit widths are implemented and
    /// narrower widths fail loudly.
    fn perform_memory_write_multiple_abstract<V: RiscvValue32>(
        &mut self,
        address: u32,
        data: &[V],
    ) -> Result<(), RiscvError> {
        if V::WIDTH != RiscvBusAccess::A32 {
            return Err(RiscvError::UnsupportedBusAccessWidth(V::WIDTH));
        }

        self.write_dm_register(Data1(address))?;

        for value in data {
            self.write_dm_register(Data0((*value).into()))?;
            self.execute_abstract_command(ACCESS_MEMORY_WRITE)?;
        }

        Ok(())
    }

    /// Perform memory read from a single location using the program buffer.
    /// Only reads up to a width of 32 bits are currently supported.
    fn perform_memory_read_progbuf<V: RiscvValue32>(
        &mut self,
        address: u32,
    ) -> Result<V, RiscvError> {
        // assemble
        //  lb s1, 0(s0)

        let s0 = self.save_s0()?;

        let lw_command = assembly::lw(0, 8, V::WIDTH as u8, 8);

        self.schedule_setup_program_buffer(&[lw_command])?;

        self.schedule_write_dm_register(Data0(address))?;

        // Write s0, then execute program buffer
        let mut command = AccessRegisterCommand(0);
        command.set_cmd_type(0);
        command.set_transfer(true);
        command.set_write(true);

        // registers are 32 bit, so we have size 2 here
        command.set_aarsize(RiscvBusAccess::A32);
        command.set_postexec(true);

        // register s0, ie. 0x1008
        command.set_regno((registers::S0).id.0 as u32);

        self.schedule_write_dm_register(command)?;

        let abstractcs_idx = self.schedule_read_dm_register::<Abstractcs>()?;

        // Read back s0
        let value = self.abstract_cmd_register_read(&registers::S0)?;

        let abstractcs = Abstractcs(self.dtm.read_deferred_result(abstractcs_idx)?.into_u32());
        if abstractcs.cmderr() != 0 {
            return Err(RiscvError::AbstractCommand(
                AbstractCommandErrorKind::parse(abstractcs.cmderr() as u8),
            ));
        }

        // Restore s0 register
        self.restore_s0(s0)?;

        Ok(V::from_register_value(value))
    }

    fn perform_memory_read_multiple_progbuf<V: RiscvValue32>(
        &mut self,
        address: u32,
        data: &mut [V],
        timeout: Duration,
    ) -> Result<(), RiscvError> {
        // Backup registers s0 and s1
        let s0 = self.save_s0()?;
        let s1 = self.save_s1()?;

        // Load a word from address in register 8 (S0), with offset 0, into register 9 (S9)
        let lw_command: u32 = assembly::lw(0, 8, V::WIDTH as u8, 9);

        self.schedule_setup_program_buffer(&[
            lw_command,
            assembly::addi(8, 8, V::WIDTH.byte_width() as i16),
        ])?;

        self.schedule_write_dm_register(Data0(address))?;

        // Write s0, then execute program buffer
        let mut command = AccessRegisterCommand(0);
        command.set_cmd_type(0);
        command.set_transfer(true);
        command.set_write(true);

        // registers are 32 bit, so we have size 2 here
        command.set_aarsize(RiscvBusAccess::A32);
        command.set_postexec(true);

        // register s0, ie. 0x1008
        command.set_regno((registers::S0).id.0 as u32);

        self.schedule_write_dm_register(command)?;

        // The program-buffer commands must run in bounded batches: one
        // giant deferred queue overruns the debug module when a load is
        // slow (the program buffer loads from flash, for example, take
        // many more cycles than a RAM load), and later commands pile
        // onto a busy module until the DMI reports invalid operations.
        // Executing each batch and waiting for it to settle first keeps
        // the queue depth compatible with the slowest load.
        let total_words = data.len();
        let mut word_index = 0;
        for batch in data.chunks_mut(WORDS_PER_BATCH) {
            let mut result_idxs = Vec::with_capacity(batch.len());
            for _ in 0..batch.len() {
                let mut command = AccessRegisterCommand(0);
                command.set_cmd_type(0);
                command.set_transfer(true);
                command.set_write(false);

                // registers are 32 bit, so we have size 2 here
                command.set_aarsize(RiscvBusAccess::A32);
                // The read of the LAST word of the range must not execute
                // the program buffer afterwards: the trailing load would
                // read one word past the requested range, which faults and
                // latches cmderr when the range ends at a memory-region
                // boundary.
                command.set_postexec(word_index + 1 < total_words);

                command.set_regno((registers::S1).id.0 as u32);

                self.schedule_write_dm_register(command)?;

                // Read back s1
                result_idxs.push(self.schedule_read_dm_register::<Data0>()?);
                word_index += 1;
            }

            self.execute()?;

            for (slot, idx) in batch.iter_mut().zip(result_idxs) {
                let value = Data0::from(self.dtm.read_deferred_result(idx)?.into_u32());
                *slot = V::from_register_value(value.0);
            }

            // Wait for the last command of the batch to finish before
            // queueing the next one.
            let start_time = Instant::now();
            loop {
                let status: Abstractcs = self.read_dm_register()?;
                if !status.busy() {
                    if status.cmderr() != 0 {
                        return Err(RiscvError::AbstractCommand(
                            AbstractCommandErrorKind::parse(status.cmderr() as u8),
                        ));
                    }
                    break;
                }
                if start_time.elapsed() > timeout {
                    return Err(RiscvError::Timeout);
                }
            }
        }

        self.restore_s0(s0)?;
        self.restore_s1(s1)?;

        Ok(())
    }

    /// Memory write using system bus
    fn perform_memory_write_sysbus<V: RiscvValue>(
        &mut self,
        address: u32,
        data: &[V],
    ) -> Result<(), RiscvError> {
        let mut sbcs = Sbcs(0);

        // Set correct access width
        sbcs.set_sbaccess(V::WIDTH as u32);
        sbcs.set_sbautoincrement(true);

        self.schedule_write_dm_register(sbcs)?;

        self.schedule_write_dm_register(Sbaddress0(address))?;

        for value in data {
            self.schedule_write_large_dtm_register::<V, Sbdata>(*value)?;
        }

        // Check that the write was succesful
        let sbcs = self.read_dm_register::<Sbcs>()?;

        if sbcs.sberror() != 0 {
            Err(RiscvError::SystemBusAccess)
        } else {
            Ok(())
        }
    }

    /// Perform memory write to a single location using the program buffer.
    /// Only writes up to a width of 32 bits are currently supported.
    fn perform_memory_write_progbuf<V: RiscvValue32>(
        &mut self,
        address: u32,
        data: V,
    ) -> Result<(), RiscvError> {
        tracing::debug!(
            "Memory write using progbuf - {:#010x} = {:#?}",
            address,
            data
        );

        // Backup registers s0 and s1
        let s0 = self.save_s0()?;
        let s1 = self.save_s1()?;

        let sw_command = assembly::sw(0, 8, V::WIDTH as u32, 9);

        self.schedule_setup_program_buffer(&[sw_command])?;

        // write address into s0
        self.abstract_cmd_register_write(&registers::S0, address)?;

        // write data into data 0
        self.schedule_write_dm_register(Data0(data.into()))?;

        // Write s1, then execute program buffer
        let mut command = AccessRegisterCommand(0);
        command.set_cmd_type(0);
        command.set_transfer(true);
        command.set_write(true);

        // registers are 32 bit, so we have size 2 here
        command.set_aarsize(RiscvBusAccess::A32);
        command.set_postexec(true);

        // register s1, ie. 0x1009
        command.set_regno((registers::S1).id.0 as u32);

        self.schedule_write_dm_register(command)?;

        let status = self.read_dm_register::<Abstractcs>()?;

        if status.cmderr() != 0 {
            let error = AbstractCommandErrorKind::parse(status.cmderr() as u8);

            tracing::error!(
                "Executing the abstract command for write_{} failed: {:?} ({:x?})",
                V::WIDTH.byte_width() * 8,
                error,
                status,
            );

            return Err(RiscvError::AbstractCommand(error));
        }

        self.restore_s0(s0)?;
        self.restore_s1(s1)?;

        Ok(())
    }

    /// Perform multiple memory writes to consecutive locations using the program buffer.
    /// Only writes up to a width of 32 bits are currently supported.
    fn perform_memory_write_multiple_progbuf<V: RiscvValue32>(
        &mut self,
        address: u32,
        data: &[V],
    ) -> Result<(), RiscvError> {
        let s0 = self.save_s0()?;
        let s1 = self.save_s1()?;

        // Setup program buffer for multiple writes
        // Store value from register s9 into memory,
        // then increase the address for next write.
        self.schedule_setup_program_buffer(&[
            assembly::sw(0, 8, V::WIDTH as u32, 9),
            assembly::addi(8, 8, V::WIDTH.byte_width() as i16),
        ])?;

        // write address into s0
        self.abstract_cmd_register_write(&registers::S0, address)?;

        for value in data {
            // write address into data 0
            self.schedule_write_dm_register(Data0((*value).into()))?;

            // Write s0, then execute program buffer
            let mut command = AccessRegisterCommand(0);
            command.set_cmd_type(0);
            command.set_transfer(true);
            command.set_write(true);

            // registers are 32 bit, so we have size 2 here
            command.set_aarsize(RiscvBusAccess::A32);
            command.set_postexec(true);

            // register s1
            command.set_regno((registers::S1).id.0 as u32);

            self.schedule_write_dm_register(command)?;
        }

        // Errors are sticky, so we can just check at the end if everything worked.
        let status = self.read_dm_register::<Abstractcs>()?;

        if status.cmderr() != 0 {
            let error = AbstractCommandErrorKind::parse(status.cmderr() as u8);

            tracing::error!(
                "Executing the abstract command for write_multiple_{} failed: {:?} ({:x?})",
                V::WIDTH.byte_width() * 8,
                error,
                status,
            );

            return Err(RiscvError::AbstractCommand(error));
        }

        // Restore register s0 and s1

        self.restore_s0(s0)?;
        self.restore_s1(s1)?;

        Ok(())
    }

    pub(crate) fn execute_abstract_command(&mut self, command: u32) -> Result<(), RiscvError> {
        // ensure that preconditions are fullfileld
        // haltreq      = 0
        // resumereq    = 0
        // ackhavereset = 0

        let mut dmcontrol: Dmcontrol = self.read_dm_register()?;
        dmcontrol.set_dmactive(true);
        dmcontrol.set_haltreq(false);
        dmcontrol.set_resumereq(false);
        dmcontrol.set_ackhavereset(false);
        self.schedule_write_dm_register(dmcontrol)?;

        // Clear any previous command errors.
        let mut abstractcs_clear = Abstractcs(0);
        abstractcs_clear.set_cmderr(0x7);

        self.schedule_write_dm_register(abstractcs_clear)?;
        self.schedule_write_dm_register(Command(command))?;

        let start_time = Instant::now();

        // Poll busy flag in abstractcs.
        let mut abstractcs;
        loop {
            abstractcs = self.read_dm_register::<Abstractcs>()?;

            if !abstractcs.busy() {
                break;
            }

            if start_time.elapsed() > RISCV_TIMEOUT {
                return Err(RiscvError::Timeout);
            }
        }

        tracing::debug!("abstracts: {:?}", abstractcs);

        // Check command result for error.
        if abstractcs.cmderr() != 0 {
            return Err(RiscvError::AbstractCommand(
                AbstractCommandErrorKind::parse(abstractcs.cmderr() as u8),
            ));
        }

        Ok(())
    }

    /// Check if a register can be accessed via abstract commands
    fn check_abstract_cmd_register_support(
        &self,
        regno: RegisterId,
        rw: CoreRegisterAbstractCmdSupport,
    ) -> bool {
        if let Some(status) = self.state.abstract_cmd_register_info.get(&regno) {
            status.supports(rw)
        } else {
            // If not cached yet, assume the register is accessible
            true
        }
    }

    /// Remember, that the given register can not be accessed via abstract commands
    fn set_abstract_cmd_register_unsupported(
        &mut self,
        regno: RegisterId,
        rw: CoreRegisterAbstractCmdSupport,
    ) {
        let entry = self
            .state
            .abstract_cmd_register_info
            .entry(regno)
            .or_insert(CoreRegisterAbstractCmdSupport::BOTH);

        entry.unset(rw);
    }

    // Read a core register using an abstract command
    pub(crate) fn abstract_cmd_register_read(
        &mut self,
        regno: impl Into<RegisterId>,
    ) -> Result<u32, RiscvError> {
        let regno = regno.into();

        // Check if the register was already tried via abstract cmd
        if !self.check_abstract_cmd_register_support(regno, CoreRegisterAbstractCmdSupport::READ) {
            return Err(RiscvError::AbstractCommand(
                AbstractCommandErrorKind::NotSupported,
            ));
        }

        // read from data0
        let mut command = AccessRegisterCommand(0);
        command.set_cmd_type(0);
        command.set_transfer(true);
        command.set_aarsize(RiscvBusAccess::A32);

        command.set_regno(regno.0 as u32);

        match self.execute_abstract_command(command.0) {
            Ok(_) => (),
            err @ Err(RiscvError::AbstractCommand(AbstractCommandErrorKind::NotSupported)) => {
                // Remember, that this register is unsupported
                self.set_abstract_cmd_register_unsupported(
                    regno,
                    CoreRegisterAbstractCmdSupport::READ,
                );
                err?;
            }
            Err(e) => return Err(e),
        }

        let register_value: Data0 = self.read_dm_register()?;

        Ok(register_value.into())
    }

    pub(crate) fn abstract_cmd_register_write<V: RiscvValue>(
        &mut self,
        regno: impl Into<RegisterId>,
        value: V,
    ) -> Result<(), RiscvError> {
        let regno = regno.into();

        // Check if the register was already tried via abstract cmd
        if !self.check_abstract_cmd_register_support(regno, CoreRegisterAbstractCmdSupport::WRITE) {
            return Err(RiscvError::AbstractCommand(
                AbstractCommandErrorKind::NotSupported,
            ));
        }

        // write to data0
        let mut command = AccessRegisterCommand(0);
        command.set_cmd_type(0);
        command.set_transfer(true);
        command.set_write(true);
        command.set_aarsize(V::WIDTH);

        command.set_regno(regno.0 as u32);

        self.schedule_write_large_dtm_register::<V, Arg0>(value)?;

        match self.execute_abstract_command(command.0) {
            Ok(_) => Ok(()),
            err @ Err(RiscvError::AbstractCommand(AbstractCommandErrorKind::NotSupported)) => {
                // Remember, that this register is unsupported
                self.set_abstract_cmd_register_unsupported(
                    regno,
                    CoreRegisterAbstractCmdSupport::WRITE,
                );
                err
            }
            Err(e) => Err(e),
        }
    }

    /// Read the CSR `progbuf` register.
    pub fn read_csr_progbuf(&mut self, address: u16) -> Result<u32, RiscvError> {
        tracing::debug!("Reading CSR {:#04x}", address);

        // Validate that the CSR address is valid
        if address > RISCV_MAX_CSR_ADDR {
            return Err(RiscvError::UnsupportedCsrAddress(address));
        }

        let s0 = self.save_s0()?;

        // Read csr value into register 8 (s0)
        let csrr_cmd = assembly::csrr(8, address);

        self.schedule_setup_program_buffer(&[csrr_cmd])?;

        // command: postexec
        let mut postexec_cmd = AccessRegisterCommand(0);
        postexec_cmd.set_postexec(true);

        self.execute_abstract_command(postexec_cmd.0)?;

        // read the s0 value
        let reg_value = self.abstract_cmd_register_read(&registers::S0)?;

        // restore original value in s0
        self.restore_s0(s0)?;

        Ok(reg_value)
    }

    /// Write the CSR `progbuf` register.
    pub fn write_csr_progbuf(&mut self, address: u16, value: u32) -> Result<(), RiscvError> {
        tracing::debug!("Writing CSR {:#04x}={}", address, value);

        // Validate that the CSR address is valid
        if address > RISCV_MAX_CSR_ADDR {
            return Err(RiscvError::UnsupportedCsrAddress(address));
        }

        // Backup register s0
        let s0 = self.save_s0()?;

        // Write value into s0
        self.abstract_cmd_register_write(&registers::S0, value)?;

        // Built the CSRW command to write into the program buffer
        let csrw_cmd = assembly::csrw(address, 8);
        self.schedule_setup_program_buffer(&[csrw_cmd])?;

        // command: postexec
        let mut postexec_cmd = AccessRegisterCommand(0);
        postexec_cmd.set_postexec(true);

        self.execute_abstract_command(postexec_cmd.0)?;

        // command: transfer, regno = 0x1008
        // restore original value in s0
        self.restore_s0(s0)?;

        Ok(())
    }

    fn read_word<V: RiscvValue32>(&mut self, address: u32) -> Result<V, crate::Error> {
        let result = match self.state.memory_access_method(V::WIDTH) {
            MemoryAccessMethod::ProgramBuffer => self.perform_memory_read_progbuf(address)?,
            MemoryAccessMethod::SystemBus => self.perform_memory_read_sysbus(address)?,
            MemoryAccessMethod::AbstractCommand => {
                let mut word = [V::from_register_value(0); 1];
                self.perform_memory_read_multiple_abstract(address, &mut word)?;
                word[0]
            }
        };

        Ok(result)
    }

    fn read_multiple<V: RiscvValue32>(
        &mut self,
        address: u32,
        data: &mut [V],
    ) -> Result<(), crate::Error> {
        tracing::debug!("read_32 from {:#08x}", address);

        match self.state.memory_access_method(RiscvBusAccess::A32) {
            MemoryAccessMethod::ProgramBuffer => {
                self.perform_memory_read_multiple_progbuf(address, data, RISCV_TIMEOUT)?;
            }
            MemoryAccessMethod::SystemBus => {
                self.perform_memory_read_multiple_sysbus(address, data)?;
            }
            MemoryAccessMethod::AbstractCommand => {
                self.perform_memory_read_multiple_abstract(address, data)?;
            }
        };

        Ok(())
    }

    fn write_word<V: RiscvValue32>(&mut self, address: u32, data: V) -> Result<(), crate::Error> {
        match self.state.memory_access_method(V::WIDTH) {
            MemoryAccessMethod::ProgramBuffer => {
                self.perform_memory_write_progbuf(address, data)?
            }
            MemoryAccessMethod::SystemBus => self.perform_memory_write_sysbus(address, &[data])?,
            MemoryAccessMethod::AbstractCommand => {
                self.perform_memory_write_multiple_abstract(address, &[data])?;
            }
        };

        Ok(())
    }

    fn write_multiple<V: RiscvValue32>(
        &mut self,
        address: u32,
        data: &[V],
    ) -> Result<(), crate::Error> {
        match self.state.memory_access_method(V::WIDTH) {
            MemoryAccessMethod::SystemBus => self.perform_memory_write_sysbus(address, data)?,
            MemoryAccessMethod::ProgramBuffer => {
                self.perform_memory_write_multiple_progbuf(address, data)?
            }
            MemoryAccessMethod::AbstractCommand => {
                self.perform_memory_write_multiple_abstract(address, data)?;
            }
        }

        Ok(())
    }

    pub(crate) fn execute(&mut self) -> Result<(), RiscvError> {
        self.dtm.execute()
    }

    pub(crate) fn schedule_write_dm_register<R: MemoryMappedRegister<u32>>(
        &mut self,
        register: R,
    ) -> Result<(), RiscvError> {
        // write write command to dmi register

        tracing::debug!(
            "Write DM register '{}' at {:#010x} = {:x?}",
            R::NAME,
            R::get_mmio_address(),
            register
        );

        self.schedule_write_dm_register_untyped(R::get_mmio_address(), register.into())?;
        Ok(())
    }

    /// Write to a DM register
    ///
    /// Use the [`schedule_write_dm_register`] function if possible.
    fn schedule_write_dm_register_untyped(
        &mut self,
        address: u64,
        value: u32,
    ) -> Result<Option<DeferredResultIndex>, RiscvError> {
        self.dtm.schedule_write(address, value)
    }

    pub(super) fn schedule_read_dm_register<R: MemoryMappedRegister<u32>>(
        &mut self,
    ) -> Result<DeferredResultIndex, RiscvError> {
        tracing::debug!(
            "Reading DM register '{}' at {:#010x}",
            R::NAME,
            R::get_mmio_address()
        );

        self.schedule_read_dm_register_untyped(R::get_mmio_address())
    }

    /// Read from a DM register
    ///
    /// Use the [`schedule_read_dm_register`] function if possible.
    fn schedule_read_dm_register_untyped(
        &mut self,
        address: u64,
    ) -> Result<DeferredResultIndex, RiscvError> {
        // Prepare the read by sending a read request with the register address
        self.dtm.schedule_read(address)
    }

    fn schedule_read_large_dtm_register<V, R>(&mut self) -> Result<DeferredResultIndex, RiscvError>
    where
        V: RiscvValue,
        R: LargeRegister,
    {
        V::schedule_read_from_register::<R>(self)
    }

    fn read_large_dtm_register<V, R>(&mut self) -> Result<V, RiscvError>
    where
        V: RiscvValue,
        R: LargeRegister,
    {
        V::read_from_register::<R>(self)
    }

    fn schedule_write_large_dtm_register<V, R>(
        &mut self,
        value: V,
    ) -> Result<Option<DeferredResultIndex>, RiscvError>
    where
        V: RiscvValue,
        R: LargeRegister,
    {
        V::schedule_write_to_register::<R>(self, value)
    }

    /// Check if the connected device supports halt after reset.
    ///
    /// Returns a cached value if available, otherwise queries the
    /// `hasresethaltreq` bit in the `dmstatus` register.
    pub(crate) fn supports_reset_halt_req(&mut self) -> Result<bool, RiscvError> {
        if let Some(has_reset_halt_req) = self.state.hasresethaltreq {
            Ok(has_reset_halt_req)
        } else {
            let dmstatus: Dmstatus = self.read_dm_register()?;

            self.state.hasresethaltreq = Some(dmstatus.hasresethaltreq());

            Ok(dmstatus.hasresethaltreq())
        }
    }

    // Resume the core.
    pub(crate) fn resume_core(&mut self) -> Result<(), RiscvError> {
        // set resume request.
        let mut dmcontrol: Dmcontrol = self.read_dm_register()?;
        dmcontrol.set_dmactive(true);
        dmcontrol.set_resumereq(true);
        self.write_dm_register(dmcontrol)?;

        // check if request has been acknowleged.
        let status: Dmstatus = self.read_dm_register()?;
        if !status.allresumeack() {
            println!("resume_core");
            return Err(RiscvError::RequestNotAcknowledged);
        }

        // clear resume request.
        dmcontrol.set_resumereq(false);
        self.write_dm_register(dmcontrol)?;

        Ok(())
    }

    pub(crate) fn reset_hart_and_halt(&mut self, timeout: Duration) -> Result<(), RiscvError> {
        tracing::debug!("Resetting core, setting hartreset bit");

        let mut dmcontrol: Dmcontrol = self.read_dm_register()?;
        dmcontrol.set_dmactive(true);
        dmcontrol.set_hartreset(true);
        dmcontrol.set_haltreq(true);

        self.write_dm_register(dmcontrol)?;

        // Read back register to verify reset is supported
        let readback: Dmcontrol = self.read_dm_register()?;

        if readback.hartreset() {
            tracing::debug!("Clearing hartreset bit");
            // Reset is performed by setting the bit high, and then low again
            let mut dmcontrol = readback;
            dmcontrol.set_dmactive(true);
            dmcontrol.set_hartreset(false);

            self.write_dm_register(dmcontrol)?;
        } else {
            // Hartreset is not supported, whole core needs to be reset
            //
            // TODO: Cache this
            tracing::debug!("Hartreset bit not supported, using ndmreset");
            dmcontrol.set_hartreset(false);
            dmcontrol.set_ndmreset(true);
            dmcontrol.set_haltreq(true);

            self.write_dm_register(dmcontrol)?;

            tracing::debug!("Clearing ndmreset bit");
            dmcontrol.set_ndmreset(false);
            dmcontrol.set_haltreq(true);

            self.write_dm_register(dmcontrol)?;
        }

        let start = Instant::now();

        loop {
            // check that cores have reset
            let readback: Dmstatus = self.read_dm_register()?;

            if readback.allhavereset() && readback.allhalted() {
                break;
            }
            
            if start.elapsed() > timeout {
                println!("reset_hart_and_halt");
                return Err(RiscvError::RequestNotAcknowledged);
            }
        }

        // clear the reset request
        dmcontrol.set_haltreq(false);
        dmcontrol.set_ackhavereset(true);
        dmcontrol.set_hartreset(false);
        dmcontrol.set_ndmreset(false);

        self.write_dm_register(dmcontrol)?;

        // Reenable halt on breakpoint because this gets disabled if we reset the core
        self.debug_on_sw_breakpoint(true)?; // TODO: only restore if enabled before?

        Ok(())
    }

    pub(crate) fn debug_on_sw_breakpoint(&mut self, enabled: bool) -> Result<(), RiscvError> {
        let mut dcsr = Dcsr(self.read_csr(0x7b0)?);

        dcsr.set_ebreakm(enabled);
        dcsr.set_ebreaks(enabled);
        dcsr.set_ebreaku(enabled);

        match self.abstract_cmd_register_write(0x7b0, dcsr.0) {
            Err(RiscvError::AbstractCommand(AbstractCommandErrorKind::NotSupported)) => {
                tracing::debug!("Could not write core register {:#x} with abstract command, falling back to program buffer", 0x7b0);
                self.write_csr_progbuf(0x7b0, dcsr.0)
            }
            other => other,
        }
    }
}

pub(crate) trait LargeRegister {
    const R0_ADDRESS: u8;
    const R1_ADDRESS: u8;
    const R2_ADDRESS: u8;
    const R3_ADDRESS: u8;
}

struct Sbdata {}

impl LargeRegister for Sbdata {
    const R0_ADDRESS: u8 = Sbdata0::ADDRESS_OFFSET as u8;
    const R1_ADDRESS: u8 = Sbdata1::ADDRESS_OFFSET as u8;
    const R2_ADDRESS: u8 = Sbdata2::ADDRESS_OFFSET as u8;
    const R3_ADDRESS: u8 = Sbdata3::ADDRESS_OFFSET as u8;
}

struct Arg0 {}

impl LargeRegister for Arg0 {
    const R0_ADDRESS: u8 = Data0::ADDRESS_OFFSET as u8;
    const R1_ADDRESS: u8 = Data1::ADDRESS_OFFSET as u8;
    const R2_ADDRESS: u8 = Data2::ADDRESS_OFFSET as u8;
    const R3_ADDRESS: u8 = Data3::ADDRESS_OFFSET as u8;
}

/// Helper trait, limited to RiscvValue no larger than 32 bits
pub(crate) trait RiscvValue32: RiscvValue + Into<u32> {
    fn from_register_value(value: u32) -> Self;
}

impl RiscvValue32 for u8 {
    fn from_register_value(value: u32) -> Self {
        value as u8
    }
}
impl RiscvValue32 for u16 {
    fn from_register_value(value: u32) -> Self {
        value as u16
    }
}
impl RiscvValue32 for u32 {
    fn from_register_value(value: u32) -> Self {
        value
    }
}

/// Marker trait for different values which
/// can be read / written using the debug module.
pub(crate) trait RiscvValue: std::fmt::Debug + Copy + Sized {
    const WIDTH: RiscvBusAccess;

    fn schedule_read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<DeferredResultIndex, RiscvError>
    where
        R: LargeRegister;

    fn schedule_write_to_register<R>(
        interface: &mut RiscvCommunicationInterface,
        value: Self,
    ) -> Result<Option<DeferredResultIndex>, RiscvError>
    where
        R: LargeRegister;

    fn read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<Self, RiscvError>
    where
        R: LargeRegister;
}

impl RiscvValue for u8 {
    const WIDTH: RiscvBusAccess = RiscvBusAccess::A8;

    fn schedule_read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<DeferredResultIndex, RiscvError>
    where
        R: LargeRegister,
    {
        interface.schedule_read_dm_register_untyped(R::R0_ADDRESS as u64)
    }

    fn schedule_write_to_register<R>(
        interface: &mut RiscvCommunicationInterface,
        value: Self,
    ) -> Result<Option<DeferredResultIndex>, RiscvError>
    where
        R: LargeRegister,
    {
        interface.schedule_write_dm_register_untyped(R::R0_ADDRESS as u64, value as u32)
    }

    fn read_from_register<R>(
            interface: &mut RiscvCommunicationInterface,
        ) -> Result<Self, RiscvError>
        where
            R: LargeRegister
    {
        interface.read_dm_register_untyped(R::R0_ADDRESS as u64).map(|x| x as u8)
    }
}

impl RiscvValue for u16 {
    const WIDTH: RiscvBusAccess = RiscvBusAccess::A16;

    fn schedule_read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<DeferredResultIndex, RiscvError>
    where
        R: LargeRegister,
    {
        interface.schedule_read_dm_register_untyped(R::R0_ADDRESS as u64)
    }

    fn schedule_write_to_register<R>(
        interface: &mut RiscvCommunicationInterface,
        value: Self,
    ) -> Result<Option<DeferredResultIndex>, RiscvError>
    where
        R: LargeRegister,
    {
        interface.schedule_write_dm_register_untyped(R::R0_ADDRESS as u64, value as u32)
    }

    fn read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<Self, RiscvError>
    where
        R: LargeRegister
    {
        interface.read_dm_register_untyped(R::R0_ADDRESS as u64).map(|x| x as u16)
    }
}

impl RiscvValue for u32 {
    const WIDTH: RiscvBusAccess = RiscvBusAccess::A32;

    fn schedule_read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<DeferredResultIndex, RiscvError>
    where
        R: LargeRegister,
    {
        interface.schedule_read_dm_register_untyped(R::R0_ADDRESS as u64)
    }
    fn schedule_write_to_register<R>(
        interface: &mut RiscvCommunicationInterface,
        value: Self,
    ) -> Result<Option<DeferredResultIndex>, RiscvError>
    where
        R: LargeRegister,
    {
        interface.schedule_write_dm_register_untyped(R::R0_ADDRESS as u64, value)
    }

    fn read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<Self, RiscvError>
    where
        R: LargeRegister
    {
        interface.read_dm_register_untyped(R::R0_ADDRESS as u64).map(|x| x as u32)
    }
}

impl RiscvValue for u64 {
    const WIDTH: RiscvBusAccess = RiscvBusAccess::A64;

    fn schedule_read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<DeferredResultIndex, RiscvError>
    where
        R: LargeRegister,
    {
        interface.schedule_read_dm_register_untyped(R::R1_ADDRESS as u64)?;
        interface.schedule_read_dm_register_untyped(R::R0_ADDRESS as u64)
    }

    fn schedule_write_to_register<R>(
        interface: &mut RiscvCommunicationInterface,
        value: Self,
    ) -> Result<Option<DeferredResultIndex>, RiscvError>
    where
        R: LargeRegister,
    {
        let upper_bits = (value >> 32) as u32;
        let lower_bits = (value & 0xffff_ffff) as u32;

        // R0 has to be written last, side effects are triggerd by writes from
        // this register.

        interface.schedule_write_dm_register_untyped(R::R1_ADDRESS as u64, upper_bits)?;
        interface.schedule_write_dm_register_untyped(R::R0_ADDRESS as u64, lower_bits)
    }
    fn read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<Self, RiscvError>
    where
        R: LargeRegister
    {
        let r1 = interface.read_dm_register_untyped(R::R1_ADDRESS as u64)? as u64;
        let r0 = interface.read_dm_register_untyped(R::R0_ADDRESS as u64)? as u64;
        Ok((r1<<32)+r0)
    }
}

impl RiscvValue for u128 {
    const WIDTH: RiscvBusAccess = RiscvBusAccess::A128;

    fn schedule_read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<DeferredResultIndex, RiscvError>
    where
        R: LargeRegister,
    {
        interface.schedule_read_dm_register_untyped(R::R3_ADDRESS as u64)?;
        interface.schedule_read_dm_register_untyped(R::R2_ADDRESS as u64)?;
        interface.schedule_read_dm_register_untyped(R::R1_ADDRESS as u64)?;
        interface.schedule_read_dm_register_untyped(R::R0_ADDRESS as u64)
    }

    fn schedule_write_to_register<R>(
        interface: &mut RiscvCommunicationInterface,
        value: Self,
    ) -> Result<Option<DeferredResultIndex>, RiscvError>
    where
        R: LargeRegister,
    {
        let bits_3 = (value >> 96) as u32;
        let bits_2 = (value >> 64) as u32;
        let bits_1 = (value >> 32) as u32;
        let bits_0 = (value & 0xffff_ffff) as u32;

        // R0 has to be written last, side effects are triggerd by writes from
        // this register.

        interface.schedule_write_dm_register_untyped(R::R3_ADDRESS as u64, bits_3)?;
        interface.schedule_write_dm_register_untyped(R::R2_ADDRESS as u64, bits_2)?;
        interface.schedule_write_dm_register_untyped(R::R1_ADDRESS as u64, bits_1)?;
        interface.schedule_write_dm_register_untyped(R::R0_ADDRESS as u64, bits_0)
    }

    fn read_from_register<R>(
        interface: &mut RiscvCommunicationInterface,
    ) -> Result<Self, RiscvError>
    where
        R: LargeRegister
    {
        let r3 = interface.read_dm_register_untyped(R::R3_ADDRESS as u64)? as u128;
        let r2 = interface.read_dm_register_untyped(R::R2_ADDRESS as u64)? as u128;
        let r1 = interface.read_dm_register_untyped(R::R1_ADDRESS as u64)? as u128;
        let r0 = interface.read_dm_register_untyped(R::R0_ADDRESS as u64)? as u128;
        Ok((r3<<96)+(r2<<64)+(r1<<32)+r0)
    }
}

impl MemoryInterface for RiscvCommunicationInterface<'_> {
    fn supports_native_64bit_access(&mut self) -> bool {
        false
    }

    fn read_word_64(&mut self, address: u64) -> Result<u64, crate::error::Error> {
        let address = valid_32bit_address(address)?;
        let mut ret = self.read_word::<u32>(address)? as u64;
        ret |= (self.read_word::<u32>(address + 4)? as u64) << 32;

        Ok(ret)
    }

    fn read_word_32(&mut self, address: u64) -> Result<u32, crate::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("read_word_32 from {:#08x}", address);
        self.read_word(address)
    }

    fn read_word_16(&mut self, address: u64) -> Result<u16, crate::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("read_word_16 from {:#08x}", address);
        self.read_word(address)
    }

    fn read_word_8(&mut self, address: u64) -> Result<u8, crate::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("read_word_8 from {:#08x}", address);
        self.read_word(address)
    }

    fn read_64(&mut self, address: u64, data: &mut [u64]) -> Result<(), crate::error::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("read_64 from {:#08x}", address);

        for (i, d) in data.iter_mut().enumerate() {
            *d = self.read_word_64((address + (i as u32 * 8)).into())?;
        }

        Ok(())
    }

    fn read_32(&mut self, address: u64, data: &mut [u32]) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("read_32 from {:#08x}", address);
        self.read_multiple(address, data)
    }

    fn read_16(&mut self, address: u64, data: &mut [u16]) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("read_16 from {:#08x}", address);
        self.read_multiple(address, data)
    }

    fn read_8(&mut self, address: u64, data: &mut [u8]) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("read_8 from {:#08x}", address);

        self.read_multiple(address, data)
    }

    fn read(&mut self, address: u64, data: &mut [u8]) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        self.read_multiple(address, data)
    }

    fn write_word_64(&mut self, address: u64, data: u64) -> Result<(), crate::error::Error> {
        let address = valid_32bit_address(address)?;
        let low_word = data as u32;
        let high_word = (data >> 32) as u32;

        self.write_word(address, low_word)?;
        self.write_word(address + 4, high_word)
    }

    fn write_word_32(&mut self, address: u64, data: u32) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        self.write_word(address, data)
    }

    fn write_word_16(&mut self, address: u64, data: u16) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        self.write_word(address, data)
    }

    fn write_word_8(&mut self, address: u64, data: u8) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        self.write_word(address, data)
    }

    fn write_64(&mut self, address: u64, data: &[u64]) -> Result<(), crate::error::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("write_64 to {:#08x}", address);

        for (i, d) in data.iter().enumerate() {
            self.write_word_64((address + (i as u32 * 8)).into(), *d)?;
        }

        Ok(())
    }

    fn write_32(&mut self, address: u64, data: &[u32]) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("write_32 to {:#08x}", address);

        self.write_multiple(address, data)
    }

    fn write_16(&mut self, address: u64, data: &[u16]) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("write_16 to {:#08x}", address);

        self.write_multiple(address, data)
    }

    fn write_8(&mut self, address: u64, data: &[u8]) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        tracing::debug!("write_8 to {:#08x}", address);

        self.write_multiple(address, data)
    }

    fn write(&mut self, address: u64, data: &[u8]) -> Result<(), crate::Error> {
        let address = valid_32bit_address(address)?;
        self.write_multiple(address, data)
    }

    fn supports_8bit_transfers(&self) -> Result<bool, crate::Error> {
        Ok(true)
    }

    fn flush(&mut self) -> Result<(), crate::Error> {
        Ok(())
    }
}

/// Access width for bus access.
/// This is used both for system bus access (`sbcs` register),
/// as well for abstract commands.
#[derive(Copy, Clone, PartialEq, PartialOrd, Hash, Eq, Debug)]
pub enum RiscvBusAccess {
    /// 1 byte
    A8 = 0,
    /// 2 bytes
    A16 = 1,
    /// 4 bytes
    A32 = 2,
    /// 8 bytes
    A64 = 3,
    /// 16 bytes.
    A128 = 4,
}

impl RiscvBusAccess {
    /// Width of an access in bytes
    const fn byte_width(&self) -> usize {
        match self {
            RiscvBusAccess::A8 => 1,
            RiscvBusAccess::A16 => 2,
            RiscvBusAccess::A32 => 4,
            RiscvBusAccess::A64 => 8,
            RiscvBusAccess::A128 => 16,
        }
    }
}

impl From<RiscvBusAccess> for u8 {
    fn from(value: RiscvBusAccess) -> Self {
        value as u8
    }
}

/// Different methods of memory access,
/// which can be supported by a debug module.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum MemoryAccessMethod {
    /// Memory access using the program buffer is supported
    ProgramBuffer,
    /// Memory access using an abstract command is supported
    AbstractCommand,
    /// Memory access using system bus access supported
    SystemBus,
}

memory_mapped_bitfield_register! {
    /// Abstract command register, located at address 0x17
    /// This is not for all commands, only for the ones
    /// from the debug spec.
    pub struct AccessRegisterCommand(u32);
    0x17, "command",
    impl From;
    /// This is 0 to indicate Access Register Command.
    pub _, set_cmd_type: 31, 24;
    /// 2: Access the lowest 32 bits of the register.\
    /// 3: Access the lowest 64 bits of the register.\
    /// 4: Access the lowest 128 bits of the register.
    ///
    /// If `aarsize` specifies a size larger than the register’s
    /// actual size, then the access must fail. If a register is accessible, then reads of `aarsize` less than
    /// or equal to the register’s actual size must be supported.
    ///
    /// This field controls the Argument Width as referenced in Table 3.1.
    pub u8, from into RiscvBusAccess, _, set_aarsize: 22, 20;
    /// 0: No effect. This variant must be supported.\
    /// 1: After a successful register access, `regno` is incremented (wrapping around to 0). Supporting
    /// this variant is optional.
    pub _, set_aarpostincrement: 19;
    /// 0: No effect. This variant must be supported, and
    /// is the only supported one if `progbufsize` is 0.\
    /// 1: Execute the program in the Program Buffer
    /// exactly once after performing the transfer, if any.
    /// Supporting this variant is optional.
    pub _, set_postexec: 18;
    /// 0: Don’t do the operation specified by write.\
    /// 1: Do the operation specified by write.
    /// This bit can be used to just execute the Program Buffer without having to worry about placing valid values into `aarsize` or `regno`
    pub _, set_transfer: 17;
    /// When transfer is set: 0: Copy data from the specified register into arg0 portion of data.
    /// 1: Copy data from arg0 portion of data into the
    /// specified register.
    pub _, set_write: 16;
    /// Number of the register to access, as described in
    /// Table 3.3. dpc may be used as an alias for PC if
    /// this command is supported on a non-halted hart.
    pub _, set_regno: 15, 0;
}

memory_mapped_bitfield_register! {
    /// System Bus Access Control and Status (see 3.12.18)
    pub struct Sbcs(u32);
    0x38, "sbcs",
    impl From;
    /// 0: The System Bus interface conforms to mainline
    /// drafts of this spec older than 1 January, 2018.\
    /// 1: The System Bus interface conforms to this version of the spec.
    ///
    /// Other values are reserved for future versions
    sbversion, _: 31, 29;
    /// Set when the debugger attempts to read data
    /// while a read is in progress, or when the debugger initiates a new access while one is already in
    /// progress (while `sbbusy` is set). It remains set until
    /// it’s explicitly cleared by the debugger.
    /// While this field is set, no more system bus accesses
    /// can be initiated by the Debug Module.
    sbbusyerror, set_sbbusyerror: 22;
    /// When 1, indicates the system bus master is busy.
    /// (Whether the system bus itself is busy is related,
    /// but not the same thing.) This bit goes high immediately when a read or write is requested for
    /// any reason, and does not go low until the access
    /// is fully completed.
    ///
    /// Writes to `sbcs` while `sbbusy` is high result in undefined behavior. A debugger must not write to
    /// sbcs until it reads `sbbusy` as 0.
    sbbusy, _: 21;
    /// When 1, every write to `sbaddress0` automatically
    /// triggers a system bus read at the new address.
    sbreadonaddr, set_sbreadonaddr: 20;
    /// Select the access size to use for system bus accesses.
    ///
    /// 0: 8-bit\
    /// 1: 16-bit\
    /// 2: 32-bit\
    /// 3: 64-bit\
    /// 4: 128-bit
    ///
    /// If `sbaccess` has an unsupported value when the
    /// DM starts a bus access, the access is not performed and `sberror` is set to 4.
    sbaccess, set_sbaccess: 19, 17;
    /// When 1, `sbaddress` is incremented by the access
    /// size (in bytes) selected in `sbaccess` after every system bus access.
    sbautoincrement, set_sbautoincrement: 16;
    /// When 1, every read from `sbdata0` automatically
    /// triggers a system bus read at the (possibly autoincremented) address.
    sbreadondata, set_sbreadondata: 15;
    /// When the Debug Module’s system bus master encounters an error, this field gets set. The bits in
    /// this field remain set until they are cleared by writing 1 to them. While this field is non-zero, no
    /// more system bus accesses can be initiated by the
    /// Debug Module.
    /// An implementation may report “Other” (7) for any error condition.
    ///
    /// 0: There was no bus error.\
    /// 1: There was a timeout.\
    /// 2: A bad address was accessed.\
    /// 3: There was an alignment error.\
    /// 4: An access of unsupported size was requested.\
    /// 7: Other.
    sberror, set_sberror: 14, 12;
    /// Width of system bus addresses in bits. (0 indicates there is no bus access support.)
    sbasize, _: 11, 5;
    /// 1 when 128-bit system bus accesses are supported.
    sbaccess128, _: 4;
    /// 1 when 64-bit system bus accesses are supported.
    sbaccess64, _: 3;
    /// 1 when 32-bit system bus accesses are supported.
    sbaccess32, _: 2;
    /// 1 when 16-bit system bus accesses are supported.
    sbaccess16, _: 1;
    /// 1 when 8-bit system bus accesses are supported.
    sbaccess8, _: 0;
}

memory_mapped_bitfield_register! {
    /// Abstract Command Autoexec (see 3.12.8)
    #[derive(Eq, PartialEq)]
    pub struct Abstractauto(u32);
    0x18, "abstractauto",
    impl From;
    /// When a bit in this field is 1, read or write accesses to the corresponding `progbuf` word cause
    /// the command in command to be executed again.
    autoexecprogbuf, set_autoexecprogbuf: 31, 16;
    /// When a bit in this field is 1, read or write accesses to the corresponding data word cause the
    /// command in command to be executed again.
    autoexecdata, set_autoexecdata: 11, 0;
}

memory_mapped_bitfield_register! {
    /// Abstract command register, located at address 0x17
    /// This is not for all commands, only for the ones
    /// from the debug spec. (see 3.6.1.3)
    pub struct AccessMemoryCommand(u32);
    0x17, "command",
    /// This is 2 to indicate Access Memory Command.
    _, set_cmd_type: 31, 24;
    /// An implementation does not have to implement
    /// both virtual and physical accesses, but it must
    /// fail accesses that it doesn’t support.

    /// 0: Addresses are physical (to the hart they are
    /// performed on).\
    /// 1: Addresses are virtual, and translated the way
    /// they would be from M-mode, with `MPRV` set.
    pub _, set_aamvirtual: 23;
    /// 0: Access the lowest 8 bits of the memory location.\
    /// 1: Access the lowest 16 bits of the memory location.\
    /// 2: Access the lowest 32 bits of the memory location.\
    /// 3: Access the lowest 64 bits of the memory location.\
    /// 4: Access the lowest 128 bits of the memory location.
    pub _, set_aamsize: 22,20;
    /// After a memory access has completed, if this bit
    /// is 1, increment arg1 (which contains the address
    /// used) by the number of bytes encoded in `aamsize`.
    pub _, set_aampostincrement: 19;
    /// 0: Copy data from the memory location specified
    /// in arg1 into arg0 portion of data.\
    /// 1: Copy data from arg0 portion of data into the
    /// memory location specified in arg1.
    pub _, set_write: 16;
    /// These bits are reserved for target-specific uses.
    pub _, set_target_specific: 15, 14;
}

impl From<AccessMemoryCommand> for u32 {
    fn from(register: AccessMemoryCommand) -> Self {
        let mut reg = register;
        reg.set_cmd_type(2);
        reg.0
    }
}

impl From<u32> for AccessMemoryCommand {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

memory_mapped_bitfield_register! { pub struct Sbaddress0(u32); 0x39, "sbaddress0", impl From; }
memory_mapped_bitfield_register! { pub struct Sbaddress1(u32); 0x3a, "sbaddress1", impl From; }
memory_mapped_bitfield_register! { pub struct Sbaddress2(u32); 0x3b, "sbaddress2", impl From; }
memory_mapped_bitfield_register! { pub struct Sbaddress3(u32); 0x37, "sbaddress3", impl From; }

memory_mapped_bitfield_register! { pub struct Sbdata0(u32); 0x3c, "sbdata0", impl From; }
memory_mapped_bitfield_register! { pub struct Sbdata1(u32); 0x3d, "sbdata1", impl From; }
memory_mapped_bitfield_register! { pub struct Sbdata2(u32); 0x3e, "sbdata2", impl From; }
memory_mapped_bitfield_register! { pub struct Sbdata3(u32); 0x3f, "sbdata3", impl From; }

memory_mapped_bitfield_register! { pub struct Confstrptr0(u32); 0x19, "confstrptr0", impl From; }
memory_mapped_bitfield_register! { pub struct Confstrptr1(u32); 0x1a, "confstrptr1", impl From; }
memory_mapped_bitfield_register! { pub struct Confstrptr2(u32); 0x1b, "confstrptr2", impl From; }
memory_mapped_bitfield_register! { pub struct Confstrptr3(u32); 0x1c, "confstrptr3", impl From; }

/// Scriptable [DtmAccess] double for hardware-free testing of the
/// memory-access paths: immediate reads consult a register map (or a
/// scripted response queue), writes are recorded, and scheduled
/// operations commit on execute like a real transport.
#[cfg(test)]
pub(crate) mod test_mocks {
    use std::collections::{HashMap, VecDeque};
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use super::RiscvError;
    use crate::architecture::riscv::dtm::dtm_access::DtmAccess;
    use crate::probe::{CommandResult, DebugProbeError, DeferredResultIndex};
    use crate::MemoryMappedRegister;

    /// One transport-level observation, in order of occurrence.
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub(crate) enum MockEvent {
        Write(u64, u32),
        Read(u64),
        ScheduledRead(u64),
        Execute,
    }

    /// Observable activity of a [MockDtmAccess]; shareable so tests can
    /// inspect it after the interface has consumed the mock.
    #[derive(Debug, Default)]
    pub(crate) struct MockLog {
        /// Every queued write, immediate or scheduled, in order.
        pub writes: Vec<(u64, u32)>,
        /// Addresses of scheduled reads, in order.
        pub scheduled_reads: Vec<u64>,
        /// Number of execute() flushes.
        pub executes: usize,
        /// Full transport timeline, in order.
        pub events: Vec<MockEvent>,
    }

    #[derive(Debug, Default)]
    pub(crate) struct MockDtmAccess {
        pub log: Arc<Mutex<MockLog>>,
        /// Committed DM register state, visible to immediate reads.
        pub registers: HashMap<u64, u32>,
        /// Values served by immediate reads before the register map, in
        /// order; models data that advances with every read.
        pub read_responses: VecDeque<u32>,
        /// Values served by read_deferred_result, in order.
        pub deferred_results: VecDeque<u32>,
        /// Raw captures served by read_deferred_result before
        /// [`Self::deferred_results`], in order; models the byte
        /// captures of raw DMI scans.
        pub deferred_raw_results: VecDeque<Vec<u8>>,
        /// Value reported by abstractcs reads once data reads have been
        /// scheduled; models a busy or failed command state that appears
        /// with the first batch's completion.
        pub abstractcs_after_data_reads: Option<u32>,
        /// Writes that are queued but not yet executed.
        pending_writes: Vec<(u64, u32)>,
    }

    impl MockDtmAccess {
        pub fn with_read_responses(mut self, responses: impl IntoIterator<Item = u32>) -> Self {
            self.read_responses = responses.into_iter().collect();
            self
        }

        /// Script the abstractcs value reported once data reads have been
        /// scheduled (models a busy or failed command state appearing with
        /// the first batch's completion).
        pub fn with_abstractcs_after_data_reads(mut self, value: u32) -> Self {
            self.abstractcs_after_data_reads = Some(value);
            self
        }

        /// Provide the values served by read_deferred_result, in order.
        pub fn with_deferred_results(mut self, results: impl IntoIterator<Item = u32>) -> Self {
            self.deferred_results = results.into_iter().collect();
            self
        }

        /// Provide the raw captures served by read_deferred_result, in
        /// order; raw scans consume these before the u32 queue.
        pub fn with_deferred_raw_results(
            mut self,
            results: impl IntoIterator<Item = Vec<u8>>,
        ) -> Self {
            self.deferred_raw_results = results.into_iter().collect();
            self
        }

        fn record_write(&mut self, address: u64, value: u32) {
            let mut log = self.log.lock().unwrap();
            log.writes.push((address, value));
            log.events.push(MockEvent::Write(address, value));
        }

        /// Apply a register write with hardware-accurate semantics where
        /// they matter: abstractcs.cmderr, and sbcs sberror/sbbusyerror,
        /// are write-one-to-clear, so error-clear writes and plain
        /// re-configuration writes do not silently wipe sticky errors.
        fn commit_register_write(&mut self, address: u64, value: u32) {
            if address == super::Abstractcs::get_mmio_address() {
                let entry = self.registers.entry(address).or_insert(0);
                let remaining_cmderr = (*entry >> 8) & !((value >> 8) & 0x7);
                *entry = (*entry & !(0x7 << 8)) | (remaining_cmderr << 8);
            } else if address == super::Sbcs::get_mmio_address() {
                let entry = self.registers.entry(address).or_insert(0);
                let kept_sberror = (*entry >> 12) & !((value >> 12) & 0x7);
                let kept_sbbusyerror = ((*entry >> 22) & 1) & !((value >> 22) & 1);
                let merged = (value & !(0x7 << 12) & !(1 << 22))
                    | (kept_sberror << 12)
                    | (kept_sbbusyerror << 22);
                *entry = merged;
            } else {
                self.registers.insert(address, value);
            }
        }

        fn data0_reads_scheduled(&self) -> bool {
            let data0 = super::Data0::get_mmio_address();
            self.log
                .lock()
                .unwrap()
                .scheduled_reads
                .iter()
                .any(|address| *address == data0)
        }
    }

    impl DtmAccess for MockDtmAccess {
        fn target_reset_assert(&mut self) -> Result<(), DebugProbeError> {
            Ok(())
        }

        fn target_reset_deassert(&mut self) -> Result<(), DebugProbeError> {
            Ok(())
        }

        fn clear_error_state(&mut self) -> Result<(), RiscvError> {
            Ok(())
        }

        fn read_deferred_result(
            &mut self,
            _index: DeferredResultIndex,
        ) -> Result<CommandResult, RiscvError> {
            if let Some(bytes) = self.deferred_raw_results.pop_front() {
                return Ok(CommandResult::VecU8(bytes));
            }
            self.deferred_results
                .pop_front()
                .map(CommandResult::U32)
                .ok_or(RiscvError::BatchedResultNotAvailable)
        }

        fn execute(&mut self) -> Result<(), RiscvError> {
            {
                let mut log = self.log.lock().unwrap();
                log.executes += 1;
                log.events.push(MockEvent::Execute);
            }
            let pending = std::mem::take(&mut self.pending_writes);
            for (address, value) in pending {
                self.commit_register_write(address, value);
            }
            Ok(())
        }

        fn schedule_write(
            &mut self,
            address: u64,
            value: u32,
        ) -> Result<Option<DeferredResultIndex>, RiscvError> {
            self.record_write(address, value);
            self.pending_writes.push((address, value));
            Ok(Some(DeferredResultIndex::new_for_testing()))
        }

        fn schedule_read(&mut self, address: u64) -> Result<DeferredResultIndex, RiscvError> {
            let mut log = self.log.lock().unwrap();
            log.scheduled_reads.push(address);
            log.events.push(MockEvent::ScheduledRead(address));
            Ok(DeferredResultIndex::new_for_testing())
        }

        fn read_with_timeout(
            &mut self,
            address: u64,
            _timeout: Duration,
        ) -> Result<u32, RiscvError> {
            {
                let mut log = self.log.lock().unwrap();
                log.events.push(MockEvent::Read(address));
            }
            // A command state scripted to appear with the first batch's
            // completion overrides the plain register map.
            if address == super::Abstractcs::get_mmio_address()
                && self.abstractcs_after_data_reads.is_some()
                && self.data0_reads_scheduled()
            {
                return Ok(self.abstractcs_after_data_reads.unwrap());
            }
            // Scripted responses model memory data arriving in data0 or
            // sbdata0; the command machinery's own register reads
            // (dmcontrol, abstractcs) see the plain register map so
            // commands complete immediately and cleanly.
            if address == super::Data0::get_mmio_address()
                || address == super::Sbdata0::get_mmio_address()
            {
                if let Some(value) = self.read_responses.pop_front() {
                    return Ok(value);
                }
            }
            Ok(self.registers.get(&address).copied().unwrap_or(0))
        }

        fn write_with_timeout(
            &mut self,
            address: u64,
            value: u32,
            _timeout: Duration,
        ) -> Result<Option<u32>, RiscvError> {
            self.record_write(address, value);
            self.commit_register_write(address, value);
            Ok(Some(0))
        }

        fn read_idcode(&mut self) -> Result<Option<u32>, DebugProbeError> {
            Ok(None)
        }

        fn schedule_raw_read(&mut self, address: u64) -> Result<DeferredResultIndex, RiscvError> {
            let mut log = self.log.lock().unwrap();
            log.scheduled_reads.push(address);
            log.events.push(MockEvent::ScheduledRead(address));
            Ok(DeferredResultIndex::new_for_testing())
        }

        fn schedule_raw_noop(&mut self) -> Result<DeferredResultIndex, RiscvError> {
            let mut log = self.log.lock().unwrap();
            log.scheduled_reads.push(u64::MAX);
            log.events.push(MockEvent::ScheduledRead(u64::MAX));
            Ok(DeferredResultIndex::new_for_testing())
        }

        fn schedule_raw_read_uncaptured(&mut self, address: u64) -> Result<(), RiscvError> {
            let mut log = self.log.lock().unwrap();
            log.scheduled_reads.push(address);
            log.events.push(MockEvent::ScheduledRead(address));
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_mocks::{MockDtmAccess, MockEvent, MockLog};
    use super::*;

    fn make_interface(
        mock: MockDtmAccess,
        state: &'_ mut RiscvCommunicationInterfaceState,
    ) -> RiscvCommunicationInterface<'_> {
        RiscvCommunicationInterface::new(Box::new(mock), state)
    }

    #[test]
    fn abstract_access_memory_command_encoding() {
        // The constants must match the AccessMemoryCommand field layout
        // bit for bit: cmdtype 2 (access memory), aamsize 2 (32-bit),
        // physical addressing (aamvirtual clear), post-increment set,
        // and the target-specific bits [15:14] left clear.
        let mut read = AccessMemoryCommand(0);
        read.set_cmd_type(2);
        read.set_aamsize(2);
        read.set_aampostincrement(true);

        assert_eq!(ACCESS_MEMORY_READ, read.0);
        assert_eq!(ACCESS_MEMORY_READ, 0x0228_0000);
        assert_eq!((ACCESS_MEMORY_READ >> 14) & 0x3, 0);

        let mut write = AccessMemoryCommand(0);
        write.set_cmd_type(2);
        write.set_aamsize(2);
        write.set_aampostincrement(true);
        write.set_write(true);

        // The write command differs by the write bit alone.
        assert_eq!(ACCESS_MEMORY_WRITE, write.0);
        assert_eq!(ACCESS_MEMORY_WRITE ^ ACCESS_MEMORY_READ, 1 << 16);
    }

    #[test]
    fn force_memory_access_method_overrides_every_width() {
        let mut state = RiscvCommunicationInterfaceState::new();
        let mut interface = make_interface(MockDtmAccess::default(), &mut state);

        interface.force_memory_access_method(Some(MemoryAccessMethod::AbstractCommand));
        for width in [
            RiscvBusAccess::A8,
            RiscvBusAccess::A16,
            RiscvBusAccess::A32,
            RiscvBusAccess::A64,
            RiscvBusAccess::A128,
        ] {
            assert_eq!(
                interface.state.memory_access_method(width),
                MemoryAccessMethod::AbstractCommand
            );
        }
    }

    #[test]
    fn force_memory_access_method_none_restores_default() {
        let mut state = RiscvCommunicationInterfaceState::new();
        let mut interface = make_interface(MockDtmAccess::default(), &mut state);

        interface.force_memory_access_method(Some(MemoryAccessMethod::SystemBus));
        interface.force_memory_access_method(None);
        for width in [
            RiscvBusAccess::A8,
            RiscvBusAccess::A16,
            RiscvBusAccess::A32,
            RiscvBusAccess::A64,
            RiscvBusAccess::A128,
        ] {
            assert_eq!(
                interface.state.memory_access_method(width),
                MemoryAccessMethod::ProgramBuffer
            );
        }
    }

    #[test]
    fn memory_dispatch_completes_via_abstract_command() {
        // Regression for the removed unimplemented!() panics: with the
        // method forced to abstract commands, every dispatch entry point
        // completes against the mock instead of panicking.
        let mut state = RiscvCommunicationInterfaceState::new();
        let mut interface = make_interface(
            MockDtmAccess::default().with_read_responses([0x11, 0x22, 0x33]),
            &mut state,
        );
        interface.force_memory_access_method(Some(MemoryAccessMethod::AbstractCommand));

        assert_eq!(interface.read_word::<u32>(0x1000).unwrap(), 0x11);

        let mut words = [0u32; 2];
        interface.read_multiple(0x1000, &mut words).unwrap();
        assert_eq!(words, [0x22, 0x33]);

        interface.write_word(0x1000, 0x44u32).unwrap();

        let mut state = RiscvCommunicationInterfaceState::new();
        let mut interface = make_interface(MockDtmAccess::default(), &mut state);
        interface.force_memory_access_method(Some(MemoryAccessMethod::AbstractCommand));
        interface.write_multiple(0x1000, &[0x55u32, 0x66]).unwrap();
    }

    #[test]
    fn abstract_read_multiple_writes_arg1_once() {
        let mut state = RiscvCommunicationInterfaceState::new();
        let mock = MockDtmAccess::default().with_read_responses([10, 20, 30]);
        let log = mock.log.clone();
        let mut interface = make_interface(mock, &mut state);
        interface.force_memory_access_method(Some(MemoryAccessMethod::AbstractCommand));

        let mut words = [0u32; 3];
        interface.read_multiple(0x2000, &mut words).unwrap();
        assert_eq!(words, [10, 20, 30]);

        let log = log.lock().unwrap();
        let arg1_writes: Vec<_> = log
            .writes
            .iter()
            .filter(|(address, _)| *address == Data1::get_mmio_address())
            .collect();
        // The start address is loaded into arg1 exactly once; the
        // post-increment carries it from there.
        assert_eq!(arg1_writes.len(), 1);
        assert_eq!(arg1_writes[0], &(Data1::get_mmio_address(), 0x2000u32));
        // One access-memory command per word.
        let command_writes: Vec<_> = log
            .writes
            .iter()
            .filter(|(address, _)| *address == Command::get_mmio_address())
            .collect();
        assert_eq!(command_writes.len(), 3);
        assert!(command_writes
            .iter()
            .all(|&(_, value)| *value == ACCESS_MEMORY_READ));
    }

    #[test]
    fn abstract_write_multiple_pinned() {
        let mut state = RiscvCommunicationInterfaceState::new();
        let mock = MockDtmAccess::default();
        let log = mock.log.clone();
        let mut interface = make_interface(mock, &mut state);
        interface.force_memory_access_method(Some(MemoryAccessMethod::AbstractCommand));

        interface
            .write_multiple(0x3000, &[0xAAu32, 0xBB, 0xCC])
            .unwrap();

        let log = log.lock().unwrap();
        // The start address is loaded into arg1 exactly once.
        let arg1_writes: Vec<_> = log
            .writes
            .iter()
            .filter(|(address, _)| *address == Data1::get_mmio_address())
            .collect();
        assert_eq!(arg1_writes.len(), 1);
        assert_eq!(arg1_writes[0], &(Data1::get_mmio_address(), 0x3000));

        // Each word's value goes through data0, in order, before its
        // write command.
        let data0_writes: Vec<_> = log
            .writes
            .iter()
            .filter(|(address, _)| *address == Data0::get_mmio_address())
            .map(|&(_, value)| value)
            .collect();
        assert_eq!(data0_writes, vec![0xAA, 0xBB, 0xCC]);

        let command_writes: Vec<_> = log
            .writes
            .iter()
            .filter(|(address, _)| *address == Command::get_mmio_address())
            .map(|&(_, value)| value)
            .collect();
        assert_eq!(command_writes.len(), 3);
        assert!(command_writes
            .iter()
            .all(|value| *value == ACCESS_MEMORY_WRITE));
    }

    #[test]
    fn abstract_memory_access_rejects_narrow_widths() {
        // A forced abstract method with a sub-word width must fail loudly
        // instead of degrading into a 32-bit access that corrupts the
        // neighbouring bytes.
        let mut state = RiscvCommunicationInterfaceState::new();
        let mut interface = make_interface(MockDtmAccess::default(), &mut state);
        interface.force_memory_access_method(Some(MemoryAccessMethod::AbstractCommand));

        let result = interface.read_word::<u8>(0x1000);
        assert!(matches!(
            result,
            Err(crate::Error::Riscv(RiscvError::UnsupportedBusAccessWidth(
                RiscvBusAccess::A8
            )))
        ));

        let result = interface.write_word(0x1000, 0x1Au16);
        assert!(matches!(
            result,
            Err(crate::Error::Riscv(RiscvError::UnsupportedBusAccessWidth(
                RiscvBusAccess::A16
            )))
        ));
    }

    /// Segment the transport timeline at execute() flushes and count the
    /// data0 reads queued inside each segment: those are the batch
    /// sizes of the program-buffer read path.
    fn progbuf_batch_sizes(log: &MockLog) -> Vec<usize> {
        let data0 = Data0::get_mmio_address();
        let mut sizes = Vec::new();
        let mut current = 0;
        for event in &log.events {
            match event {
                MockEvent::ScheduledRead(address) if *address == data0 => current += 1,
                MockEvent::Execute => {
                    if current > 0 {
                        sizes.push(current);
                    }
                    current = 0;
                }
                _ => {}
            }
        }
        sizes
    }

    fn progbuf_mock(
        word_count: usize,
    ) -> (MockDtmAccess, std::sync::Arc<std::sync::Mutex<MockLog>>) {
        let mock = MockDtmAccess::default().with_deferred_results(0..word_count as u32);
        let log = mock.log.clone();
        (mock, log)
    }

    #[test]
    fn progbuf_bulk_read_splits_into_batches_of_32() {
        assert_eq!(WORDS_PER_BATCH, 32);

        for (words, expected) in [
            (100usize, vec![32usize, 32, 32, 4]),
            (32, vec![32]),
            (1, vec![1]),
        ] {
            let mut state = RiscvCommunicationInterfaceState::new();
            state.progbuf_size = 16;
            let (mock, log) = progbuf_mock(words);
            let mut interface = make_interface(mock, &mut state);

            let mut data = vec![0u32; words];
            interface
                .perform_memory_read_multiple_progbuf(0x2000, &mut data, RISCV_TIMEOUT)
                .unwrap();

            // Every deferred result lands in its own slot, in order.
            let expected_values: Vec<u32> = (0..words as u32).collect();
            assert_eq!(&data, &expected_values, "{words} words");
            let log = log.lock().unwrap();
            assert_eq!(progbuf_batch_sizes(&log), expected, "{words} words");
            // Exactly one flush per batch, no stray trailing flush.
            assert_eq!(log.executes, expected.len(), "{words} words");
        }
    }

    #[test]
    fn progbuf_batch_checks_cmderr_before_next_batch() {
        let mut state = RiscvCommunicationInterfaceState::new();
        state.progbuf_size = 16;
        let (mock, log) = progbuf_mock(100);
        let mut interface =
            make_interface(mock.with_abstractcs_after_data_reads(1 << 8), &mut state);

        let mut data = vec![0u32; 100];
        let result =
            interface.perform_memory_read_multiple_progbuf(0x2000, &mut data, RISCV_TIMEOUT);

        assert!(matches!(
            result,
            Err(RiscvError::AbstractCommand(AbstractCommandErrorKind::Busy))
        ));
        // The error surfaced after the FIRST batch: no later batch was
        // ever queued.
        assert_eq!(progbuf_batch_sizes(&log.lock().unwrap()), vec![32]);
    }

    #[test]
    fn progbuf_batch_times_out_when_abstractcs_stays_busy() {
        let mut state = RiscvCommunicationInterfaceState::new();
        state.progbuf_size = 16;
        let (mock, log) = progbuf_mock(70);
        let mut interface =
            make_interface(mock.with_abstractcs_after_data_reads(1 << 12), &mut state);

        let mut data = vec![0u32; 70];
        let start = Instant::now();
        let result = interface.perform_memory_read_multiple_progbuf(
            0x2000,
            &mut data,
            Duration::from_millis(100),
        );

        assert!(matches!(result, Err(RiscvError::Timeout)));
        // The injected bound governs, not the default five seconds.
        assert!(start.elapsed() < Duration::from_secs(2));
        // The poll hung on the FIRST batch's completion.
        assert_eq!(progbuf_batch_sizes(&log.lock().unwrap()), vec![32]);
    }

    #[test]
    fn progbuf_bulk_read_last_word_executes_no_trailing_load() {
        // Every read command except the last one re-arms the program
        // buffer, whose execution loads the NEXT word. The last word's
        // command must leave postexec off: the trailing load would read
        // one word past the requested range and fault (latching cmderr)
        // when the range ends at a memory-region boundary.
        let mut state = RiscvCommunicationInterfaceState::new();
        state.progbuf_size = 16;
        let (mock, log) = progbuf_mock(3);
        let mut interface = make_interface(mock, &mut state);

        let mut data = vec![0u32; 3];
        interface
            .perform_memory_read_multiple_progbuf(0x2000, &mut data, RISCV_TIMEOUT)
            .unwrap();
        assert_eq!(&data, &[0, 1, 2]);

        let cmd_addr = AccessRegisterCommand::get_mmio_address();
        let postexec_bit = 1 << 18;
        let commands: Vec<u32> = log
            .lock()
            .unwrap()
            .events
            .iter()
            .filter_map(|event| match event {
                MockEvent::Write(address, value) if *address == cmd_addr => Some(*value),
                _ => None,
            })
            .collect();
        // Command-register layout: save s0, save s1, the setup write to
        // s0, one read per word, then the register restores.
        assert_eq!(commands.len(), 8);
        let word_reads = &commands[3..6];
        for command in &word_reads[..2] {
            assert!(
                command & postexec_bit != 0,
                "postexec missing on a non-final word: {command:#x}"
            );
        }
        assert!(
            word_reads[2] & postexec_bit == 0,
            "postexec set on the final word's command: {:#x}",
            word_reads[2]
        );
    }

    #[test]
    fn sysbus_batch_scan_count_matches_pipeline_depth() {
        assert_eq!(RiscvCommunicationInterface::sysbus_batch_scan_count(0), 0);
        assert_eq!(RiscvCommunicationInterface::sysbus_batch_scan_count(1), 0);
        assert_eq!(RiscvCommunicationInterface::sysbus_batch_scan_count(2), 2);
        assert_eq!(RiscvCommunicationInterface::sysbus_batch_scan_count(4), 4);
        assert_eq!(RiscvCommunicationInterface::sysbus_batch_scan_count(64), 64);
    }

    #[test]
    fn sysbus_batch_data_from_scans_fills_by_role() {
        let mk = |data: u32, op: u8, role: ScanRole| ScanOutcome {
            response: RawDmiResponse {
                data,
                op,
                raw: vec![],
            },
            role,
        };
        // Batch for N=4: three data-bearing scans (two reads plus the
        // no-op tail) and one ignored filler.
        let scans = vec![
            mk(0xA500_0000, 0, ScanRole::Data(0)),
            mk(0xA501_0101, 0, ScanRole::Data(1)),
            mk(0xA702_0202, 0, ScanRole::Data(2)),
            mk(0xdead_beef, 0, ScanRole::Ignored),
        ];
        let mut out = [0u32; 3];
        RiscvCommunicationInterface::sysbus_batch_data_from_scans(&scans, &mut out).unwrap();
        assert_eq!(out, [0xA500_0000, 0xA501_0101, 0xA702_0202]);
    }

    #[test]
    fn sysbus_batch_data_from_scans_rejects_stalled_response() {
        let scans = vec![ScanOutcome {
            response: RawDmiResponse {
                data: 1,
                op: 3,
                raw: vec![],
            },
            role: ScanRole::Data(0),
        }];
        let mut out = [0u32; 1];
        assert!(
            RiscvCommunicationInterface::sysbus_batch_data_from_scans(&scans, &mut out).is_err()
        );
    }

    #[test]
    fn decode_raw_response_extracts_op_and_data() {
        // The DMI response places the data in bits [33:2] and the op
        // status in bits [1:0]; encode data 0x1234_5678 with op 0b01.
        let value: u128 = (0x1234_5678u64 as u128) << 2 | 0b01;
        let bytes = value.to_le_bytes()[..6].to_vec();

        let decoded = decode_raw_response(CommandResult::VecU8(bytes.clone())).unwrap();
        assert_eq!(decoded.data, 0x1234_5678);
        assert_eq!(decoded.op, 0b01);
        assert_eq!(decoded.raw, bytes);
    }

    #[test]
    fn decode_raw_response_rejects_non_byte_result() {
        assert!(matches!(
            decode_raw_response(CommandResult::U32(0)),
            Err(RiscvError::BatchedResultNotAvailable)
        ));
    }

    #[test]
    fn sysbus_batch_partition_covers_tail_batches() {
        let plan = |word_count, word_offset, address_offset_bytes| SysbusBatchPlan {
            word_count,
            word_offset,
            address_offset_bytes,
        };
        assert_eq!(sysbus_batch_partition(1, 4), vec![plan(1, 0, 0)]);
        assert_eq!(sysbus_batch_partition(2, 4), vec![plan(2, 0, 0)]);
        assert_eq!(sysbus_batch_partition(64, 4), vec![plan(64, 0, 0)]);
        assert_eq!(
            sysbus_batch_partition(65, 4),
            vec![plan(64, 0, 0), plan(1, 64, 256)]
        );
        // Narrower words advance the bus address proportionally.
        assert_eq!(
            sysbus_batch_partition(65, 2),
            vec![plan(64, 0, 0), plan(1, 64, 128)]
        );
    }

    #[test]
    fn sysbus_read_pipeline_end_to_end_via_mock() {
        // Five words: the batch queues five scans (one uncaptured priming
        // read, three captured reads, one captured no-op tail); scan k
        // carries data[k-1], so the raw captures yield words 0..3 and the
        // final word arrives from the post-stop sbdata0 read.
        let words: Vec<u32> = vec![0x100, 0x200, 0x300, 0x400, 0x500];
        let raw_capture = |data: u32| -> Vec<u8> {
            let value = (data as u128) << 2;
            value.to_le_bytes()[..6].to_vec()
        };
        let mock = MockDtmAccess::default()
            .with_deferred_raw_results(words[..4].iter().map(|w| raw_capture(*w)))
            .with_read_responses([words[4]]);
        let log = mock.log.clone();

        let mut state = RiscvCommunicationInterfaceState::new();
        let mut interface = make_interface(mock, &mut state);

        let mut data = [0u32; 5];
        interface
            .perform_memory_read_multiple_sysbus(0x8000_0000, &mut data)
            .unwrap();
        assert_eq!(&data, &words[..]);

        let log = log.lock().unwrap();
        let sbcs_addr = Sbcs::get_mmio_address();
        let sbdata_addr = Sbdata0::get_mmio_address();

        // The engine is stopped (readondata and autoincrement cleared)
        // BEFORE the final sbdata0 read - reading data with the engine
        // still armed would trigger an access past the range.
        let final_read = log
            .events
            .iter()
            .rposition(|event| matches!(event, MockEvent::Read(addr) if *addr == sbdata_addr))
            .expect("final sbdata0 read");
        let stop_written_before = log.events[..final_read].iter().any(|event| {
            matches!(
                *event,
                MockEvent::Write(addr, value)
                    if addr == sbcs_addr
                        && (value >> 15) & 1 == 0
                        && (value >> 16) & 1 == 0
                        && (value >> 17) & 0x7 == 2
            )
        });
        assert!(
            stop_written_before,
            "engine must be stopped before the final word is read"
        );

        // Exactly the planned scans were queued: three captured reads and
        // one uncaptured priming read on sbdata0, plus one no-op tail.
        let sbdata_scheduled = log
            .scheduled_reads
            .iter()
            .filter(|addr| **addr == sbdata_addr)
            .count();
        assert_eq!(sbdata_scheduled, 4);
        let noop_scheduled = log
            .scheduled_reads
            .iter()
            .filter(|addr| **addr == u64::MAX)
            .count();
        assert_eq!(noop_scheduled, 1);
    }

    #[test]
    fn sysbus_read_pipeline_abandons_engine_on_scan_error() {
        // A failed pipeline member must not leave the engine armed: the
        // error path waits out sbbusy and writes the stop configuration,
        // so a caller retry never writes sbcs against a busy engine
        // (undefined behavior per the SBCS contract).
        let raw_capture = |data: u32, op: u128| -> Vec<u8> {
            let value = ((data as u128) << 2) | op;
            value.to_le_bytes()[..6].to_vec()
        };
        let mock = MockDtmAccess::default().with_deferred_raw_results([
            raw_capture(0x100, 0),
            raw_capture(0x200, 2), // stalled op surfaces as an error
            raw_capture(0, 0),     // tail no-op response
        ]);
        let log = mock.log.clone();

        let mut state = RiscvCommunicationInterfaceState::new();
        let mut interface = make_interface(mock, &mut state);

        let mut data = [0u32; 3];
        let result = interface.perform_memory_read_multiple_sysbus(0x8000_0000, &mut data);
        assert!(matches!(result, Err(RiscvError::SystemBusAccess)));

        let log = log.lock().unwrap();
        let sbcs_addr = Sbcs::get_mmio_address();
        let last_sbcs_write = log.events.iter().rev().find_map(|event| match event {
            MockEvent::Write(addr, value) if *addr == sbcs_addr => Some(*value),
            _ => None,
        });
        // The last sbcs write is the stop configuration the abandon path
        // leaves behind: the access-triggering bits are off.
        let stop = last_sbcs_write.expect("a stop write after the error");
        assert_eq!((stop >> 15) & 1, 0, "readondata must be off: {stop:#x}");
        assert_eq!((stop >> 16) & 1, 0, "autoincrement must be off: {stop:#x}");
    }

    #[test]
    fn sysbus_read_pipeline_multi_batch_advances_address() {
        // 65 words span two batches (64 + 1). Batch 1 fills words 0..62
        // from raw captures, its tail word 63 from the post-stop read;
        // batch 2 re-arms and reads word 64 the same way. This pins the
        // driver's consumption of the batch plan: per-batch output
        // slicing and the per-batch bus address.
        let word = |i: u32| 0xA000_0000 + i;
        let raw_capture = |data: u32| -> Vec<u8> {
            let value = (data as u128) << 2;
            value.to_le_bytes()[..6].to_vec()
        };
        let mock = MockDtmAccess::default()
            .with_deferred_raw_results((0..63).map(|i| raw_capture(word(i))))
            .with_read_responses([word(63), word(64)]);
        let log = mock.log.clone();

        let mut state = RiscvCommunicationInterfaceState::new();
        let mut interface = make_interface(mock, &mut state);

        let mut data = [0u32; 65];
        interface
            .perform_memory_read_multiple_sysbus(0x8000_0000, &mut data)
            .unwrap();

        // Every word lands in its own slot across the batch boundary.
        let expected: Vec<u32> = (0..65).map(word).collect();
        assert_eq!(&data[..], &expected[..]);

        let log = log.lock().unwrap();
        let sbaddress0_writes: Vec<u32> = log
            .writes
            .iter()
            .filter(|(address, _)| *address == Sbaddress0::get_mmio_address())
            .map(|&(_, value)| value)
            .collect();
        assert_eq!(sbaddress0_writes.len(), 2);
        // The second batch starts one batch's words past the range start.
        assert_eq!(sbaddress0_writes[0], 0x8000_0000);
        assert_eq!(sbaddress0_writes[1], 0x8000_0000 + 64 * 4);

        // The pipeline is re-armed for the second batch (readondata and
        // autoincrement set again after the first batch's stop write).
        let rearm_count = log
            .writes
            .iter()
            .filter(|(address, value)| {
                *address == Sbcs::get_mmio_address()
                    && (value >> 15) & 1 == 1
                    && (value >> 16) & 1 == 1
            })
            .count();
        assert_eq!(rearm_count, 2);
    }

    #[test]
    fn sysbus_read_pipeline_clears_sticky_error_on_failure() {
        // A sticky sberror left in the engine surfaces as an error, and
        // the failure path clears exactly the sticky bits (write-one-to-
        // clear) after the final word read so later system-bus work
        // starts clean.
        let word = |i: u32| 0xB000_0000 + i;
        let raw_capture = |data: u32| -> Vec<u8> {
            let value = (data as u128) << 2;
            value.to_le_bytes()[..6].to_vec()
        };
        let mut mock = MockDtmAccess::default()
            .with_deferred_raw_results((0..4).map(|i| raw_capture(word(i))))
            .with_read_responses([word(4)]);
        // Seed a sticky sberror = 1; re-configuration writes do not clear
        // it (write-one-to-clear), so the final status check sees it.
        mock.registers.insert(Sbcs::get_mmio_address(), 1 << 12);
        let log = mock.log.clone();

        let mut state = RiscvCommunicationInterfaceState::new();
        let mut interface = make_interface(mock, &mut state);

        let mut data = [0u32; 5];
        let result = interface.perform_memory_read_multiple_sysbus(0x8000_0000, &mut data);
        assert!(matches!(result, Err(RiscvError::SystemBusAccess)));

        let log = log.lock().unwrap();
        // The clear write carries exactly the sticky bits that were set.
        let clear_writes: Vec<(usize, u32)> = log
            .events
            .iter()
            .enumerate()
            .filter_map(|(position, event)| match event {
                MockEvent::Write(address, value)
                    if *address == Sbcs::get_mmio_address() && *value == 1 << 12 =>
                {
                    Some((position, *value))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            clear_writes.len(),
            1,
            "exactly one clear write with only the sticky sberror bits"
        );

        // The clear happens after the final word was collected.
        let last_sbdata_read = log
            .events
            .iter()
            .rposition(|event| {
                matches!(event, MockEvent::Read(addr) if *addr == Sbdata0::get_mmio_address())
            })
            .expect("final sbdata0 read");
        assert!(clear_writes[0].0 > last_sbdata_read);
    }
}
