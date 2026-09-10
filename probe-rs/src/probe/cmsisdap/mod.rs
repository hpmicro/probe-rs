//! CMSIS-DAP probe implementation.
mod commands;
mod tools;

use anyhow::anyhow;

use crate::{
    architecture::{
        arm::{
            communication_interface::{DapProbe, UninitializedArmProbe},
            dp::{Abort, Ctrl},
            swo::poll_interval_from_buf_size,
            ArmCommunicationInterface, ArmError, DapError, Pins, PortType, RawDapAccess, Register,
            SwoAccess, SwoConfig, SwoMode,
        },
        riscv::{communication_interface::RiscvInterfaceBuilder, dtm::jtag_dtm::JtagDtmBuilder},
        xtensa::communication_interface::{
            XtensaCommunicationInterface, XtensaDebugInterfaceState,
        },
    },
    probe::{
        cmsisdap::commands::{
            general::info::{CapabilitiesCommand, PacketCountCommand, SWOTraceBufferSizeCommand},
            CmsisDapError,
        },
        common::{JtagDriverState, RawJtagIo},
        BatchCommand, DebugProbe, DebugProbeError, DebugProbeInfo, DebugProbeSelector, JTAGAccess,
        JtagChainItem, ProbeFactory, WireProtocol,
    },
    CoreStatus,
};

use commands::{
    general::{
        connect::{ConnectRequest, ConnectResponse},
        disconnect::{DisconnectRequest, DisconnectResponse},
        host_status::{HostStatusRequest, HostStatusResponse},
        info::Capabilities,
        reset::{ResetRequest, ResetResponse},
    },
    jtag::{
        configure::{
            ConfigureRequest as JtagConfigureRequest, ConfigureResponse as JtagConfigureResponse,
        },
        sequence::{
            Sequence as JtagSequence, SequenceRequest as JtagSequenceRequest,
            SequenceResponse as JtagSequenceResponse,
        },
    },
    swd,
    swj::{
        clock::{SWJClockRequest, SWJClockResponse},
        pins::{SWJPinsRequest, SWJPinsRequestBuilder, SWJPinsResponse},
        sequence::{SequenceRequest, SequenceResponse},
    },
    swo,
    transfer::{
        configure::{ConfigureRequest, ConfigureResponse},
        Ack, TransferBlockRequest, TransferBlockResponse, TransferRequest,
    },
    CmsisDapDevice, Status,
};
use probe_rs_target::ScanChainElement;

use std::{fmt::Write, time::Duration};

use bitvec::prelude::*;

use super::common::{extract_idcodes, extract_ir_lengths, ScanChainError};

/// A factory for creating [`CmsisDap`] probes.
#[derive(Debug)]
pub struct CmsisDapFactory;

impl std::fmt::Display for CmsisDapFactory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("CMSIS-DAP")
    }
}

impl ProbeFactory for CmsisDapFactory {
    fn open(&self, selector: &DebugProbeSelector) -> Result<Box<dyn DebugProbe>, DebugProbeError> {
        Ok(Box::new(CmsisDap::new_from_device(
            tools::open_device_from_selector(selector)?,
        )?))
    }

    fn list_probes(&self) -> Vec<DebugProbeInfo> {
        tools::list_cmsisdap_devices()
    }
}

/// A CMSIS-DAP probe.
pub struct CmsisDap {
    device: CmsisDapDevice,
    _hw_version: u8,
    _jtag_version: u8,
    protocol: Option<WireProtocol>,

    packet_size: u16,
    packet_count: u8,
    capabilities: Capabilities,
    swo_buffer_size: Option<usize>,
    swo_active: bool,
    swo_streaming: bool,
    connected: bool,

    /// Speed in kHz
    speed_khz: u32,
    scan_chain: Option<Vec<ScanChainElement>>,

    batch: Vec<BatchCommand>,

    jtag_driver_state: JtagDriverState,
    jtag_sequences: Vec<JtagSequence>,
}

impl std::fmt::Debug for CmsisDap {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fmt.debug_struct("CmsisDap")
            .field("protocol", &self.protocol)
            .field("packet_size", &self.packet_size)
            .field("packet_count", &self.packet_count)
            .field("capabilities", &self.capabilities)
            .field("swo_buffer_size", &self.swo_buffer_size)
            .field("swo_active", &self.swo_active)
            .field("swo_streaming", &self.swo_streaming)
            .field("speed_khz", &self.speed_khz)
            .finish()
    }
}

impl CmsisDap {
    fn new_from_device(mut device: CmsisDapDevice) -> Result<Self, DebugProbeError> {
        // Discard anything left in buffer, as otherwise
        // we'll get out of sync between requests and responses.
        device.drain();

        // Determine and set the packet size. We do this as soon as possible after
        // opening the probe to ensure all future communication uses the correct size.
        let packet_size = device.find_packet_size()? as u16;

        // Read remaining probe information.
        let packet_count = commands::send_command(&mut device, PacketCountCommand {})?;
        let caps: Capabilities = commands::send_command(&mut device, CapabilitiesCommand {})?;
        tracing::debug!("Detected probe capabilities: {:?}", caps);
        let mut swo_buffer_size = None;
        if caps.swo_uart_implemented || caps.swo_manchester_implemented {
            let swo_size = commands::send_command(&mut device, SWOTraceBufferSizeCommand {})?;
            swo_buffer_size = Some(swo_size as usize);
            tracing::debug!("Probe SWO buffer size: {}", swo_size);
        }
        Ok(Self {
            device,
            _hw_version: 0,
            _jtag_version: 0,
            protocol: None,
            packet_count,
            packet_size,
            capabilities: caps,
            swo_buffer_size,
            swo_active: false,
            swo_streaming: false,
            connected: false,
            speed_khz: 1_000,
            scan_chain: None,
            batch: Vec::new(),
            jtag_driver_state: JtagDriverState::default(),
            jtag_sequences: Vec::<JtagSequence>::new(),
        })
    }

    /// Set maximum JTAG/SWD clock frequency to use, in Hz.
    ///
    /// The actual clock frequency used by the device might be lower.
    fn set_swj_clock(&mut self, clock_hz: u32) -> Result<(), CmsisDapError> {
        commands::send_command(&mut self.device, SWJClockRequest(clock_hz))
            .map_err(CmsisDapError::from)
            .and_then(|v| match v {
                SWJClockResponse(Status::DAPOk) => Ok(()),
                SWJClockResponse(Status::DAPError) => Err(CmsisDapError::ErrorResponse),
            })
    }

    fn transfer_configure(&mut self, request: ConfigureRequest) -> Result<(), CmsisDapError> {
        commands::send_command(&mut self.device, request)
            .map_err(CmsisDapError::from)
            .and_then(|v| match v {
                ConfigureResponse(Status::DAPOk) => Ok(()),
                ConfigureResponse(Status::DAPError) => Err(CmsisDapError::ErrorResponse),
            })
    }

    fn configure_swd(
        &mut self,
        request: swd::configure::ConfigureRequest,
    ) -> Result<(), CmsisDapError> {
        commands::send_command(&mut self.device, request)
            .map_err(CmsisDapError::from)
            .and_then(|v| match v {
                swd::configure::ConfigureResponse(Status::DAPOk) => Ok(()),
                swd::configure::ConfigureResponse(Status::DAPError) => {
                    Err(CmsisDapError::ErrorResponse)
                }
            })
    }

    /// Reset JTAG state machine to Test-Logic-Reset.
    fn jtag_ensure_test_logic_reset(&mut self) -> Result<(), CmsisDapError> {
        let sequence = JtagSequence::no_capture(true, &bitvec![u8, Lsb0; 0; 6])?;
        let sequences = vec![sequence];

        self.send_jtag_sequences(JtagSequenceRequest::new(sequences)?)?;

        Ok(())
    }

    /// Reset JTAG state machine to Run-Test/Idle, as requisite precondition for DAP_Transfer commands.
    fn jtag_ensure_run_test_idle(&mut self) -> Result<(), CmsisDapError> {
        // These could be coalesced into one sequence request, but for now we'll keep things simple.

        // First reach Test-Logic-Reset
        self.jtag_ensure_test_logic_reset()?;

        // Then transition to Run-Test-Idle
        let sequence = JtagSequence::no_capture(false, &bitvec![u8, Lsb0; 0; 1])?;
        let sequences = vec![sequence];
        self.send_jtag_sequences(JtagSequenceRequest::new(sequences)?)?;

        Ok(())
    }

    /// Scan JTAG chain, detecting TAPs and their IDCODEs and IR lengths.
    ///
    /// If IR lengths for each TAP are known, provide them in `ir_lengths`.
    ///
    /// Returns a new JTAG chain.
    fn jtag_scan(
        &mut self,
        ir_lengths: Option<&[usize]>,
    ) -> Result<Vec<JtagChainItem>, CmsisDapError> {
        let (ir, dr) = self.jtag_reset_scan()?;
        let idcodes = extract_idcodes(&dr)?;
        let ir_lens = extract_ir_lengths(&ir, idcodes.len(), ir_lengths)?;

        Ok(idcodes
            .into_iter()
            .zip(ir_lens)
            .map(|(idcode, irlen)| JtagChainItem { irlen, idcode })
            .collect())
    }

    /// Capture the power-up scan chain values, including all IDCODEs.
    ///
    /// Returns the IR and DR results as (IR, DR).
    fn jtag_reset_scan(&mut self) -> Result<(BitVec<u8>, BitVec<u8>), CmsisDapError> {
        let dr = self.jtag_scan_dr()?;
        let ir = self.jtag_scan_ir()?;

        // Return to Run-Test/Idle, so the probe is ready for DAP_Transfer commands again.
        self.jtag_ensure_run_test_idle()?;

        Ok((ir, dr))
    }

    /// Detect the IR chain length and return its current contents.
    ///
    /// Replaces the current contents with all 1s (BYPASS) and enters
    /// the Run-Test/Idle state.
    fn jtag_scan_ir(&mut self) -> Result<BitVec<u8>, CmsisDapError> {
        self.jtag_ensure_shift_ir()?;
        let data = self.jtag_scan_inner("IR")?;
        Ok(data)
    }

    /// Detect the DR chain length and return its contents.
    ///
    /// Replaces the current contents with all 1s and enters
    /// the Run-Test/Idle state.
    fn jtag_scan_dr(&mut self) -> Result<BitVec<u8>, CmsisDapError> {
        self.jtag_ensure_shift_dr()?;
        let data = self.jtag_scan_inner("DR")?;
        Ok(data)
    }

    /// Detect current chain length and return its contents.
    /// Must already be in either Shift-IR or Shift-DR state.
    fn jtag_scan_inner(&mut self, name: &'static str) -> Result<BitVec<u8>, CmsisDapError> {
        // Max scan chain length (in bits) to attempt to detect.
        const MAX_LENGTH: usize = 128;
        // How many bytes to write out / read in per request.
        const BYTES_PER_REQUEST: usize = 16;
        // How many requests are needed to read/write at least MAX_LENGTH bits.
        const REQUESTS: usize =
            (MAX_LENGTH + (BYTES_PER_REQUEST * 8 - 1)) / (BYTES_PER_REQUEST * 8);

        // Completely fill xR with 0s, capture result.
        let mut tdo_bytes: Vec<u8> = Vec::with_capacity(REQUESTS * BYTES_PER_REQUEST);
        for _ in 0..REQUESTS {
            let sequences = vec![
                JtagSequence::capture(false, &bitvec![u8, Lsb0; 0; 64])?,
                JtagSequence::capture(false, &bitvec![u8, Lsb0; 0; 64])?,
            ];

            tdo_bytes.extend(
                self.send_jtag_sequences(JtagSequenceRequest::new(sequences)?)?
                    .iter(),
            );
        }
        let d0 = tdo_bytes.view_bits::<Lsb0>();

        // Completely fill xR with 1s, capture result.
        let mut tdo_bytes: Vec<u8> = Vec::with_capacity(REQUESTS * BYTES_PER_REQUEST);
        for _ in 0..REQUESTS {
            let sequences = vec![
                JtagSequence::capture(false, &bitvec![u8, Lsb0; 1; 64])?,
                JtagSequence::capture(false, &bitvec![u8, Lsb0; 1; 64])?,
            ];

            tdo_bytes.extend(
                self.send_jtag_sequences(JtagSequenceRequest::new(sequences)?)?
                    .iter(),
            );
        }
        let d1 = tdo_bytes.view_bits::<Lsb0>();

        // Find first 1 in d1, which indicates length of register.
        let n = match d1.first_one() {
            Some(n) => {
                tracing::info!("JTAG {name} scan chain detected as {n} bits long");
                n
            }
            None => {
                tracing::error!(
                    "JTAG {name} scan chain either broken or too long: did not detect 1"
                );
                return Err(CmsisDapError::ErrorResponse);
            }
        };

        // Check at least one register is detected in the scan chain.
        if n == 0 {
            tracing::error!("JTAG {name} scan chain is empty");
            return Err(CmsisDapError::ErrorResponse);
        }

        // Check d0[n..] are all 0.
        if d0[n..].any() {
            tracing::error!("JTAG {name} scan chain either broken or too long: did not detect 0");
            return Err(CmsisDapError::ErrorResponse);
        }

        // Extract d0[..n] as the initial scan chain contents.
        let data = d0[..n].to_bitvec();

        Ok(data)
    }

    fn jtag_ensure_shift_dr(&mut self) -> Result<(), CmsisDapError> {
        // Transition to Test-Logic-Reset.
        self.jtag_ensure_test_logic_reset()?;

        // Transition to Shift-DR
        let sequences = vec![
            JtagSequence::no_capture(false, &bitvec![u8, Lsb0; 0; 1])?,
            JtagSequence::no_capture(true, &bitvec![u8, Lsb0; 0; 1])?,
            JtagSequence::no_capture(false, &bitvec![u8, Lsb0; 0; 2])?,
        ];
        self.send_jtag_sequences(JtagSequenceRequest::new(sequences)?)?;

        Ok(())
    }

    fn jtag_ensure_shift_ir(&mut self) -> Result<(), CmsisDapError> {
        // Transition to Test-Logic-Reset.
        self.jtag_ensure_test_logic_reset()?;

        // Transition to Shift-IR
        let sequences = vec![
            JtagSequence::no_capture(false, &bitvec![u8, Lsb0; 0; 1])?,
            JtagSequence::no_capture(true, &bitvec![u8, Lsb0; 0; 2])?,
            JtagSequence::no_capture(false, &bitvec![u8, Lsb0; 0; 2])?,
        ];
        self.send_jtag_sequences(JtagSequenceRequest::new(sequences)?)?;

        Ok(())
    }

    fn send_jtag_configure(&mut self, request: JtagConfigureRequest) -> Result<(), CmsisDapError> {
        commands::send_command(&mut self.device, request)
            .map_err(CmsisDapError::from)
            .and_then(|v| match v {
                JtagConfigureResponse(Status::DAPOk) => Ok(()),
                JtagConfigureResponse(Status::DAPError) => Err(CmsisDapError::ErrorResponse),
            })
    }

    fn send_jtag_sequences(
        &mut self,
        request: JtagSequenceRequest,
    ) -> Result<Vec<u8>, CmsisDapError> {
        commands::send_command(&mut self.device, request)
            .map_err(CmsisDapError::from)
            .and_then(|v| match v {
                JtagSequenceResponse(Status::DAPOk, tdo) => Ok(tdo),
                JtagSequenceResponse(Status::DAPError, _) => Err(CmsisDapError::ErrorResponse),
            })
    }

    fn send_swj_sequences(&mut self, request: SequenceRequest) -> Result<(), CmsisDapError> {
        // Ensure all pending commands are processed.
        //self.process_batch()?;

        commands::send_command(&mut self.device, request)
            .map_err(CmsisDapError::from)
            .and_then(|v| match v {
                SequenceResponse(Status::DAPOk) => Ok(()),
                SequenceResponse(Status::DAPError) => Err(CmsisDapError::ErrorResponse),
            })
    }

    /// Read the CTRL register from the currently selected debug port.
    ///
    /// According to the ARM specification, this *should* never fail.
    /// In practice, it can unfortunately happen.
    ///
    /// To avoid an endless recursion in this cases, this function is provided
    /// as an alternative to [`Self::process_batch()`]. This function will return any errors,
    /// and not retry any transfers.
    fn read_ctrl_register(&mut self) -> Result<Ctrl, ArmError> {
        let response = commands::send_command(
            &mut self.device,
            TransferRequest::read(PortType::DebugPort, Ctrl::ADDRESS),
        )
        .map_err(CmsisDapError::from)
        .map_err(DebugProbeError::from)?;

        // We can assume that the single transfer is always executed,
        // no need to check here.

        if response.last_transfer_response.protocol_error {
            // TODO: What does this protocol error mean exactly?
            //       Should be verified in CMSIS-DAP spec
            Err(DapError::SwdProtocol.into())
        } else {
            if response.last_transfer_response.ack != Ack::Ok {
                tracing::debug!(
                    "Error reading debug port CTRL register: {:?}. This should never fail!",
                    response.last_transfer_response.ack
                );
            }

            match response.last_transfer_response.ack {
                Ack::Ok => {
                    Ok(Ctrl(response.transfers[0].data.expect(
                        "CMSIS-DAP probe should always return data for a read.",
                    )))
                }
                Ack::Wait => Err(DapError::WaitResponse.into()),
                Ack::Fault => Err(DapError::FaultResponse.into()),
                Ack::NoAck => Err(DapError::NoAcknowledge.into()),
            }
        }
    }

    fn write_abort(&mut self, abort: Abort) -> Result<(), ArmError> {
        let response = commands::send_command(
            &mut self.device,
            TransferRequest::write(PortType::DebugPort, Abort::ADDRESS, abort.into()),
        )
        .map_err(CmsisDapError::from)
        .map_err(DebugProbeError::from)?;

        // We can assume that the single transfer is always executed,
        // no need to check here.

        if response.last_transfer_response.protocol_error {
            // TODO: What does this protocol error mean exactly?
            //       Should be verified in CMSIS-DAP spec
            Err(DapError::SwdProtocol.into())
        } else {
            match response.last_transfer_response.ack {
                Ack::Ok => Ok(()),
                Ack::Wait => Err(DapError::WaitResponse.into()),
                Ack::Fault => Err(DapError::FaultResponse.into()),
                Ack::NoAck => Err(DapError::NoAcknowledge.into()),
            }
        }
    }

    /// Immediately send whatever is in our batch if it is not empty.
    ///
    /// If the last transfer was a read, result is Some with the read value.
    /// Otherwise, the result is None.
    ///
    /// This will ensure any pending writes are processed and errors from them
    /// raised if necessary.
    #[tracing::instrument(skip(self))]
    fn process_batch(&mut self) -> Result<Option<u32>, ArmError> {
        let mut batch = std::mem::take(&mut self.batch);
        if batch.is_empty() {
            return Ok(None);
        }

        tracing::debug!("{} items in batch", batch.len());

        for retry in (0..5).rev() {
            tracing::debug!("Attempting batch of {} items", batch.len());
            if batch.is_empty() {
                break;
            }

            let mut transfers = TransferRequest::empty();
            for command in batch.iter().copied() {
                match command {
                    BatchCommand::Read(port, register) => {
                        transfers.add_read(port, register as u8);
                    }
                    BatchCommand::Write(port, register, value) => {
                        transfers.add_write(port, register as u8, value);
                    }
                }
            }

            let response = commands::send_command(&mut self.device, transfers)
                .map_err(CmsisDapError::from)
                .map_err(DebugProbeError::from)?;

            let count = response.transfers.len();

            tracing::debug!("{} of batch of {} items executed", count, batch.len());

            if response.last_transfer_response.protocol_error {
                if count > 0 {
                    tracing::debug!("Protocol error in response to command {}", batch[count - 1]);
                }

                return Err(DapError::SwdProtocol.into());
            }

            match response.last_transfer_response.ack {
                Ack::Ok => {
                    tracing::trace!("Transfer status: ACK");
                    return Ok(response.transfers[count - 1].data);
                }
                Ack::NoAck => {
                    tracing::debug!(
                        "Transfer status for batch item {}/{}: NACK",
                        count,
                        batch.len()
                    );
                    // TODO: Try a reset?
                    return Err(DapError::NoAcknowledge.into());
                }
                Ack::Fault => {
                    tracing::debug!(
                        "Transfer status for batch item {}/{}: FAULT",
                        count,
                        batch.len()
                    );

                    // To avoid a potential endless recursion,
                    // call a separate function to read the ctrl register,
                    // which doesn't use the batch API.
                    let ctrl = self.read_ctrl_register()?;

                    tracing::trace!("Ctrl/Stat register value is: {:?}", ctrl);

                    if ctrl.sticky_err() {
                        // Clear sticky error flags.
                        self.write_abort({
                            let mut abort = Abort(0);
                            abort.set_stkerrclr(ctrl.sticky_err());
                            abort
                        })?;
                    }

                    let successful = count.saturating_sub(1);
                    tracing::trace!("draining {:?} and retries left {:?}", successful, retry);
                    batch.drain(0..successful);
                }
                Ack::Wait => {
                    tracing::debug!(
                        "Transfer status for batch item {}/{}: WAIT",
                        count,
                        batch.len()
                    );

                    self.write_abort({
                        let mut abort = Abort(0);
                        abort.set_dapabort(true);
                        abort
                    })?;

                    return Err(DapError::WaitResponse.into());
                }
            }
        }

        Err(DapError::FaultResponse.into())
    }

    /// Add a BatchCommand to our current batch.
    ///
    /// If the BatchCommand is a Read, this will immediately process the batch
    /// and return the read value. If the BatchCommand is a write, the write is
    /// executed immediately if the batch is full, otherwise it is queued for
    /// later execution.
    fn batch_add(&mut self, command: BatchCommand) -> Result<Option<u32>, ArmError> {
        tracing::debug!("Adding command to batch: {}", command);

        self.batch.push(command);

        // We always immediately process any reads, which means there will never
        // be more than one read in a batch. We also process whenever the batch
        // is as long as can fit in one packet.
        let max_writes = (self.packet_size as usize - 3) / (1 + 4);
        match command {
            BatchCommand::Read(_, _) => self.process_batch(),
            _ if self.batch.len() == max_writes => self.process_batch(),
            _ => Ok(None),
        }
    }

    /// Set SWO port to use requested transport.
    ///
    /// Check the probe capabilities to determine which transports are available.
    fn set_swo_transport(
        &mut self,
        transport: swo::TransportRequest,
    ) -> Result<(), DebugProbeError> {
        let response = commands::send_command(&mut self.device, transport)?;
        match response {
            swo::TransportResponse(Status::DAPOk) => Ok(()),
            swo::TransportResponse(Status::DAPError) => Err(CmsisDapError::ErrorResponse.into()),
        }
    }

    /// Set SWO port to specified mode.
    ///
    /// Check the probe capabilities to determine which modes are available.
    fn set_swo_mode(&mut self, mode: swo::ModeRequest) -> Result<(), DebugProbeError> {
        let response = commands::send_command(&mut self.device, mode)?;
        match response {
            swo::ModeResponse(Status::DAPOk) => Ok(()),
            swo::ModeResponse(Status::DAPError) => Err(CmsisDapError::ErrorResponse.into()),
        }
    }

    /// Set SWO port to specified baud rate.
    ///
    /// Returns `SwoBaudrateNotConfigured` if the probe returns 0,
    /// indicating the requested baud rate was not configured,
    /// and returns the configured baud rate on success (which
    /// may differ from the requested baud rate).
    fn set_swo_baudrate(&mut self, baud: swo::BaudrateRequest) -> Result<u32, DebugProbeError> {
        let response = commands::send_command(&mut self.device, baud)?;
        tracing::debug!("Requested baud {}, got {}", baud.0, response);
        if response == 0 {
            Err(CmsisDapError::SwoBaudrateNotConfigured.into())
        } else {
            Ok(response)
        }
    }

    /// Start SWO trace data capture.
    fn start_swo_capture(&mut self) -> Result<(), DebugProbeError> {
        let response = commands::send_command(&mut self.device, swo::ControlRequest::Start)?;
        match response {
            swo::ControlResponse(Status::DAPOk) => Ok(()),
            swo::ControlResponse(Status::DAPError) => Err(CmsisDapError::ErrorResponse.into()),
        }
    }

    /// Stop SWO trace data capture.
    fn stop_swo_capture(&mut self) -> Result<(), DebugProbeError> {
        let response = commands::send_command(&mut self.device, swo::ControlRequest::Stop)?;
        match response {
            swo::ControlResponse(Status::DAPOk) => Ok(()),
            swo::ControlResponse(Status::DAPError) => Err(CmsisDapError::ErrorResponse.into()),
        }
    }

    /// Fetch current SWO trace status.
    #[allow(dead_code)]
    fn get_swo_status(&mut self) -> Result<swo::StatusResponse, DebugProbeError> {
        Ok(commands::send_command(
            &mut self.device,
            swo::StatusRequest,
        )?)
    }

    /// Fetch extended SWO trace status.
    ///
    /// request.request_status: request trace status
    /// request.request_count: request remaining bytes in trace buffer
    /// request.request_index: request sequence number and timestamp of next trace sequence
    #[allow(dead_code)]
    fn get_swo_extended_status(
        &mut self,
        request: swo::ExtendedStatusRequest,
    ) -> Result<swo::ExtendedStatusResponse, DebugProbeError> {
        Ok(commands::send_command(&mut self.device, request)?)
    }

    /// Fetch latest SWO trace data by sending a DAP_SWO_Data request.
    fn get_swo_data(&mut self) -> Result<Vec<u8>, DebugProbeError> {
        match self.swo_buffer_size {
            Some(swo_buffer_size) => {
                // We'll request the smaller of the probe's SWO buffer and
                // its maximum packet size. If the probe has less data to
                // send it will respond with as much as it can.
                let n = usize::min(swo_buffer_size, self.packet_size as usize) as u16;

                let response: swo::DataResponse =
                    commands::send_command(&mut self.device, swo::DataRequest { max_count: n })?;
                if response.status.error {
                    Err(CmsisDapError::SwoTraceStreamError.into())
                } else {
                    Ok(response.data)
                }
            }
            None => Ok(Vec::new()),
        }
    }

    fn connect_if_needed(&mut self) -> Result<(), DebugProbeError> {
        if self.connected {
            return Ok(());
        }

        let protocol = if let Some(protocol) = self.protocol {
            match protocol {
                WireProtocol::Swd => ConnectRequest::Swd,
                WireProtocol::Jtag => ConnectRequest::Jtag,
            }
        } else {
            ConnectRequest::DefaultPort
        };

        let used_protocol = commands::send_command(&mut self.device, protocol)
            .map_err(CmsisDapError::from)
            .and_then(|v| match v {
                ConnectResponse::SuccessfulInitForSWD => Ok(WireProtocol::Swd),
                ConnectResponse::SuccessfulInitForJTAG => Ok(WireProtocol::Jtag),
                ConnectResponse::InitFailed => Err(CmsisDapError::ErrorResponse),
            })?;

        // Store the actually used protocol, to handle cases where the default protocol is used.
        tracing::info!("Using protocol {}", used_protocol);
        self.protocol = Some(used_protocol);
        self.connected = true;

        Ok(())
    }
}

impl DebugProbe for CmsisDap {
    fn get_name(&self) -> &str {
        "CMSIS-DAP"
    }

    /// Get the currently set maximum speed.
    ///
    /// CMSIS-DAP offers no possibility to get the actual speed used.
    fn speed_khz(&self) -> u32 {
        self.speed_khz
    }

    /// For CMSIS-DAP, we can set the maximum speed. The actual speed
    /// used by the probe cannot be determined, but it will not be
    /// higher than this value.
    fn set_speed(&mut self, speed_khz: u32) -> Result<u32, DebugProbeError> {
        self.set_swj_clock(speed_khz * 1_000)?;
        self.speed_khz = speed_khz;

        Ok(speed_khz)
    }

    fn set_scan_chain(&mut self, scan_chain: Vec<ScanChainElement>) -> Result<(), DebugProbeError> {
        tracing::info!("Setting scan chain to {:?}", scan_chain);
        self.scan_chain = Some(scan_chain);
        Ok(())
    }

    /// Returns the JTAG scan chain
    fn scan_chain(&self) -> Result<&[ScanChainElement], DebugProbeError> {
        match self.active_protocol() {
            Some(WireProtocol::Jtag) => {
                if let Some(ref chain) = self.scan_chain {
                    Ok(chain.as_slice())
                } else {
                    Ok(&[])
                }
            }
            _ => Err(DebugProbeError::InterfaceNotAvailable {
                interface_name: "JTAG",
            }),
        }
    }

    /// Enters debug mode.
    #[tracing::instrument(skip(self))]
    fn attach(&mut self) -> Result<(), DebugProbeError> {
        tracing::debug!("Attaching to target system (clock = {}kHz)", self.speed_khz);

        // Run connect sequence (may already be done earlier via swj operations)
        self.connect_if_needed()?;

        // Set speed after connecting as it can be reset during protocol selection
        self.set_speed(self.speed_khz)?;

        self.transfer_configure(ConfigureRequest {
            idle_cycles: 0,
            wait_retry: 0xffff,
            match_retry: 0,
        })?;

        if self.active_protocol() == Some(WireProtocol::Jtag) {
            // no-op: we configure JTAG in debug_port_setup,
            // because that is where we execute the SWJ-DP Switch Sequence
            // to ensure the debug port is ready for JTAG signals,
            // at which point we can interrogate the scan chain
            // and configure the probe with the given IR lengths.
            {
                // For arm MCU there is no need to do anything, but for riscv it is necessary. Or one day in architecture\riscv code to do the relevant processing, here you can delete.
                self.scan_chain()?;
                self.select_target(0)?;
            }
        } else {
            self.configure_swd(swd::configure::ConfigureRequest {})?;
        }

        // Tell the probe we are connected so it can turn on an LED.
        let _: Result<HostStatusResponse, _> =
            commands::send_command(&mut self.device, HostStatusRequest::connected(true));

        Ok(())
    }

    /// Leave debug mode.
    fn detach(&mut self) -> Result<(), crate::Error> {
        self.process_batch()?;

        if self.swo_active {
            self.disable_swo()?;
        }

        let response = commands::send_command(&mut self.device, DisconnectRequest {})
            .map_err(|e| DebugProbeError::ProbeSpecific(Box::new(e)))?;

        // Tell probe we are disconnected so it can turn off its LED.
        let _: Result<HostStatusResponse, _> =
            commands::send_command(&mut self.device, HostStatusRequest::connected(false));

        self.connected = false;

        match response {
            DisconnectResponse(Status::DAPOk) => Ok(()),
            DisconnectResponse(Status::DAPError) => {
                Err(crate::Error::Probe(CmsisDapError::ErrorResponse.into()))
            }
        }
    }

    fn select_protocol(&mut self, protocol: WireProtocol) -> Result<(), DebugProbeError> {
        match protocol {
            WireProtocol::Jtag if self.capabilities._jtag_implemented => {
                self.protocol = Some(WireProtocol::Jtag);
                Ok(())
            }
            WireProtocol::Swd if self.capabilities._swd_implemented => {
                self.protocol = Some(WireProtocol::Swd);
                Ok(())
            }
            _ => Err(DebugProbeError::UnsupportedProtocol(protocol)),
        }
    }

    fn active_protocol(&self) -> Option<WireProtocol> {
        self.protocol
    }

    /// Asserts the nRESET pin.
    fn target_reset(&mut self) -> Result<(), DebugProbeError> {
        commands::send_command(&mut self.device, ResetRequest).map(|v: ResetResponse| {
            tracing::info!("Target reset response: {:?}", v);
        })?;
        Ok(())
    }

    fn target_reset_assert(&mut self) -> Result<(), DebugProbeError> {
        let request = SWJPinsRequestBuilder::new().nreset(false).build();

        commands::send_command(&mut self.device, request).map(|v: SWJPinsResponse| {
            tracing::info!("Pin response: {:?}", v);
        })?;
        Ok(())
    }

    fn target_reset_deassert(&mut self) -> Result<(), DebugProbeError> {
        let request = SWJPinsRequestBuilder::new().nreset(true).build();

        commands::send_command(&mut self.device, request).map(|v: SWJPinsResponse| {
            tracing::info!("Pin response: {:?}", v);
        })?;
        Ok(())
    }

    fn get_swo_interface(&self) -> Option<&dyn SwoAccess> {
        Some(self as _)
    }

    fn get_swo_interface_mut(&mut self) -> Option<&mut dyn SwoAccess> {
        Some(self as _)
    }

    fn try_get_arm_interface<'probe>(
        self: Box<Self>,
    ) -> Result<Box<dyn UninitializedArmProbe + 'probe>, (Box<dyn DebugProbe>, DebugProbeError)>
    {
        Ok(Box::new(ArmCommunicationInterface::new(self, false)))
    }

    fn has_arm_interface(&self) -> bool {
        true
    }

    fn try_get_riscv_interface_builder<'probe>(
        &'probe mut self,
    ) -> Result<Box<dyn RiscvInterfaceBuilder<'probe> + 'probe>, DebugProbeError> {
        if self.has_riscv_interface() {
            Ok(Box::new(JtagDtmBuilder::new(self)))
        } else {
            Err(DebugProbeError::InterfaceNotAvailable {
                interface_name: "RISC-V",
            })
        }
    }

    fn has_riscv_interface(&self) -> bool {
        self.capabilities._jtag_implemented
    }

    fn try_get_xtensa_interface<'probe>(
        &'probe mut self,
        state: &'probe mut XtensaDebugInterfaceState,
    ) -> Result<XtensaCommunicationInterface<'probe>, DebugProbeError> {
        if self.has_xtensa_interface() {
            Ok(XtensaCommunicationInterface::new(self, state))
        } else {
            Err(DebugProbeError::InterfaceNotAvailable {
                interface_name: "RISC-V",
            })
        }
    }

    fn has_xtensa_interface(&self) -> bool {
        self.capabilities._jtag_implemented
    }

    fn into_probe(self: Box<Self>) -> Box<dyn DebugProbe> {
        self
    }

    fn try_as_dap_probe(&mut self) -> Option<&mut dyn DapProbe> {
        Some(self)
    }
}

impl RawDapAccess for CmsisDap {
    fn core_status_notification(&mut self, status: CoreStatus) -> Result<(), DebugProbeError> {
        let running = status.is_running();
        commands::send_command(&mut self.device, HostStatusRequest::running(running))?;
        Ok(())
    }

    /// Reads the DAP register on the specified port and address.
    fn raw_read_register(&mut self, port: PortType, addr: u8) -> Result<u32, ArmError> {
        let res = self.batch_add(BatchCommand::Read(port, addr as u16))?;

        // NOTE(unwrap): batch_add will always return Some if the last command is a read
        // and running the batch was successful.
        Ok(res.unwrap())
    }

    /// Writes a value to the DAP register on the specified port and address.
    fn raw_write_register(&mut self, port: PortType, addr: u8, value: u32) -> Result<(), ArmError> {
        self.batch_add(BatchCommand::Write(port, addr as u16, value))
            .map(|_| ())
    }

    fn raw_write_block(
        &mut self,
        port: PortType,
        register_address: u8,
        values: &[u32],
    ) -> Result<(), ArmError> {
        self.process_batch()?;

        // the overhead for a single packet is 6 bytes
        //
        // [0]: HID overhead
        // [1]: Category
        // [2]: DAP Index
        // [3]: Len 1
        // [4]: Len 2
        // [5]: Request type
        //

        let max_packet_size_words = (self.packet_size - 6) / 4;

        let data_chunk_len = max_packet_size_words as usize;

        for (i, chunk) in values.chunks(data_chunk_len).enumerate() {
            let request =
                TransferBlockRequest::write_request(register_address, port, Vec::from(chunk));

            tracing::debug!("Transfer block: chunk={}, len={} bytes", i, chunk.len() * 4);

            let resp: TransferBlockResponse =
                commands::send_command(&mut self.device, request).map_err(DebugProbeError::from)?;

            if resp.transfer_response != 1 {
                return Err(DebugProbeError::from(CmsisDapError::ErrorResponse).into());
            }
        }

        Ok(())
    }

    fn raw_read_block(
        &mut self,
        port: PortType,
        register_address: u8,
        values: &mut [u32],
    ) -> Result<(), ArmError> {
        self.process_batch()?;

        // the overhead for a single packet is 6 bytes
        //
        // [0]: HID overhead
        // [1]: Category
        // [2]: DAP Index
        // [3]: Len 1
        // [4]: Len 2
        // [5]: Request type
        //

        let max_packet_size_words = (self.packet_size - 6) / 4;

        let data_chunk_len = max_packet_size_words as usize;

        for (i, chunk) in values.chunks_mut(data_chunk_len).enumerate() {
            let request =
                TransferBlockRequest::read_request(register_address, port, chunk.len() as u16);

            tracing::debug!("Transfer block: chunk={}, len={} bytes", i, chunk.len() * 4);

            let resp: TransferBlockResponse =
                commands::send_command(&mut self.device, request).map_err(DebugProbeError::from)?;

            if resp.transfer_response != 1 {
                return Err(DebugProbeError::from(CmsisDapError::ErrorResponse).into());
            }

            chunk.clone_from_slice(&resp.transfer_data[..]);
        }

        Ok(())
    }

    fn raw_flush(&mut self) -> Result<(), ArmError> {
        self.process_batch()?;
        Ok(())
    }

    fn into_probe(self: Box<Self>) -> Box<dyn DebugProbe> {
        self
    }

    fn configure_jtag(&mut self, skip_scan: bool) -> Result<(), DebugProbeError> {
        let ir_lengths = if skip_scan {
            self.scan_chain
                .as_ref()
                .map(|chain| chain.iter().filter_map(|s| s.ir_len).collect::<Vec<u8>>())
                .unwrap_or_default()
        } else {
            let chain = self.jtag_scan(
                self.scan_chain
                    .as_ref()
                    .map(|chain| {
                        chain
                            .iter()
                            .filter_map(|s| s.ir_len)
                            .map(|s| s as usize)
                            .collect::<Vec<usize>>()
                    })
                    .as_deref(),
            )?;
            chain.iter().map(|item| item.irlen as u8).collect()
        };
        tracing::info!("Configuring JTAG with ir lengths: {:?}", ir_lengths);
        self.send_jtag_configure(JtagConfigureRequest::new(ir_lengths)?)?;

        Ok(())
    }

    fn jtag_sequence(&mut self, cycles: u8, tms: bool, tdi: u64) -> Result<(), DebugProbeError> {
        self.connect_if_needed()?;

        let tdi_bytes = tdi.to_le_bytes();
        let sequence = JtagSequence::new(cycles, false, tms, tdi_bytes)?;
        let sequences = vec![sequence];

        self.send_jtag_sequences(JtagSequenceRequest::new(sequences)?)?;

        Ok(())
    }

    fn swj_sequence(&mut self, bit_len: u8, bits: u64) -> Result<(), DebugProbeError> {
        self.connect_if_needed()?;

        let data = bits.to_le_bytes();

        if tracing::enabled!(tracing::Level::TRACE) {
            let mut seq = String::new();

            let _ = write!(&mut seq, "swj sequence:");

            for i in 0..bit_len {
                let bit = (bits >> i) & 1;

                if bit == 1 {
                    let _ = write!(&mut seq, "1");
                } else {
                    let _ = write!(&mut seq, "0");
                }
            }
            tracing::trace!("{}", seq);
        }

        self.send_swj_sequences(SequenceRequest::new(&data, bit_len)?)?;

        Ok(())
    }

    fn swj_pins(
        &mut self,
        pin_out: u32,
        pin_select: u32,
        pin_wait: u32,
    ) -> Result<u32, DebugProbeError> {
        self.connect_if_needed()?;

        let request = SWJPinsRequest::from_raw_values(pin_out as u8, pin_select as u8, pin_wait);

        let Pins(response) = commands::send_command(&mut self.device, request)?;

        Ok(response as u32)
    }
}

impl DapProbe for CmsisDap {}

impl SwoAccess for CmsisDap {
    fn enable_swo(&mut self, config: &SwoConfig) -> Result<(), ArmError> {
        let caps = self.capabilities;

        // Check requested mode is available in probe capabilities
        match config.mode() {
            SwoMode::Uart if !caps.swo_uart_implemented => {
                return Err(DebugProbeError::ProbeSpecific(
                    CmsisDapError::SwoModeNotAvailable.into(),
                )
                .into())
            }
            SwoMode::Manchester if !caps.swo_manchester_implemented => {
                return Err(DebugProbeError::ProbeSpecific(
                    CmsisDapError::SwoModeNotAvailable.into(),
                )
                .into())
            }
            _ => (),
        }

        // Stop any ongoing trace
        self.stop_swo_capture()?;

        // Set transport. If the dedicated endpoint is available and we have opened
        // the probe in V2 mode and it has an SWO endpoint, request that, otherwise
        // request the DAP_SWO_Data polling mode.
        if caps.swo_streaming_trace_implemented && self.device.swo_streaming_supported() {
            tracing::debug!("Starting SWO capture with streaming transport");
            self.set_swo_transport(swo::TransportRequest::WinUsbEndpoint)?;
            self.swo_streaming = true;
        } else {
            tracing::debug!("Starting SWO capture with polled transport");
            self.set_swo_transport(swo::TransportRequest::DataCommand)?;
            self.swo_streaming = false;
        }

        // Set mode. We've already checked that the requested mode is listed as supported.
        match config.mode() {
            SwoMode::Uart => self.set_swo_mode(swo::ModeRequest::Uart)?,
            SwoMode::Manchester => self.set_swo_mode(swo::ModeRequest::Manchester)?,
        }

        // Set baud rate.
        let baud = self.set_swo_baudrate(swo::BaudrateRequest(config.baud()))?;
        if baud != config.baud() {
            tracing::warn!(
                "Target SWO baud rate not met: requested {}, got {}",
                config.baud(),
                baud
            );
        }

        self.start_swo_capture()?;

        self.swo_active = true;
        Ok(())
    }

    fn disable_swo(&mut self) -> Result<(), ArmError> {
        tracing::debug!("Stopping SWO capture");
        self.stop_swo_capture()?;
        self.swo_active = false;
        Ok(())
    }

    fn read_swo_timeout(&mut self, timeout: Duration) -> Result<Vec<u8>, ArmError> {
        if self.swo_active {
            if self.swo_streaming {
                let buffer = self
                    .device
                    .read_swo_stream(timeout)
                    .map_err(DebugProbeError::from)?;
                tracing::trace!("SWO streaming buffer: {:?}", buffer);
                Ok(buffer)
            } else {
                let data = self.get_swo_data()?;
                tracing::trace!("SWO polled data: {:?}", data);
                Ok(data)
            }
        } else {
            Ok(Vec::new())
        }
    }

    fn swo_poll_interval_hint(&mut self, config: &SwoConfig) -> Option<std::time::Duration> {
        let caps = self.capabilities;
        if caps.swo_streaming_trace_implemented && self.device.swo_streaming_supported() {
            // Streaming reads block waiting for new data so any polling interval is fine
            Some(std::time::Duration::from_secs(0))
        } else {
            match self.swo_buffer_size {
                // Given the buffer size and SWO baud rate we can estimate a poll rate.
                Some(buf_size) => poll_interval_from_buf_size(config, buf_size),

                // If we don't know the buffer size, we can't give a meaningful hint.
                None => None,
            }
        }
    }

    fn swo_buffer_size(&mut self) -> Option<usize> {
        self.swo_buffer_size
    }
}

impl RawJtagIo for CmsisDap {
    fn state(&self) -> &JtagDriverState {
        &self.jtag_driver_state
    }
    fn state_mut(&mut self) -> &mut JtagDriverState {
        &mut self.jtag_driver_state
    }
    fn shift_bit(&mut self, tms: bool, tdi: bool, capture: bool) -> Result<(), DebugProbeError> {
        shift_bit_with(
            tms,
            tdi,
            capture,
            &mut self.jtag_sequences,
            &mut self.jtag_driver_state,
        );
        Ok(())
    }

    fn shift_bits(
        &mut self,
        tms: impl IntoIterator<Item = bool>,
        tdi: impl IntoIterator<Item = bool>,
        cap: impl IntoIterator<Item = bool>,
    ) -> Result<(), DebugProbeError> {
        encode_shift_bits_with(
            tms,
            tdi,
            cap,
            &mut self.jtag_sequences,
            &mut self.jtag_driver_state,
        );
        Ok(())
    }
    fn read_captured_bits(&mut self) -> Result<BitVec<u8, Lsb0>, DebugProbeError> {
        // The queue leaves the driver up front: whatever happens on the
        // wire, no stale sequence survives into the next transfer.
        let mut sequences = std::mem::take(&mut self.jtag_sequences);
        let mut transport = DeviceTransport {
            device: &mut self.device,
            packet_size: self.packet_size,
        };
        read_captured_bits_pipelined(
            &mut sequences,
            self.packet_size,
            JTAG_SEQUENCES_PER_COMMAND,
            &mut transport,
        )
    }
}

/// Feeds encoded sequence batches straight to the device queues. Bulk
/// devices keep several packets in flight, so the flush loop can
/// overlap USB round trips with device execution.
struct DeviceTransport<'a> {
    device: &'a mut CmsisDapDevice,
    packet_size: u16,
}

impl SequenceTransport for DeviceTransport<'_> {
    fn depth(&self) -> usize {
        self.device.pipeline_depth()
    }

    fn submit_batch(&mut self, batch: Vec<JtagSequence>) -> Result<(), DebugProbeError> {
        use commands::Request;

        let request = JtagSequenceRequest::new(batch).map_err(CmsisDapError::from)?;
        let mut buffer = vec![0u8; self.packet_size as usize + 1];
        buffer[1] = <JtagSequenceRequest as Request>::COMMAND_ID as u8;
        let size = request
            .to_bytes(&mut buffer[2..])
            .map_err(|e| CmsisDapError::Send {
                command_id: <JtagSequenceRequest as Request>::COMMAND_ID,
                source: e,
            })?
            + 2;
        buffer.truncate(size);

        self.device
            .submit_buffer(&buffer, self.packet_size as usize)
            .map_err(|e| map_send_error(e))?;
        Ok(())
    }

    fn collect_batch(&mut self) -> Result<Vec<u8>, DebugProbeError> {
        use commands::{Request, SendError, Status};

        let response = self
            .device
            .collect_response(self.packet_size as usize, commands::USB_TIMEOUT)
            .map_err(map_send_error)?;

        if response.first() != Some(&(<JtagSequenceRequest as Request>::COMMAND_ID as u8)) {
            return Err(DebugProbeError::ProbeSpecific(Box::new(CmsisDapError::Send {
                command_id: <JtagSequenceRequest as Request>::COMMAND_ID,
                source: SendError::CommandIdMismatch(response.first().copied().unwrap_or(0)),
            })));
        }
        let status = Status::from_byte(*response.get(1).unwrap_or(&0xFF)).map_err(map_send_error)?;
        match status {
            Status::DAPOk => Ok(response[2..].to_vec()),
            Status::DAPError => Err(DebugProbeError::ProbeSpecific(Box::new(
                CmsisDapError::ErrorResponse,
            ))),
        }
    }
}

fn map_send_error(e: commands::SendError) -> DebugProbeError {
    DebugProbeError::ProbeSpecific(Box::new(CmsisDapError::Send {
        command_id: commands::CommandId::JtagSequence,
        source: e,
    }))
}
impl Drop for CmsisDap {
    fn drop(&mut self) {
        tracing::debug!("Detaching from CMSIS-DAP probe");
        // We ignore the error cases as we can't do much about it anyways.
        let _ = self.process_batch();

        // Cancel any pipelined inbound transfers and read away whatever
        // the device still holds, so the next program to open the probe
        // starts from a synchronized stream.
        self.device.drain();

        // If SWO is active, disable it before calling detach,
        // which ensures detach won't error on disabling SWO.
        if self.swo_active {
            let _ = self.disable_swo();
        }

        let _ = self.detach();
    }
}

/// Sequence cap per DAP_JTAG_SEQUENCE command. The protocol encodes the
/// count in one byte (maximum 255); the packet byte budgets below are
/// what actually bound bulk transfers, this cap only keeps any single
/// command's response wait bounded.
const JTAG_SEQUENCES_PER_COMMAND: usize = 128;

/// Wire bytes every DAP_JTAG_SEQUENCE command spends before any sequence:
/// the command id, the sequence count byte, and one byte of report
/// framing.
const JTAG_SEQUENCE_HEADER_BYTES: usize = 3;

/// Response bytes spent before any captured TDO data: the status byte.
/// (On HID v1 devices the response report adds its own framing byte,
/// which the negotiated report size already accounts for.)
const JTAG_SEQUENCE_RESPONSE_HEADER_BYTES: usize = 1;

/// Split queued sequences into DAP_JTAG_SEQUENCE commands under three
/// bounds: at most `max_per_command` sequences per command, each
/// command's wire size within the device packet (every sequence costs
/// its info byte plus its TDI payload padded to whole bytes, on top of
/// the command header), and the command's captured TDO data within the
/// response packet (every captured sequence answers with its bits
/// padded to whole bytes, on top of the status byte). Capture-dense
/// batches overflow the response side long before the request side -
/// a command full of 41-bit captures spends ~7 request bytes but ~6
/// response bytes per sequence - and an overflowed response silently
/// truncates the capture stream, so both budgets bind. A single
/// sequence too large for a whole packet is an error rather than a
/// split point.
fn plan_sequence_commands(
    sequences: &[JtagSequence],
    packet_size: u16,
    max_per_command: usize,
) -> Result<Vec<Vec<JtagSequence>>, DebugProbeError> {
    let byte_budget = packet_size as usize - JTAG_SEQUENCE_HEADER_BYTES;
    let response_budget = packet_size as usize - JTAG_SEQUENCE_RESPONSE_HEADER_BYTES;
    let mut batches: Vec<Vec<JtagSequence>> = Vec::new();
    let mut batch: Vec<JtagSequence> = Vec::new();
    let mut sequences_left = max_per_command;
    let mut bytes_left = byte_budget;
    let mut response_bytes_left = response_budget;

    for sequence in sequences {
        let cost = sequence.wire_bytes();
        let response_cost = sequence
            .captured_bits()
            .map(|bits| bits.div_ceil(8))
            .unwrap_or(0);
        if cost > byte_budget || response_cost > response_budget {
            return Err(DebugProbeError::Other(anyhow!(
                "a single JTAG sequence costs {cost} wire bytes / \
                 {response_cost} response bytes and cannot fit the \
                 {byte_budget}-byte command / {response_budget}-byte \
                 response budgets"
            )));
        }
        if !batch.is_empty()
            && (sequences_left == 0
                || bytes_left < cost
                || response_bytes_left < response_cost)
        {
            batches.push(std::mem::take(&mut batch));
            sequences_left = max_per_command;
            bytes_left = byte_budget;
            response_bytes_left = response_budget;
        }
        sequences_left -= 1;
        bytes_left -= cost;
        response_bytes_left -= response_cost;
        batch.push(*sequence);
    }
    if !batch.is_empty() {
        batches.push(batch);
    }
    Ok(batches)
}

/// Extract the captured TDO bits from one command's response. The DAP
/// pads every captured sequence to a whole number of bytes (e.g. 41 TDO
/// bits arrive as 48); keep each sequence's captured bits and skip its
/// padding, so concatenated captures stay aligned - concatenating the
/// raw bytes and truncating the total would keep interior padding and
/// misalign every capture after the first.
fn slice_sequence_captures(
    batch: &[JtagSequence],
    response_bytes: &[u8],
) -> Result<BitVec<u8, Lsb0>, DebugProbeError> {
    let mut stream = BitVec::<u8, Lsb0>::from_vec(response_bytes.to_vec());
    let mut capture = BitVec::<u8, Lsb0>::new();

    for sequence in batch {
        let padded = sequence.captured_bits_padded();
        if padded == 0 {
            continue;
        }
        let captured = sequence.captured_bits().expect("padded implies captured");
        if stream.len() < padded {
            return Err(DebugProbeError::Other(anyhow!(
                "short JTAG sequence response: need {padded} bits, have {}",
                stream.len()
            )));
        }
        capture.extend_from_bitslice(&stream[..captured]);
        stream = stream[padded..].to_bitvec();
    }
    Ok(capture)
}

/// Transport for flushing queued sequences to the probe: commands are
/// submitted without waiting and responses come back in submission
/// order. Parameterized so the planning, slicing, pipelining, and
/// queue-clearing contracts are testable without hardware.
pub(crate) trait SequenceTransport {
    /// How many commands may be in flight before the oldest response
    /// must be collected.
    fn depth(&self) -> usize;

    /// Submit one batch of sequences. The response arrives later via
    /// [`SequenceTransport::collect`].
    fn submit_batch(&mut self, batch: Vec<JtagSequence>) -> Result<(), DebugProbeError>;

    /// Collect the TDO bytes of the oldest submitted, uncollected batch.
    fn collect_batch(&mut self) -> Result<Vec<u8>, DebugProbeError>;
}

/// Flush queued sequences through a [`SequenceTransport`] and collect
/// the captured TDO bits. The queue is consumed up front - whatever
/// happens on the wire, no stale sequence survives into the next
/// transfer - and up to `depth` commands run in flight, which is what
/// lets a bulk probe overlap its USB round trips with device execution.
fn read_captured_bits_pipelined(
    queue: &mut Vec<JtagSequence>,
    packet_size: u16,
    max_per_command: usize,
    transport: &mut impl SequenceTransport,
) -> Result<BitVec<u8, Lsb0>, DebugProbeError> {
    let sequences = std::mem::take(queue);
    let batches = plan_sequence_commands(&sequences, packet_size, max_per_command)?;
    let depth = transport.depth().max(1);

    let mut responses = Vec::with_capacity(batches.len());
    let mut in_flight = 0usize;
    for batch in batches.iter() {
        if in_flight >= depth {
            responses.push(transport.collect_batch()?);
            in_flight -= 1;
        }
        transport.submit_batch(batch.clone())?;
        in_flight += 1;
    }
    while in_flight > 0 {
        responses.push(transport.collect_batch()?);
        in_flight -= 1;
    }

    let mut capture = BitVec::<u8, Lsb0>::new();
    for (batch, response_bytes) in batches.into_iter().zip(responses) {
        capture.extend_from_bitslice(&slice_sequence_captures(&batch, &response_bytes)?);
    }
    Ok(capture)
}

/// Per-clock encoding primitive: advance the tracked TAP state and merge
/// the clock into the queue tail, starting a fresh sequence when the
/// (tms, capture) pair changes or the tail is full. Extracted from the
/// `CmsisDap::shift_bit` body so the per-bit semantics are testable and
/// serve as the reference side of the batch encoder's differential
/// tests.
pub(crate) fn shift_bit_with(
    tms: bool,
    tdi: bool,
    capture: bool,
    queue: &mut Vec<JtagSequence>,
    state: &mut JtagDriverState,
) {
    state.state.update(tms);
    let merged = queue
        .last_mut()
        .map(|seq| seq.append(tms, tdi, capture).is_ok())
        .unwrap_or(false);
    if !merged {
        queue.push(
            JtagSequence::new(1, capture, tms, [u8::from(tdi), 0, 0, 0, 0, 0, 0, 0])
                .expect("a single-clock sequence is always within the 1..=64 bound"),
        );
    }
}

/// Encode a shift_bits triple into the sequence queue with per-clock
/// semantics bit-identical to feeding every clock through
/// [`shift_bit_with`]: a sequence boundary appears only where the (tms,
/// capture) pair changes or the 64-clock bound forces a chunk, and the
/// queue tail participates in merging exactly as the per-bit path would
/// (partial merges included). Unlike the per-bit loop, a same-pair
/// stretch packs its TDI bits straight into the sequence data bytes at
/// run granularity, which removes the per-clock host cost.
pub(crate) fn encode_shift_bits_with<I1, I2, I3>(
    tms: I1,
    tdi: I2,
    cap: I3,
    queue: &mut Vec<JtagSequence>,
    state: &mut JtagDriverState,
) where
    I1: IntoIterator<Item = bool>,
    I2: IntoIterator<Item = bool>,
    I3: IntoIterator<Item = bool>,
{
    let mut run: Option<(bool, bool, [u8; 8], usize)> = None;
    for ((tms, tdi), cap) in tms.into_iter().zip(tdi).zip(cap) {
        state.state.update(tms);
        match &mut run {
            Some((r_tms, r_cap, bits, len)) if *r_tms == tms && *r_cap == cap && *len < 64 => {
                bits[*len / 8] |= u8::from(tdi) << (*len % 8);
                *len += 1;
            }
            _ => {
                flush_run(run.take(), queue);
                let mut bits = [0u8; 8];
                bits[0] = u8::from(tdi);
                run = Some((tms, cap, bits, 1));
            }
        }
    }
    flush_run(run.take(), queue);
}

/// Commit one completed (tms, capture) run into the queue with per-clock
/// merge semantics: clocks first extend a same-pair tail up to its
/// remaining 64-clock capacity, and the remainder becomes a fresh
/// sequence. A committed run never exceeds 64 clocks, so at most one
/// fresh sequence is created.
fn flush_run(run: Option<(bool, bool, [u8; 8], usize)>, queue: &mut Vec<JtagSequence>) {
    let Some((tms, cap, bits, mut len)) = run else {
        return;
    };
    let mut taken = 0;
    if let Some(tail) = queue.last_mut() {
        taken = tail.append_bits(tms, cap, &bits, len);
        len -= taken;
    }
    if len > 0 {
        let mut shifted = [0u8; 8];
        for i in 0..len {
            let bit = (bits[(taken + i) / 8] >> ((taken + i) % 8)) & 1;
            shifted[i / 8] |= bit << (i % 8);
        }
        queue.push(
            JtagSequence::new(len as u8, cap, tms, shifted)
                .expect("a committed run never exceeds the 64-clock bound"),
        );
    }
}

impl From<ScanChainError> for CmsisDapError {
    fn from(error: ScanChainError) -> Self {
        match error {
            ScanChainError::InvalidIdCode => CmsisDapError::InvalidIdCode,
            ScanChainError::InvalidIR => CmsisDapError::InvalidIR,
        }
    }
}

#[cfg(test)]
mod sequence_capture_tests {
    use super::*;

    #[test]
    fn plan_sequence_commands_splits_by_byte_budget() {
        let scan = || JtagSequence::new(41, true, false, [0xFF; 8]).unwrap();
        let bit = || JtagSequence::new(1, false, false, [1, 0, 0, 0, 0, 0, 0, 0]).unwrap();

        // A 41-cycle sequence costs 7 wire bytes; under a 61-byte budget
        // (64-byte packet minus the header) eight of them fit (8*7=56)
        // and the ninth would not (63).
        let sequences: Vec<JtagSequence> = (0..12).map(|_| scan()).collect();
        let batches = plan_sequence_commands(&sequences, 64, JTAG_SEQUENCES_PER_COMMAND).unwrap();
        let sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![8, 4]);

        // Twenty one-bit sequences (2 bytes each) fit one command.
        let sequences: Vec<JtagSequence> = (0..20).map(|_| bit()).collect();
        let batches = plan_sequence_commands(&sequences, 64, JTAG_SEQUENCES_PER_COMMAND).unwrap();
        assert_eq!(batches.len(), 1);

        // Sixty one-bit sequences under a 61-byte budget split by the BYTE
        // budget (30*2=60 fits, 31*2=62 does not), not by the count cap.
        let sequences: Vec<JtagSequence> = (0..60).map(|_| bit()).collect();
        let batches = plan_sequence_commands(&sequences, 64, JTAG_SEQUENCES_PER_COMMAND).unwrap();
        let sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![30, 30]);

        // With byte room to spare (131-byte budget), the same sixty
        // sequences fit one command: neither the byte budget nor the
        // count cap binds.
        let batches = plan_sequence_commands(&sequences, 131, JTAG_SEQUENCES_PER_COMMAND).unwrap();
        let sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![60]);

        // Two hundred one-bit sequences under a 1021-byte budget split
        // by the COUNT cap instead.
        let sequences: Vec<JtagSequence> = (0..200).map(|_| bit()).collect();
        let batches = plan_sequence_commands(&sequences, 1024, JTAG_SEQUENCES_PER_COMMAND).unwrap();
        let sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![128, 72]);

        // Capture-dense batches: every 41-bit capture sequence answers
        // with 6 response bytes, so the response budget must bind too -
        // and no emitted command may exceed either budget.
        let scans: Vec<JtagSequence> = (0..200).map(|_| scan()).collect();
        let batches = plan_sequence_commands(&scans, 512, JTAG_SEQUENCES_PER_COMMAND).unwrap();
        assert!(batches.len() >= 3);
        for batch in &batches {
            let wire: usize = batch.iter().map(|s| s.wire_bytes()).sum();
            let response: usize = batch
                .iter()
                .map(|s| s.captured_bits().map(|b| b.div_ceil(8)).unwrap_or(0))
                .sum();
            assert!(wire <= 512 - JTAG_SEQUENCE_HEADER_BYTES, "wire {wire}");
            assert!(
                response <= 512 - JTAG_SEQUENCE_RESPONSE_HEADER_BYTES,
                "response {response}"
            );
        }

        // Exact fill: with a 64-byte budget, the 32nd one-bit sequence
        // (remaining 2 == cost) is admitted - an off-by-one split at
        // equality would produce [31, 29].
        let sequences: Vec<JtagSequence> = (0..60).map(|_| bit()).collect();
        let batches = plan_sequence_commands(&sequences, 67, JTAG_SEQUENCES_PER_COMMAND).unwrap();
        let sizes: Vec<usize> = batches.iter().map(Vec::len).collect();
        assert_eq!(sizes, vec![32, 28]);
    }

    #[test]
    fn single_sequence_over_budget_rejected() {
        // A 64-cycle sequence costs 9 wire bytes; a packet leaving only 8
        // cannot carry it even as a command of its own - that is an
        // error, not a wrapping split.
        let sequence = JtagSequence::new(64, true, false, [0xFF; 8]).unwrap();
        let result = plan_sequence_commands(&[sequence], 11, JTAG_SEQUENCES_PER_COMMAND);
        assert!(result.is_err());
    }

    fn padded_capture_stream(captured: &[bool], padding_bits: usize) -> Vec<u8> {
        // Model the wire format: the captured bits followed by padding to
        // a whole byte boundary, with the padding bits reading as ones.
        let mut stream = BitVec::<u8, Lsb0>::new();
        stream.extend(captured.iter().copied());
        stream.extend(std::iter::repeat(true).take(padding_bits));
        stream.into_vec()
    }

    #[test]
    fn slice_sequence_captures_strips_interior_padding() {
        let first: Vec<bool> = (0..41).map(|i| i % 3 == 0).collect();
        let second: Vec<bool> = (0..41).map(|i| i % 2 == 0).collect();
        let mut response = padded_capture_stream(&first, 7);
        response.extend(padded_capture_stream(&second, 7));

        let batch = vec![
            JtagSequence::new(41, true, false, [0x00; 8]).unwrap(),
            JtagSequence::new(41, true, false, [0x00; 8]).unwrap(),
        ];
        let capture = slice_sequence_captures(&batch, &response).unwrap();

        // Exactly the 82 captured bits survive; the second capture starts
        // at bit 41, not at the padded 48-bit boundary.
        assert_eq!(capture.len(), 82);
        let extracted: Vec<bool> = capture.iter().by_vals().collect();
        assert_eq!(&extracted[..41], &first[..]);
        assert_eq!(&extracted[41..], &second[..]);
    }

    #[test]
    fn slice_sequence_captures_rejects_short_response() {
        let batch = vec![JtagSequence::new(41, true, false, [0x00; 8]).unwrap()];
        // Five bytes carry 40 bits, one short of the 48 the padded
        // capture must occupy; truncating silently would misalign every
        // later capture, so this must be an error.
        let response = vec![0xFFu8; 5];
        assert!(slice_sequence_captures(&batch, &response).is_err());
    }

    #[test]
    fn read_captured_bits_clears_queue_on_send_error() {
        // A failing transport must propagate the error AND leave no stale
        // sequence behind: the queue is drained up front, so the next
        // transfer starts from an empty bitstream.
        let mut queue: Vec<JtagSequence> = (0..4)
            .map(|_| JtagSequence::new(41, true, false, [0x00; 8]).unwrap())
            .collect();
        let mut transport = MockTransport::error("usb write failed");
        let result = read_captured_bits_pipelined(
            &mut queue,
            64,
            JTAG_SEQUENCES_PER_COMMAND,
            &mut transport,
        );
        assert!(result.is_err());
        assert!(queue.is_empty());
    }

    #[test]
    fn read_captured_bits_clears_queue_on_short_response() {
        // A response too short for its padded capture fails the flush; the
        // queue must still be empty afterwards.
        let mut queue: Vec<JtagSequence> =
            vec![JtagSequence::new(41, true, false, [0x00; 8]).unwrap()];
        let mut transport = MockTransport::short_response();
        let result = read_captured_bits_pipelined(
            &mut queue,
            64,
            JTAG_SEQUENCES_PER_COMMAND,
            &mut transport,
        );
        assert!(result.is_err());
        assert!(queue.is_empty());
    }

    #[test]
    fn read_captured_bits_slices_padding_end_to_end() {
        // Two captured 41-bit scans queued as one command; the transport
        // returns the wire-format response with interior padding, and the
        // collected capture contains both scans back to back.
        let first: Vec<bool> = (0..41).map(|i| i % 3 == 0).collect();
        let second: Vec<bool> = (0..41).map(|i| i % 2 == 0).collect();
        let mut response = padded_capture_stream(&first, 7);
        response.extend(padded_capture_stream(&second, 7));

        let mut queue: Vec<JtagSequence> = (0..2)
            .map(|_| JtagSequence::new(41, true, false, [0x00; 8]).unwrap())
            .collect();
        let mut transport = MockTransport::responses(vec![response]);
        let capture = read_captured_bits_pipelined(
            &mut queue,
            64,
            JTAG_SEQUENCES_PER_COMMAND,
            &mut transport,
        )
        .unwrap();
        assert_eq!(capture.len(), 82);
        let extracted: Vec<bool> = capture.iter().by_vals().collect();
        assert_eq!(&extracted[..41], &first[..]);
        assert_eq!(&extracted[41..], &second[..]);
    }

    #[test]
    fn pipelined_flush_respects_the_depth_window() {
        // With more batches than the transport keeps in flight, submits
        // and collects interleave: the in-flight count never exceeds the
        // depth, every batch is submitted exactly once, and responses
        // still pair with their batches in order.
        let first: Vec<bool> = (0..41).map(|i| i % 5 == 0).collect();
        let second: Vec<bool> = (0..41).map(|i| i % 3 == 0).collect();
        let responses = vec![
            padded_capture_stream(&first, 7),
            padded_capture_stream(&second, 7),
        ];

        // One sequence per batch (a 4-byte packet budget splits them).
        let mut queue: Vec<JtagSequence> = (0..2)
            .map(|_| JtagSequence::new(41, true, false, [0x00; 8]).unwrap())
            .collect();
        let mut transport = MockTransport::responses(responses.clone());
        transport.depth = 1;
        let capture = read_captured_bits_pipelined(&mut queue, 11, 128, &mut transport).unwrap();
        assert!(transport.max_in_flight <= 1, "depth window exceeded");
        let extracted: Vec<bool> = capture.iter().by_vals().collect();
        assert_eq!(&extracted[..41], &first[..]);
        assert_eq!(&extracted[41..], &second[..]);
    }

    /// Hands back canned responses (or errors) in order and records the
    /// in-flight watermark, so flush tests can assert on pipelining
    /// behavior without hardware.
    struct MockTransport {
        canned: std::collections::VecDeque<Result<Vec<u8>, DebugProbeError>>,
        depth: usize,
        in_flight: usize,
        max_in_flight: usize,
        submitted: usize,
    }

    impl MockTransport {
        fn responses(list: Vec<Vec<u8>>) -> Self {
            Self {
                canned: list.into_iter().map(Ok).collect(),
                depth: 8,
                in_flight: 0,
                max_in_flight: 0,
                submitted: 0,
            }
        }

        fn error(message: &'static str) -> Self {
            Self {
                canned: [Err(DebugProbeError::Other(anyhow!(message)))]
                    .into_iter()
                    .collect(),
                depth: 8,
                in_flight: 0,
                max_in_flight: 0,
                submitted: 0,
            }
        }

        fn short_response() -> Self {
            Self::responses(vec![vec![0xFF; 5]])
        }
    }

    impl SequenceTransport for MockTransport {
        fn depth(&self) -> usize {
            self.depth
        }

        fn submit_batch(&mut self, _batch: Vec<JtagSequence>) -> Result<(), DebugProbeError> {
            self.submitted += 1;
            self.in_flight += 1;
            self.max_in_flight = self.max_in_flight.max(self.in_flight);
            Ok(())
        }

        fn collect_batch(&mut self) -> Result<Vec<u8>, DebugProbeError> {
            self.in_flight -= 1;
            self.canned
                .pop_front()
                .unwrap_or_else(|| Err(DebugProbeError::Other(anyhow!("no response left"))))
        }
    }
}

#[cfg(test)]
mod batch_encode_tests {
    use super::*;

    /// The production per-bit primitive, driven clock by clock: the
    /// reference side of every differential assertion.
    fn reference(clocks: &[(bool, bool, bool)]) -> (Vec<JtagSequence>, JtagDriverState) {
        let mut queue = Vec::new();
        let mut state = JtagDriverState::default();
        for &(tms, tdi, cap) in clocks {
            shift_bit_with(tms, tdi, cap, &mut queue, &mut state);
        }
        (queue, state)
    }

    fn batch(clocks: &[(bool, bool, bool)]) -> (Vec<JtagSequence>, JtagDriverState) {
        let mut queue = Vec::new();
        let mut state = JtagDriverState::default();
        encode_shift_bits_with(
            clocks.iter().map(|c| c.0),
            clocks.iter().map(|c| c.1),
            clocks.iter().map(|c| c.2),
            &mut queue,
            &mut state,
        );
        (queue, state)
    }

    fn assert_bit_identical(name: &str, clocks: &[(bool, bool, bool)]) {
        let (rq, rs) = reference(clocks);
        let (bq, bs) = batch(clocks);
        assert_eq!(rq.len(), bq.len(), "{name}: sequence count differs");
        assert_eq!(rq, bq, "{name}: sequence vectors differ");
        assert_eq!(rs.state, bs.state, "{name}: tracked TAP state differs");
    }

    /// A data word of pseudo-random TDI bits (bit diversity matters:
    /// equal-bit runs would not catch data packing bugs).
    struct BitSource(u8);
    impl BitSource {
        fn next(&mut self) -> bool {
            self.0 = self.0.rotate_left(3) ^ 0x5A;
            self.0 & 1 == 1
        }
    }

    /// 41-bit DMI DR write scan shape: navigation into Shift-DR, 40 data
    /// clocks under TMS=0, the final data bit riding the TMS=1 exit
    /// clock (captured), Update-DR, return to idle, then idle clocks.
    fn dmi_write_scan(idle: usize, seed: u8) -> Vec<(bool, bool, bool)> {
        let mut bits = BitSource(seed);
        let mut clocks = vec![
            (true, false, false), // navigate toward Shift-DR
            (false, false, false),
            (false, false, false),
        ];
        for _ in 0..40 {
            clocks.push((false, bits.next(), true));
        }
        clocks.push((true, bits.next(), true)); // exit clock carries the last data bit
        clocks.push((true, false, false)); // Update-DR
        clocks.push((false, false, false)); // back to idle
        for _ in 0..idle {
            clocks.push((false, false, false));
        }
        clocks
    }

    #[test]
    fn dmi_scan_shape_idle0_and_idle3() {
        assert_bit_identical("dmi idle=0", &dmi_write_scan(0, 0x1F));
        assert_bit_identical("dmi idle=3", &dmi_write_scan(3, 0xA7));
    }

    #[test]
    fn multi_tap_bypass_capture_toggles_inside_a_run() {
        // drpre/drpost bypass clocks switch capture false->true->false
        // inside a single constant-TMS run: the (tms, capture) pair, not
        // TMS alone, must gate sequence boundaries.
        let mut bits = BitSource(0x33);
        let mut clocks: Vec<(bool, bool, bool)> = (0..3).map(|_| (false, false, false)).collect();
        for _ in 0..41 {
            clocks.push((false, bits.next(), true));
        }
        clocks.extend((0..2).map(|_| (false, false, false)));
        assert_bit_identical("multi-tap bypass", &clocks);
    }

    #[test]
    fn runs_longer_than_64_chunk() {
        let mut bits = BitSource(0x71);
        let mut clocks: Vec<(bool, bool, bool)> =
            (0..260).map(|_| (false, bits.next(), true)).collect();
        clocks.extend((0..130).map(|_| (false, false, false)));
        assert_bit_identical("long runs", &clocks);
    }

    #[test]
    fn cross_call_merge_at_the_64_boundary() {
        // A (tms=0, capture=false) tail of 3 clocks meeting a 100-clock
        // same-pair run: the per-bit reference merges 61 clocks into the
        // tail and starts a 39-clock sequence; the batch encoder must do
        // exactly the same, not an all-or-nothing merge.
        for tail in 1..63 {
            let mut rq = Vec::new();
            let mut rs = JtagDriverState::default();
            for _ in 0..tail {
                shift_bit_with(false, false, false, &mut rq, &mut rs);
            }
            let mut bq = rq.clone();
            let mut bs = JtagDriverState::default();
            bs.state = rs.state;
            let run: Vec<(bool, bool, bool)> =
                (0..100).map(|i| (false, i % 3 == 0, false)).collect();
            for &(tms, tdi, cap) in &run {
                shift_bit_with(tms, tdi, cap, &mut rq, &mut rs);
            }
            encode_shift_bits_with(
                run.iter().map(|c| c.0),
                run.iter().map(|c| c.1),
                run.iter().map(|c| c.2),
                &mut bq,
                &mut bs,
            );
            assert_eq!(rq, bq, "tail {tail}: sequences differ");
            assert_eq!(rs.state, bs.state, "tail {tail}: state diverges");
        }
    }

    #[test]
    fn infinite_iterators_follow_zip_shortest() {
        // tms finite, tdi/cap infinite: the tms iterator is the length
        // authority, exactly like the per-bit default implementation.
        let tms = [true, false, false, false, false, false];
        let clocks: Vec<(bool, bool, bool)> = tms.iter().map(|&t| (t, false, false)).collect();
        let (rq, rs) = reference(&clocks);

        let mut bq = Vec::new();
        let mut bs = JtagDriverState::default();
        encode_shift_bits_with(
            tms,
            std::iter::repeat(false),
            std::iter::repeat(false),
            &mut bq,
            &mut bs,
        );
        assert_eq!(rq, bq);
        assert_eq!(rs.state, bs.state);
    }

    #[test]
    fn capture_toggles_produce_many_small_runs() {
        let mut clocks = Vec::new();
        for i in 0..90 {
            clocks.push((false, i % 5 == 0, i % 3 == 0));
        }
        assert_bit_identical("capture toggles", &clocks);
    }

    #[test]
    fn prefix_state_tracking_matches_reference() {
        let clocks = dmi_write_scan(2, 0x9D);
        for k in [1usize, 7, 45, 48] {
            let (_, rs) = reference(&clocks[..k]);
            let (_, bs) = batch(&clocks[..k]);
            assert_eq!(rs.state, bs.state, "prefix k={k}: state diverges");
        }
    }
}
