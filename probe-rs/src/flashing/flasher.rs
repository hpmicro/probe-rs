use probe_rs_target::{MemoryRegion, RawFlashAlgorithm};
use tracing::Level;

use super::{FlashAlgorithm, FlashBuilder, FlashError, FlashFill, FlashPage, FlashProgress};
use crate::config::NvmRegion;
use crate::flashing::encoder::FlashEncoder;
use crate::flashing::FlashLayout;
use crate::memory::MemoryInterface;
use crate::{core::CoreRegisters, session::Session, Core, InstructionSet};
use std::{
    fmt::Debug,
    time::{Duration, Instant},
};

pub(super) trait Operation {
    fn operation() -> u32;
    fn operation_name() -> &'static str {
        match Self::operation() {
            1 => "Erase",
            2 => "Program",
            3 => "Verify",
            _ => "Unknown Operation",
        }
    }
}

pub(super) struct Erase;

impl Operation for Erase {
    fn operation() -> u32 {
        1
    }
}

pub(super) struct Program;

impl Operation for Program {
    fn operation() -> u32 {
        2
    }
}

pub(super) struct Verify;

impl Operation for Verify {
    fn operation() -> u32 {
        3
    }
}

/// A structure to control the flash of an attached microchip.
///
/// Once constructed it can be used to program date to the flash.
pub(super) struct Flasher<'session> {
    session: &'session mut Session,
    core_index: usize,
    flash_algorithm: FlashAlgorithm,
    progress: FlashProgress,
    /// Whether the algorithm has been written to the target's RAM yet.
    algorithm_loaded: bool,
}

impl<'session> Flasher<'session> {
    pub(super) fn new(
        session: &'session mut Session,
        core_index: usize,
        raw_flash_algorithm: &RawFlashAlgorithm,
        progress: FlashProgress,
    ) -> Result<Self, FlashError> {
        let target = session.target();

        // Find a RAM region from which we can run the algo.
        let mm = &target.memory_map;
        let core_name = &target.cores[core_index].name;
        let ram = mm
            .iter()
            .filter_map(MemoryRegion::as_ram_region)
            .find(|ram| {
                // If the algorithm has a forced load address, we try to use it.
                // If not, then follow the CMSIS-Pack spec and use first available RAM region.
                // In theory, it should be the "first listed in the pack", but the process of
                // reading from the pack files obfuscates the list order, so we will use the first
                // one in the target spec, which is the qualifying region with the lowest start saddress.
                // - See https://open-cmsis-pack.github.io/Open-CMSIS-Pack-Spec/main/html/pdsc_family_pg.html#element_memory .
                if let Some(load_addr) = raw_flash_algorithm.load_address {
                    // The RAM must contain the forced load address _and_
                    // be accessible from the core we're going to run the
                    // algorithm on.
                    ram.range.contains(&load_addr) && ram.cores.contains(core_name)
                } else {
                    // Any RAM is okay as long as it's accessible to the core;
                    // the algorithm is presumably position-independent.
                    ram.cores.contains(core_name)
                }
            })
            .ok_or(FlashError::NoRamDefined {
                name: session.target().name.clone(),
            })?;
        tracing::info!("Chosen RAM to run the algo: {:x?}", ram);

        let data_ram = if let Some(data_load_address) = raw_flash_algorithm.data_load_address {
            mm.iter()
                .filter_map(MemoryRegion::as_ram_region)
                .find(|ram| {
                    // The RAM must contain the forced load address _and_
                    // be accessible from the core we're going to run the
                    // algorithm on.
                    ram.range.contains(&data_load_address) && ram.cores.contains(core_name)
                })
                .ok_or(FlashError::NoRamDefined {
                    name: session.target().name.clone(),
                })?
        } else {
            ram
        };
        tracing::info!("Data will be loaded to: {:x?}", data_ram);

        let flash_algorithm = FlashAlgorithm::assemble_from_raw_with_data(
            raw_flash_algorithm,
            ram,
            data_ram,
            target,
        )?;

        // The algorithm is loaded lazily on first active use: a flasher
        // built only to answer layout questions never touches the
        // target, so it must not pay the halt/reset/load cycle.
        Ok(Self {
            session,
            core_index,
            flash_algorithm,
            progress,
            algorithm_loaded: false,
        })
    }

    pub(super) fn flash_algorithm(&self) -> &FlashAlgorithm {
        &self.flash_algorithm
    }

    pub(super) fn double_buffering_supported(&self) -> bool {
        self.flash_algorithm.page_buffers.len() > 1
    }

    fn load(&mut self) -> Result<(), FlashError> {
        tracing::debug!("Initializing the flash algorithm.");
        let algo = &mut self.flash_algorithm;

        // Attach to memory and core.
        let mut core = self
            .session
            .core(self.core_index)
            .map_err(FlashError::Core)?;

        // TODO: Halt & reset target.
        tracing::debug!("Halting core {}", self.core_index);
        let cpu_info = core
            .halt(Duration::from_millis(100))
            .map_err(FlashError::Core)?;
        tracing::debug!("PC = {:010x}", cpu_info.pc);
        tracing::debug!("Reset and halt");
        core.reset_and_halt(Duration::from_millis(500))
            .map_err(FlashError::Core)?;

        // TODO: Possible special preparation of the target such as enabling faster clocks for the flash e.g.

        // Load flash algorithm code into target RAM.
        tracing::debug!("Downloading algorithm code to {:#010x}", algo.load_address);

        core.write_32(algo.load_address, algo.instructions.as_slice())
            .map_err(FlashError::Core)?;

        let mut data = vec![0; algo.instructions.len()];
        core.read_32(algo.load_address, &mut data)
            .map_err(FlashError::Core)?;

        for (offset, (original, read_back)) in algo.instructions.iter().zip(data.iter()).enumerate()
        {
            if original == read_back {
                continue;
            }

            tracing::error!(
                "Failed to verify flash algorithm. Data mismatch at address {:#010x}",
                algo.load_address + (4 * offset) as u64
            );
            tracing::error!("Original instruction: {:#010x}", original);
            tracing::error!("Readback instruction: {:#010x}", read_back);

            tracing::error!("Original: {:x?}", &algo.instructions);
            tracing::error!("Readback: {:x?}", &data);

            return Err(FlashError::FlashAlgorithmNotLoaded);
        }

        tracing::debug!("RAM contents match flashing algo blob.");

        Ok(())
    }

    pub(super) fn init<O: Operation>(
        &mut self,
        clock: Option<u32>,
    ) -> Result<ActiveFlasher<'_, O>, FlashError> {
        tracing::debug!("Preparing Flasher for operation {}", O::operation_name());
        if !self.algorithm_loaded {
            self.load()?;
            self.algorithm_loaded = true;
        }
        let memory_map = self.session.target().memory_map.clone();
        // Attach to memory and core.
        let core = self
            .session
            .core(self.core_index)
            .map_err(FlashError::Core)?;

        let mut flasher = ActiveFlasher::<O> {
            core,
            rtt: None,
            memory_map,
            progress: self.progress.clone(),
            flash_algorithm: self.flash_algorithm.clone(),
            sw_breakpoint_trap_set: false,
            cached_regs: [None; 8],
            _operation: core::marker::PhantomData,
        };

        flasher.init(clock)?;

        Ok(flasher)
    }

    pub(super) fn run_erase_all(&mut self) -> Result<(), FlashError> {
        self.progress.started_erasing();
        let result = if self.session.has_sequence_erase_all() {
            fn run(flasher: &mut Flasher) -> Result<(), FlashError> {
                flasher
                    .session
                    .sequence_erase_all()
                    .map_err(|e| FlashError::ChipEraseFailed {
                        source: Box::new(e),
                    })?;
                // We need to reload the flasher, since the debug sequence erase
                // may have invalidated any previously invalid state
                flasher.load()
            }

            run(self)
        } else {
            self.run_erase(|active| active.erase_all())
        };

        if result.is_ok() {
            self.progress.finished_erasing();
        } else {
            self.progress.failed_erasing();
        }

        result
    }

    pub(super) fn run_erase<T, F>(&mut self, f: F) -> Result<T, FlashError>
    where
        F: FnOnce(&mut ActiveFlasher<'_, Erase>) -> Result<T, FlashError> + Sized,
    {
        // TODO: Fix those values (None, None).
        let mut active = self.init(None)?;
        let r = f(&mut active)?;
        active.uninit()?;
        Ok(r)
    }

    pub(super) fn run_program<T, F>(&mut self, f: F) -> Result<T, FlashError>
    where
        F: FnOnce(&mut ActiveFlasher<'_, Program>) -> Result<T, FlashError> + Sized,
    {
        // TODO: Fix those values (None, None).
        let mut active = self.init(None)?;
        let r = f(&mut active)?;
        active.uninit()?;
        Ok(r)
    }

    pub(super) fn run_verify<T, F>(&mut self, f: F) -> Result<T, FlashError>
    where
        F: FnOnce(&mut ActiveFlasher<'_, Verify>) -> Result<T, FlashError> + Sized,
    {
        // TODO: Fix those values (None, None).
        let mut active = self.init(None)?;
        let r = f(&mut active)?;
        active.uninit()?;
        Ok(r)
    }

    pub(super) fn is_chip_erase_supported(&self) -> bool {
        self.session.has_sequence_erase_all() || self.flash_algorithm().pc_erase_all.is_some()
    }

    /// Program the contents of given `FlashBuilder` to the flash.
    ///
    /// If `restore_unwritten_bytes` is `true`, all bytes of a sector,
    /// that are not to be written during flashing will be read from the flash first
    /// and written again once the sector is erased.
    pub(super) fn program(
        &mut self,
        region: &NvmRegion,
        flash_builder: &FlashBuilder,
        restore_unwritten_bytes: bool,
        enable_double_buffering: bool,
        skip_erasing: bool,
    ) -> Result<(), FlashError> {
        tracing::debug!("Starting program procedure.");
        // Convert the list of flash operations into flash sectors and pages.
        let mut flash_layout = self.flash_layout(region, flash_builder, restore_unwritten_bytes)?;

        tracing::debug!("Double Buffering enabled: {:?}", enable_double_buffering);
        tracing::debug!(
            "Restoring unwritten bytes enabled: {:?}",
            restore_unwritten_bytes
        );

        // Read all fill areas from the flash.
        self.progress.started_filling();

        if restore_unwritten_bytes {
            let fills = flash_layout.fills().to_vec();
            for fill in fills {
                let t = Instant::now();
                let page = &mut flash_layout.pages_mut()[fill.page_index()];
                let result = self.fill_page(page, &fill);

                // If we encounter an error, catch it, gracefully report the failure and return the error.
                if result.is_err() {
                    self.progress.failed_filling();
                    return result;
                } else {
                    self.progress.page_filled(fill.size(), t.elapsed());
                }
            }
        }

        // We successfully finished filling.
        self.progress.finished_filling();

        let flash_encoder = FlashEncoder::new(self.flash_algorithm.transfer_encoding, flash_layout);

        // Skip erase if necessary (i.e. chip erase was done before)
        if !skip_erasing {
            // Erase all necessary sectors
            self.sector_erase(&flash_encoder)?;
        }

        // Flash all necessary pages.
        if self.double_buffering_supported() && enable_double_buffering {
            self.program_double_buffer(&flash_encoder)?;
        } else {
            self.program_simple(&flash_encoder)?;
        };

        Ok(())
    }

    /// Fills all the bytes of `current_page`.
    ///
    /// If `restore_unwritten_bytes` is `true`, all bytes of the page,
    /// that are not to be written during flashing will be read from the flash first
    /// and written again once the page is programmed.
    pub(super) fn fill_page(
        &mut self,
        page: &mut FlashPage,
        fill: &FlashFill,
    ) -> Result<(), FlashError> {
        let page_offset = (fill.address() - page.address()) as usize;
        let page_slice = &mut page.data_mut()[page_offset..page_offset + fill.size() as usize];
        self.run_verify(|active| {
            active
                .core
                .read(fill.address(), page_slice)
                .map_err(FlashError::Core)
        })
    }

    /// Programs the pages given in `flash_layout` into the flash.
    fn program_simple(&mut self, flash_encoder: &FlashEncoder) -> Result<(), FlashError> {
        self.progress
            .started_programming(flash_encoder.program_size());

        let mut t = Instant::now();
        let result = self.run_program(|active| {
            for page in flash_encoder.pages() {
                active
                    .program_page(page.address(), page.data())
                    .map_err(|error| FlashError::PageWrite {
                        page_address: page.address(),
                        source: Box::new(error),
                    })?;
                active.progress.page_programmed(page.size(), t.elapsed());

                t = Instant::now();
            }
            Ok(())
        });

        if result.is_ok() {
            self.progress.finished_programming();
        } else {
            self.progress.failed_programming();
        }

        result
    }

    /// Perform an erase of all sectors given in `flash_layout`.
    fn sector_erase(&mut self, flash_encoder: &FlashEncoder) -> Result<(), FlashError> {
        self.progress.started_erasing();

        let mut t = Instant::now();
        let result = self.run_erase(|active| {
            for sector in flash_encoder.sectors() {
                active
                    .erase_sector(sector.address())
                    .map_err(|e| FlashError::EraseFailed {
                        sector_address: sector.address(),
                        source: Box::new(e),
                    })?;
                active.progress.sector_erased(sector.size(), t.elapsed());

                t = Instant::now();
            }
            Ok(())
        });

        if result.is_ok() {
            self.progress.finished_erasing();
        } else {
            self.progress.failed_erasing();
        }

        result
    }

    /// Flash a program using double buffering.
    ///
    /// This uses two buffers to increase the flash speed.
    /// While the data from one buffer is programmed, the
    /// data for the next page is already downloaded
    /// into the next buffer.
    ///
    /// This is only possible if the RAM is large enough to
    /// fit at least two page buffers. See [Flasher::double_buffering_supported].
    fn program_double_buffer(&mut self, flash_encoder: &FlashEncoder) -> Result<(), FlashError> {
        // The routine's program entry passes its length straight to the
        // flash driver, which loops the pages internally - one large
        // call per batch replaces the per-page call cycle that
        // dominates the programming time. The buffers live in the
        // data region as one contiguous strip sized for a batch.
        const BATCH_PAGES: usize = 16;

        self.progress
            .started_programming(flash_encoder.program_size());

        let mut t = Instant::now();
        let result = self.run_program(|active| {
            let all_pages: Vec<_> = flash_encoder.pages().to_vec();
            let mut batch_no = 0usize;

            // Split the page list into runs of consecutive addresses: a
            // single program call writes one contiguous block, so a gap
            // in the image (unwritten bytes between sections) must end
            // the run - writing across it would shift the following
            // data into the gap.
            let mut runs: Vec<&[FlashPage]> = Vec::new();
            let mut run_start = 0usize;
            for i in 1..=all_pages.len() {
                let run_break = i == all_pages.len()
                    || all_pages[i].address() != all_pages[i - 1].address() + 256;
                if run_break {
                    runs.push(&all_pages[run_start..i]);
                    run_start = i;
                }
            }

            for chunk in runs.into_iter().flat_map(|r| r.chunks(BATCH_PAGES)) {
                let t_d = Instant::now();
                let mut block = Vec::with_capacity(chunk.len() * 256);
                for page in chunk {
                    block.extend_from_slice(page.data());
                }
                let batch_address = chunk[0].address();
                let batch_len = block.len();

                active.load_data(active.buffer_address(0), &block)?;
                let d_us = t_d.elapsed().as_micros();

                let t_s = Instant::now();
                active.program_block(batch_address, batch_len)?;
                let s_us = t_s.elapsed().as_micros();

                let t_w = Instant::now();
                // A freshly resumed core can still report its previous
                // halted state for a moment; waiting for "halted" before
                // it actually ran would report a fake completion while
                // the routine is still consuming the buffer. Confirm the
                // core left the halted state first, then wait for the
                // real halt.
                let leave = Instant::now();
                loop {
                    let halted = matches!(
                        active.core.status().map_err(FlashError::Core)?,
                        crate::CoreStatus::Halted(_)
                    );
                    if !halted {
                        break;
                    }
                    if leave.elapsed() > Duration::from_millis(100) {
                        return Err(FlashError::RoutineCallFailed {
                            name: "program_block",
                            error_code: 0,
                        });
                    }
                }
                let result = active
                    .wait_for_completion(Duration::from_secs(10))
                    .map_err(|error| FlashError::PageWrite {
                        page_address: batch_address,
                        source: Box::new(error),
                    })?;
                let w_us = t_w.elapsed().as_micros();

                batch_no += 1;
                eprintln!(
                    "BATCH n={batch_no} pages={} D={d_us}us S={s_us}us W={w_us}us",
                    chunk.len()
                );

                active.progress.page_programmed(batch_len as u32, t.elapsed());
                t = Instant::now();
                if result != 0 {
                    return Err(FlashError::RoutineCallFailed {
                        name: "program_block",
                        error_code: result,
                    });
                }
            }

            Ok(0)
        });

        if result.is_ok() {
            self.progress.finished_programming();
        } else {
            self.progress.failed_programming();

            result?;
        }

        Ok(())
    }

    fn program_double_buffer_per_page(
        &mut self,
        flash_encoder: &FlashEncoder,
    ) -> Result<(), FlashError> {
        let mut current_buf = 0;
        self.progress
            .started_programming(flash_encoder.program_size());

        let mut t = Instant::now();
        let result = self.run_program(|active| {
            let mut last_page_address = 0;
            for page in flash_encoder.pages() {
                // At the start of each loop cycle load the next page buffer into RAM.
                active.load_page_buffer(page.address(), page.data(), current_buf)?;

                // Then wait for the active RAM -> Flash copy process to finish.
                // Also check if it finished properly. If it didn't, return an error.
                let result =
                    active
                        .wait_for_completion(Duration::from_secs(2))
                        .map_err(|error| FlashError::PageWrite {
                            page_address: last_page_address,
                            source: Box::new(error),
                        })?;

                last_page_address = page.address();
                active.progress.page_programmed(page.size(), t.elapsed());

                t = Instant::now();
                if result != 0 {
                    return Err(FlashError::RoutineCallFailed {
                        name: "program_page",
                        error_code: result,
                    });
                }

                // Start the next copy process.
                let t_s = Instant::now();
                active.start_program_page_with_buffer(page.address(), current_buf)?;
                eprintln!("SSEG S={}us", t_s.elapsed().as_micros());

                // Swap the buffers
                if current_buf == 1 {
                    current_buf = 0;
                } else {
                    current_buf = 1;
                }
            }

            let result = active
                .wait_for_completion(Duration::from_secs(2))
                .map_err(|error| FlashError::PageWrite {
                    page_address: last_page_address,
                    source: Box::new(error),
                })?;

            if result != 0 {
                Err(FlashError::RoutineCallFailed {
                    name: "wait_for_completion",
                    error_code: result,
                })
            } else {
                Ok(0)
            }
        });

        if result.is_ok() {
            self.progress.finished_programming();
        } else {
            self.progress.failed_programming();

            result?;
        }

        Ok(())
    }

    pub(super) fn flash_layout(
        &self,
        region: &NvmRegion,
        flash_builder: &FlashBuilder,
        restore_unwritten_bytes: bool,
    ) -> Result<FlashLayout, FlashError> {
        flash_builder.build_sectors_and_pages(
            region,
            &self.flash_algorithm,
            restore_unwritten_bytes,
        )
    }
}

struct Registers {
    pc: u32,
    r0: Option<u32>,
    r1: Option<u32>,
    r2: Option<u32>,
    r3: Option<u32>,
}

impl Debug for Registers {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{:#010x} ({:?}, {:?}, {:?}, {:?})",
            self.pc, self.r0, self.r1, self.r2, self.r3
        )
    }
}

fn into_reg(val: u64) -> Result<u32, FlashError> {
    let reg_value: u32 = val
        .try_into()
        .map_err(|_| FlashError::RegisterValueNotSupported(val))?;

    Ok(reg_value)
}

pub(super) struct ActiveFlasher<'probe, O: Operation> {
    core: Core<'probe>,
    rtt: Option<crate::rtt::Rtt>,
    memory_map: Vec<MemoryRegion>,
    progress: FlashProgress,
    flash_algorithm: FlashAlgorithm,
    /// Whether the core's dcsr has already been set to trap on ebreak.
    /// The bit survives halt/resume, so it only needs to be written
    /// once per flasher; re-writing it costs a CSR read-and-write
    /// cycle on every algorithm call.
    sw_breakpoint_trap_set: bool,
    /// Values of the callee-saved registers (static base, stack
    /// pointer) as left by the previous algorithm call: the algorithm
    /// preserves them, so an unchanged value does not need to be
    /// written again.
    cached_regs: [Option<u32>; 8],
    _operation: core::marker::PhantomData<O>,
}

impl<'probe, O: Operation> ActiveFlasher<'probe, O> {
    #[tracing::instrument(name = "Call to flash algorithm init", skip(self, clock))]
    pub(super) fn init(&mut self, clock: Option<u32>) -> Result<(), FlashError> {
        let algo = &self.flash_algorithm;

        let address = self.flash_algorithm.flash_properties.address_range.start;

        // Execute init routine if one is present.
        if let Some(pc_init) = algo.pc_init {
            let result = self
                .call_function_and_wait(
                    &Registers {
                        pc: into_reg(pc_init)?,
                        r0: Some(into_reg(address)?),
                        r1: clock.or(Some(0)),
                        r2: Some(O::operation()),
                        r3: None,
                    },
                    true,
                    Duration::from_secs(2),
                )
                .map_err(|error| FlashError::Init(Box::new(error)))?;

            if result != 0 {
                return Err(FlashError::RoutineCallFailed {
                    name: "init",
                    error_code: result,
                });
            }
        }

        Ok(())
    }

    // pub(super) fn session_mut(&mut self) -> &mut Session {
    //     &mut self.session
    // }

    pub(super) fn uninit(&mut self) -> Result<(), FlashError> {
        tracing::debug!("Running uninit routine.");
        let algo = &self.flash_algorithm;

        if let Some(pc_uninit) = algo.pc_uninit {
            let result = self
                .call_function_and_wait(
                    &Registers {
                        pc: into_reg(pc_uninit)?,
                        r0: Some(O::operation()),
                        r1: None,
                        r2: None,
                        r3: None,
                    },
                    false,
                    Duration::from_secs(2),
                )
                .map_err(|error| FlashError::Uninit(Box::new(error)))?;

            if result != 0 {
                return Err(FlashError::RoutineCallFailed {
                    name: "uninit",
                    error_code: result,
                });
            }
        }
        Ok(())
    }

    fn call_function_and_wait(
        &mut self,
        registers: &Registers,
        init: bool,
        duration: Duration,
    ) -> Result<u32, FlashError> {
        self.call_function(registers, init)?;
        self.wait_for_completion(duration)
    }

    fn call_function(&mut self, registers: &Registers, init: bool) -> Result<(), FlashError> {
        tracing::debug!("Calling routine {:?}, init={})", registers, init);

        let algo = &self.flash_algorithm;
        let regs: &'static CoreRegisters = self.core.registers();

        let registers = [
            (self.core.program_counter(), Some(registers.pc)),
            (regs.argument_register(0), registers.r0),
            (regs.argument_register(1), registers.r1),
            (regs.argument_register(2), registers.r2),
            (regs.argument_register(3), registers.r3),
            (
                regs.core_register(9),
                if init {
                    Some(into_reg(algo.static_base)?)
                } else {
                    None
                },
            ),
            (
                self.core.stack_pointer(),
                if init {
                    Some(into_reg(algo.begin_stack)?)
                } else {
                    None
                },
            ),
            (
                self.core.return_address(),
                // For ARM Cortex-M cores, we have to add 1 to the return address,
                // to ensure that we stay in Thumb mode.
                if self.core.instruction_set()? == InstructionSet::Thumb2 {
                    Some(into_reg(algo.load_address + 1)?)
                } else {
                    Some(into_reg(algo.load_address)?)
                },
            ),
        ];

        for (slot, (description, value)) in registers.iter().enumerate() {
            if let Some(v) = value {
                let v = *v;
                // The callee-saved registers (static base and stack
                // pointer) keep their values across an algorithm call,
                // so re-writing them when the previous call already set
                // the same value is redundant. Every other register -
                // pc, the arguments, and the scratch registers - is
                // clobbered by the algorithm run and must always be
                // written.
                let cacheable = matches!(slot, 5 | 6);
                if cacheable && self.cached_regs[slot] == Some(v) {
                    continue;
                }
                if cacheable {
                    self.cached_regs[slot] = Some(v);
                }
                self.core.write_core_reg(*description, v)?;

                if tracing::enabled!(Level::DEBUG) {
                    let value: u32 = self.core.read_core_reg(*description)?;

                    tracing::debug!(
                        "content of {} {:#x}: {:#010x} should be: {:#010x}",
                        description.name(),
                        description.id.0,
                        value,
                        v
                    );
                }
            }
        }

        // Ensure RISC-V `ebreak` instructions enter debug mode,
        // this is necessary for soft breakpoints to work. The dcsr bit
        // survives halt/resume and nothing in the flasher lifetime
        // resets it, so writing it once is enough.
        if !self.sw_breakpoint_trap_set {
            self.core.debug_on_sw_breakpoint(true)?;
            self.sw_breakpoint_trap_set = true;
        }

        // Resume target operation.
        self.core.run()?;

        if let Some(rtt_address) = self.flash_algorithm.rtt_control_block {
            // FIXME: replace this with try_attach_to_rtt once it's been moved to the library
            let now = Instant::now();
            let mut last_error = None;
            while self.rtt.is_none() {
                std::thread::sleep(Duration::from_millis(1));
                let rtt = match crate::rtt::Rtt::attach_region(
                    &mut self.core,
                    &self.memory_map,
                    &crate::rtt::ScanRegion::Exact(rtt_address),
                ) {
                    Ok(rtt) => Some(rtt),
                    Err(error) => {
                        last_error = Some(error);
                        None
                    }
                };
                self.rtt = rtt;

                if now.elapsed() > Duration::from_secs(1) {
                    break;
                }
            }
            if let Some(error) = last_error {
                tracing::error!("RTT could not be initialized: {error}");
            }
        }

        Ok(())
    }

    #[tracing::instrument(skip(self))]
    pub(super) fn wait_for_completion(&mut self, timeout: Duration) -> Result<u32, FlashError> {
        tracing::debug!("Waiting for routine call completion.");
        let regs = self.core.registers();

        // Wait until halted state is active again.
        let start = Instant::now();

        let mut timeout_ocurred = true;
        while start.elapsed() < timeout {
            match self.core.status()? {
                crate::CoreStatus::Halted(_) => {
                    timeout_ocurred = false;
                    // Once the core is halted we know for sure all RTT data is written
                    // so we can read all of it.
                    self.read_rtt()?;
                    break;
                }
                crate::CoreStatus::LockedUp => {
                    return Err(FlashError::UnexpectedCoreStatus {
                        status: crate::CoreStatus::LockedUp,
                    });
                }
                _ => {
                    // All other statuses are okay: we'll just keep polling.
                }
            }

            // Periodically read RTT.
            self.read_rtt()?;

            std::thread::sleep(Duration::from_millis(1));
        }

        if timeout_ocurred {
            return Err(FlashError::Core(crate::Error::Timeout));
        }

        let r: u32 = self.core.read_core_reg(regs.result_register(0))?;
        Ok(r)
    }

    fn read_rtt(&mut self) -> Result<(), FlashError> {
        if let Some(rtt) = &mut self.rtt {
            for channel in rtt.up_channels().iter() {
                let mut buffer = vec![0; channel.buffer_size()];
                match channel.read(&mut self.core, &mut buffer) {
                    Ok(read) if read > 0 => {
                        let message = String::from_utf8_lossy(&buffer[..read]).to_string();
                        let channel = channel.name().unwrap_or("unnamed");
                        tracing::debug!("RTT({channel}): {message}");
                        self.progress.message(message);
                    }
                    Ok(_) => (),
                    Err(error) => tracing::debug!("Reading RTT failed: {error}"),
                };
            }
        }
        Ok(())
    }
}

impl<'probe> ActiveFlasher<'probe, Erase> {
    pub(super) fn erase_all(&mut self) -> Result<(), FlashError> {
        tracing::debug!("Erasing entire chip.");
        let flasher = self;
        let algo = &flasher.flash_algorithm;

        if let Some(pc_erase_all) = algo.pc_erase_all {
            let result = flasher
                .call_function_and_wait(
                    &Registers {
                        pc: into_reg(pc_erase_all)?,
                        r0: None,
                        r1: None,
                        r2: None,
                        r3: None,
                    },
                    false,
                    Duration::from_secs(30),
                )
                .map_err(|error| FlashError::ChipEraseFailed {
                    source: Box::new(error),
                })?;

            if result != 0 {
                Err(FlashError::ChipEraseFailed {
                    source: Box::new(FlashError::RoutineCallFailed {
                        name: "chip_erase",
                        error_code: result,
                    }),
                })
            } else {
                Ok(())
            }
        } else {
            Err(FlashError::ChipEraseNotSupported)
        }
    }

    pub(super) fn erase_sector(&mut self, address: u64) -> Result<(), FlashError> {
        tracing::info!("Erasing sector at address {:#010x}", address);
        let t1 = Instant::now();

        let result = self.call_function_and_wait(
            &Registers {
                pc: into_reg(self.flash_algorithm.pc_erase_sector)?,
                r0: Some(into_reg(address)?),
                r1: None,
                r2: None,
                r3: None,
            },
            false,
            Duration::from_millis(
                self.flash_algorithm.flash_properties.erase_sector_timeout as u64,
            ),
        )?;
        tracing::info!(
            "Done erasing sector. Result is {}. This took {:?}",
            result,
            t1.elapsed()
        );

        if result != 0 {
            Err(FlashError::RoutineCallFailed {
                name: "erase_sector",
                error_code: result,
            })
        } else {
            Ok(())
        }
    }
}

impl<'p> ActiveFlasher<'p, Program> {
    /// Transfers the buffer bytes to RAM.
    fn load_data(&mut self, address: u64, bytes: &[u8]) -> Result<(), FlashError> {
        tracing::debug!(
            "Loading {} bytes of data into RAM at address {:#010x}\n",
            bytes.len(),
            address
        );
        // TODO: Prevent security settings from locking the device.

        // In case some of the previous preprocessing forgets to pad the last page,
        // we will fill the missing bytes with the erased byte value.
        let empty = self.flash_algorithm.flash_properties.erased_byte_value;
        let words: Vec<u32> = bytes
            .chunks(core::mem::size_of::<u32>())
            .map(|a| {
                u32::from_le_bytes([
                    a[0],
                    a.get(1).copied().unwrap_or(empty),
                    a.get(2).copied().unwrap_or(empty),
                    a.get(3).copied().unwrap_or(empty),
                ])
            })
            .collect();

        let t1 = Instant::now();

        self.core
            .write_32(address, &words)
            .map_err(FlashError::Core)?;

        tracing::info!(
            "Took {:?} to download {} byte page into ram",
            t1.elapsed(),
            bytes.len()
        );

        Ok(())
    }

    pub(super) fn program_page(&mut self, address: u64, bytes: &[u8]) -> Result<(), FlashError> {
        let t1 = Instant::now();

        tracing::info!(
            "Flashing page at address {:#08x} with size: {}",
            address,
            bytes.len()
        );

        // Transfer the bytes to RAM.
        let begin_data = self.buffer_address(0);
        self.load_data(begin_data, bytes)?;

        let result = self
            .call_function_and_wait(
                &Registers {
                    pc: into_reg(self.flash_algorithm.pc_program_page)?,
                    r0: Some(into_reg(address)?),
                    r1: Some(bytes.len() as u32),
                    r2: Some(into_reg(begin_data)?),
                    r3: None,
                },
                false,
                Duration::from_millis(
                    self.flash_algorithm.flash_properties.program_page_timeout as u64,
                ),
            )
            .map_err(|error| FlashError::PageWrite {
                page_address: address,
                source: Box::new(error),
            })?;
        tracing::info!("Flashing took: {:?}", t1.elapsed());

        if result != 0 {
            Err(FlashError::PageWrite {
                page_address: address,
                source: Box::new(FlashError::RoutineCallFailed {
                    name: "program_page",
                    error_code: result,
                }),
            })
        } else {
            Ok(())
        }
    }

    fn buffer_address(&self, buffer_number: usize) -> u64 {
        // Ensure the buffer number is valid, otherwise there is a bug somewhere
        // in the flashing code.
        assert!(
            buffer_number < self.flash_algorithm.page_buffers.len(),
            "Trying to use non-existing buffer ({}/{}) for flashing. This is a bug. Please report it.",
            buffer_number, self.flash_algorithm.page_buffers.len()
        );

        self.flash_algorithm.page_buffers[buffer_number]
    }

    pub(super) fn start_program_page_with_buffer(
        &mut self,
        address: u64,
        buffer_number: usize,
    ) -> Result<(), FlashError> {
        let buffer_address = self.buffer_address(buffer_number);

        self.call_function(
            &Registers {
                pc: into_reg(self.flash_algorithm.pc_program_page)?,
                r0: Some(into_reg(address)?),
                r1: Some(self.flash_algorithm.flash_properties.page_size),
                r2: Some(into_reg(buffer_address)?),
                r3: None,
            },
            false,
        )
        .map_err(|error| FlashError::PageWrite {
            page_address: address,
            source: Box::new(error),
        })?;

        Ok(())
    }

    /// Program a contiguous block of arbitrary length with one routine
    /// call. The routine's program entry passes the length through to
    /// the flash driver, which loops the pages internally.
    pub(super) fn program_block(
        &mut self,
        address: u64,
        length: usize,
    ) -> Result<(), FlashError> {
        let buffer_address = self.buffer_address(0);

        self.call_function(
            &Registers {
                pc: into_reg(self.flash_algorithm.pc_program_page)?,
                r0: Some(into_reg(address)?),
                r1: Some(length as u32),
                r2: Some(into_reg(buffer_address)?),
                r3: None,
            },
            false,
        )
    }

    pub(super) fn load_page_buffer(
        &mut self,
        _address: u64,
        bytes: &[u8],
        buffer_number: usize,
    ) -> Result<(), FlashError> {
        let buffer_address = self.buffer_address(buffer_number);
        self.load_data(buffer_address, bytes)?;

        Ok(())
    }
}
