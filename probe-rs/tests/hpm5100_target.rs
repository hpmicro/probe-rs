//! Loading test for the HPM5100 series target definition.
//!
//! Guards against silently-dropped yaml fields (the variant-level scan
//! chain lives under the `jtag:` key and would be ignored silently if
//! misplaced, since Chip has no deny_unknown_fields).

use probe_rs::config::get_target_by_name;
use probe_rs_target::MemoryRegion;

#[test]
fn hpm5151_target_loads() {
    let target = get_target_by_name("HPM5151").expect("HPM5151 must load");

    assert_eq!(target.name, "HPM5151");

    // Memory map: four regions with the documented bounds.
    let mm = &target.memory_map;
    assert_eq!(mm.len(), 4, "ILM + DLM + AHB_SRAM + XPI0 expected");
    let starts: Vec<u64> = mm
        .iter()
        .map(|r| match r {
            MemoryRegion::Ram(region) => region.range.start,
            MemoryRegion::Nvm(region) => region.range.start,
            _ => unreachable!("no other region kinds in this target"),
        })
        .collect();
    assert!(starts.contains(&0x0000_0000), "ILM at 0x0");
    assert!(starts.contains(&0x0020_0000), "DLM at 0x0020_0000");
    assert!(starts.contains(&0xF020_0000), "AHB_SRAM at 0xF020_0000");
    assert!(starts.contains(&0x8000_0000), "XPI0 at 0x8000_0000");

    // Scan chain must be delivered (not silently dropped).
    let chain = target
        .jtag
        .as_ref()
        .expect("jtag section must parse")
        .scan_chain
        .as_ref()
        .expect("scan chain must parse from the jtag key");
    assert_eq!(chain[0].ir_len, Some(5));

    // Flash algorithm: blob decodes non-empty and loads inside the ILM
    // window, at or past the 8-byte blob header.
    let algo = target
        .flash_algorithms
        .iter()
        .find(|a| a.name == "flash-algo-hpm51x1")
        .expect("flash algorithm flash-algo-hpm51x1");
    let load_address = algo.load_address.expect("load address set");
    assert!(load_address >= 0x8);
    assert!(load_address < 0x1_0000, "load address inside ILM window");
    assert!(!algo.instructions.is_empty());
    assert_eq!(
        algo.flash_properties.page_size, 0x100,
        "page size is the FLM program granularity (sector stays 4 KiB)"
    );
    assert_eq!(
        algo.flash_properties.address_range.start, 0x8000_0000,
        "flash base passed to the loader via r0"
    );

    // Core list.
    let cores: Vec<&str> = target.cores.iter().map(|c| c.name.as_str()).collect();
    assert_eq!(cores, vec!["core0"]);
}

#[test]
fn hpm5151_entry_points_within_blob() {
    let target = get_target_by_name("HPM5151").expect("HPM5151 must load");
    let algo = target
        .flash_algorithms
        .iter()
        .find(|a| a.name == "flash-algo-hpm51x1")
        .expect("flash algorithm flash-algo-hpm51x1");

    // The blob is a byte array; its memory span starts at the load
    // address. Entry points and the data-section offset are offsets
    // relative to the load address, so the blob bounds live in offset
    // space: [0, blob_len).
    let blob_span = algo.instructions.len() as u64;
    assert!(blob_span > 0);
    // Every code entry point must land inside the blob - a blob/entry
    // mismatch would jump into garbage on the first flash operation.
    for entry in [
        algo.pc_init,
        algo.pc_uninit,
        Some(algo.pc_program_page),
        Some(algo.pc_erase_sector),
        algo.pc_erase_all,
    ]
    .into_iter()
    .flatten()
    {
        assert!(
            entry < blob_span,
            "entry offset {entry:#x} outside blob offsets [0, {blob_span:#x})"
        );
    }

    // The data section is NOT part of the carried blob: its offset points
    // past the loaded bytes (the loader writes it into r9 as
    // load_address + offset; nothing is loaded there). Pin that it stays
    // outside the blob - treating the offset as an in-blob entry would be
    // a blob/entry mismatch of exactly the kind guarded above.
    assert!(
        algo.data_section_offset >= blob_span,
        "data section offset {:#x} must lie past the blob end {blob_span:#x}",
        algo.data_section_offset
    );
}

#[test]
fn hpm5151_memory_map_bounds() {
    use probe_rs_target::MemoryRegion;

    let target = get_target_by_name("HPM5151").expect("HPM5151 must load");
    let mm = &target.memory_map;

    let ends: Vec<u64> = mm
        .iter()
        .map(|r| match r {
            MemoryRegion::Ram(region) => region.range.end,
            MemoryRegion::Nvm(region) => region.range.end,
            _ => unreachable!("no other region kinds in this target"),
        })
        .collect();
    assert!(ends.contains(&0x4_0000), "ILM ends at 256 KiB");
    assert!(ends.contains(&0x22_0000), "DLM ends at 128 KiB");
    assert!(ends.contains(&0xF020_8000), "AHB_SRAM ends at 32 KiB");
    assert!(ends.contains(&0x8010_0000), "XPI0 ends at 1 MiB");

    let xpi0 = mm
        .iter()
        .find_map(|r| match r {
            MemoryRegion::Nvm(region) if region.range.start == 0x8000_0000 => Some(region),
            _ => None,
        })
        .expect("XPI0 NVM region");
    assert!(xpi0.is_boot_memory, "XPI0 is the boot memory");
    assert_eq!(xpi0.cores, vec!["core0"]);

    // The flash algorithm's range must exactly cover the NVM region, and
    // its geometry must divide evenly.
    let algo = target
        .flash_algorithms
        .iter()
        .find(|a| a.name == "flash-algo-hpm51x1")
        .expect("flash algorithm flash-algo-hpm51x1");
    assert_eq!(algo.flash_properties.address_range, xpi0.range);
    let sector_size = algo.flash_properties.sectors[0].size as u64;
    assert_eq!(sector_size, 0x1000);
    assert_eq!(
        (xpi0.range.end - xpi0.range.start) % sector_size,
        0,
        "sectors cover the region exactly"
    );
    assert_eq!(
        sector_size % algo.flash_properties.page_size as u64,
        0,
        "pages divide the sector"
    );
}

#[test]
fn hpm5151_flash_algorithm_assembles_for_xpi0() {
    // Pins a stricter invariant than the loader needs: exactly one
    // DEFAULT algorithm covers XPI0 (the loader consults `default` only
    // to break ties between multiple covering algorithms). A dry-run
    // against a fake probe is not possible for a RISC-V target (the
    // fake probe offers no RISC-V interface), so assembly is checked at
    // the data level instead.
    let target = get_target_by_name("HPM5151").expect("HPM5151 must load");

    let matching: Vec<_> = target
        .flash_algorithms
        .iter()
        .filter(|algo| algo.default && algo.flash_properties.address_range.end >= 0x8010_0000)
        .collect();
    assert_eq!(
        matching.len(),
        1,
        "exactly one default algorithm covers XPI0"
    );
    assert_eq!(matching[0].name, "flash-algo-hpm51x1");
}
