//! GDB flash download: buffer `vFlashWrite` data in a FlashLoader and
//! commit it on `vFlashDone` — the same loader/commit path the CLI
//! download uses.

use crate::flashing::DownloadOptions;
use crate::gdb_server::target::RuntimeTarget;
use gdbstub::target::ext::flash::Flash;
use gdbstub::target::TargetError;

impl Flash for RuntimeTarget<'_> {
    fn flash_write(&mut self, addr: u64, data: &[u8]) -> gdbstub::target::TargetResult<(), Self> {
        // A client-supplied address near u64::MAX would overflow the
        // end-address arithmetic inside the loader; reject it up front
        // instead of panicking (debug) or silently no-oping (release).
        if addr.checked_add(data.len() as u64).is_none() {
            tracing::error!("GDB flash download: address {addr:#010x} + length overflows");
            return Err(TargetError::NonFatal);
        }

        // GDB guarantees the writes of one load arrive in increasing
        // address order. A write below the high-water mark of already
        // buffered data violates that guarantee, which can only mean a
        // new load began on this connection (an earlier one aborted
        // before its vFlashDone): its stale bytes must not leak into
        // this load — drop them and start from a clean buffer.
        if let Some(high) = self.flash_high_water {
            if addr < high {
                tracing::warn!(
                    "GDB flash download: write at {addr:#010x} below the buffered                      high-water mark {high:#010x}; dropping stale staged data"
                );
                self.flash_loader = None;
            }
        }
        self.flash_high_water = Some(
            self.flash_high_water
                .map_or(addr, |high| high.max(addr + data.len() as u64)),
        );

        let session = self.session.lock();
        if self.flash_loader.is_none() {
            self.flash_loader = Some(session.target().flash_loader());
        }
        let loader = self
            .flash_loader
            .as_mut()
            .expect("flash loader was just created");
        loader.add_data(addr, data).map_err(|e| {
            tracing::error!("GDB flash download: buffering {addr:#010x} failed: {e}");
            TargetError::NonFatal
        })?;
        Ok(())
    }

    fn flash_erase(&mut self, _addr: u64, _length: u64) -> gdbstub::target::TargetResult<(), Self> {
        // Erasing happens as part of the commit (the loader erases the
        // sectors it writes). The packet is answered OK so GDB proceeds
        // with the writes. Leftover buffered data from an aborted load
        // is NOT dropped here: the RSP spec permits a vFlashErase for a
        // higher address between vFlashWrite packets, and discarding the
        // buffer then would silently lose already-buffered data. Residue
        // is handled at vFlashDone (always) and at new-connection
        // boundaries instead.
        self.saw_flash_erase = true;
        Ok(())
    }

    fn flash_done(&mut self) -> gdbstub::target::TargetResult<(), Self> {
        // An erase-only transaction (GDB's `flash-erase` command sends
        // vFlashErase packets followed by vFlashDone) cannot be served:
        // the loader only erases sectors it programs, and pretending
        // success would be a silent no-op. Fail loudly instead.
        if self.flash_loader.is_none() {
            if self.saw_flash_erase {
                self.saw_flash_erase = false;
                tracing::error!("GDB flash-erase without writes is not supported by this stub");
                return Err(TargetError::NonFatal);
            }
            return Ok(());
        }
        self.saw_flash_erase = false;
        self.flash_high_water = None;

        let loader = self
            .flash_loader
            .take()
            .expect("flash loader presence was just checked");
        let mut session = self.session.lock();
        loader
            .commit(&mut session, DownloadOptions::default())
            .map_err(|e| {
                tracing::error!("GDB flash download: commit failed: {e}");
                TargetError::NonFatal
            })
    }
}
