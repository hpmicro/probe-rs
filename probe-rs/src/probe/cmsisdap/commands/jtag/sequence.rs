/// Implementation of the DAP_JTAG_SEQUENCE command
use super::super::{CmsisDapError, CommandId, Request, SendError, Status};

use bitvec::prelude::*;

#[derive(Clone, Copy, Debug)]
pub struct Sequence {
    /// Number of TCK cycles: 1..64 (64 encoded as 0)
    tck_cycles: u8,

    /// TDO capture
    tdo_capture: bool,

    /// TMS value
    tms: bool,

    /// Data to generate on TDI
    data: [u8; 8],
}

impl Sequence {
    /// Create a JTAG sequence, optionally capturing TDO.
    ///
    /// # Args
    ///
    /// * `tck_cycles` - The number of cycles to clock out `data` bits on TDI
    /// * `tck_capture` - Whether the probe should capture TDO
    /// * `tms` - Whether TMS should be held high or low
    /// * `tdi` - The TDI bits to clock out
    pub(crate) fn new(
        tck_cycles: u8,
        tdo_capture: bool,
        tms: bool,
        tdi: [u8; 8],
    ) -> Result<Self, CmsisDapError> {
        assert!(
            tck_cycles > 0 && tck_cycles <= 64,
            "tck_cycles = {}, but expected [1,64]",
            tck_cycles
        );

        Ok(Self {
            tck_cycles,
            tdo_capture,
            tms,
            data: tdi,
        })
    }

    /// Create a JTAG sequence, capturing TDO.
    /// The number of TCK cycles is determined by the `tdi` len.
    ///
    /// # Args
    ///
    /// * `tms` - Whether TMS should be held high or low
    /// * `tdi` - The TDI bits to clock out
    pub(crate) fn capture(tms: bool, tdi: &BitVec<u8>) -> Result<Self, CmsisDapError> {
        let tck_cycles = tdi.len();
        assert!(
            tck_cycles > 0 && tck_cycles <= 64,
            "tdi.len() = {}, but expected [1,64]",
            tck_cycles
        );

        let num_bytes = (tdi.len() + 7) / 8;
        let mut data: [u8; 8] = [0; 8];
        data[0..num_bytes].copy_from_slice(&tdi.as_raw_slice()[0..num_bytes]);

        Ok(Self {
            tck_cycles: tck_cycles as u8,
            tdo_capture: true,
            tms,
            data,
        })
    }

    /// Create a JTAG sequence, *without* capturing TDO.
    /// The number of TCK cycles is determined by the `tdi` len.
    ///
    /// # Args
    ///
    /// * `tms` - Whether TMS should be held high or low
    /// * `tdi` - The TDI bits to clock out
    pub(crate) fn no_capture(tms: bool, tdi: &BitVec<u8>) -> Result<Self, CmsisDapError> {
        let tck_cycles = tdi.len();
        assert!(
            tck_cycles > 0 && tck_cycles <= 64,
            "tdi.len() = {}, but expected [1,64]",
            tck_cycles
        );

        let num_bytes = (tdi.len() + 7) / 8;
        let mut data: [u8; 8] = [0; 8];
        data[0..num_bytes].copy_from_slice(&tdi.as_raw_slice()[0..num_bytes]);

        Ok(Self {
            tck_cycles: tck_cycles as u8,
            tdo_capture: false,
            tms,
            data,
        })
    }
    /// Append one bit sequence a JTAG sequence.
    /// It will return ok nnly if tms and capture are the same and current tck_cycles less than 64.
    pub(crate) fn append(&mut self, tms: bool, tdi: bool, capture: bool) -> Result<(), ()> {
        if self.tck_cycles < 64 && self.tms == tms && self.tdo_capture == capture {
            self.data[((self.tck_cycles) / 8) as usize] |= u8::from(tdi) << ((self.tck_cycles) % 8);
            self.tck_cycles += 1;
            Ok(())
        } else {
            Err(())
        }
    }
    /// Number of TDO bits this sequence contributes to the response
    /// (the wire format pads every captured sequence to a whole number
    /// of bytes, so consumers must skip `(captured_bits_padded -
    /// captured_bits)` padding bits after each one).
    pub(crate) fn captured_bits(&self) -> Option<usize> {
        if self.tdo_capture {
            Some(self.tck_cycles as usize)
        } else {
            None
        }
    }

    /// Total response bits the wire format reserves for this sequence,
    /// padding included.
    pub(crate) fn captured_bits_padded(&self) -> usize {
        match self.captured_bits() {
            Some(n) => (n + 7) / 8 * 8,
            None => 0,
        }
    }

    /// Bytes this sequence occupies in a DAP_JTAG_SEQUENCE command:
    /// one info byte plus the TDI payload padded to whole bytes.
    pub(crate) fn wire_bytes(&self) -> usize {
        1 + (self.tck_cycles as usize + 7) / 8
    }
}

#[derive(Clone, Debug)]
pub struct SequenceRequest {
    sequences: Vec<Sequence>,
}

impl SequenceRequest {
    pub(crate) fn new(sequences: Vec<Sequence>) -> Result<Self, CmsisDapError> {
        assert!(
            !sequences.is_empty() && sequences.len() <= (u8::MAX as usize),
            "sequences.len() == {}, but expected [1,255]",
            sequences.len()
        );
        Ok(SequenceRequest { sequences })
    }
}

impl Request for SequenceRequest {
    const COMMAND_ID: CommandId = CommandId::JtagSequence;

    type Response = SequenceResponse;

    /*
    | BYTE | BYTE **********| BYTE *********| BYTE ****|
    > 0x14 | Sequence Count | Sequence Info | TDI Data |
    |******|****************|///////////////|//////////|
     */
    fn to_bytes(&self, buffer: &mut [u8]) -> Result<usize, SendError> {
        let mut transfer_len_bytes = 0;
        buffer[transfer_len_bytes] = self.sequences.len() as u8;
        transfer_len_bytes += 1;

        self.sequences.iter().for_each(|&sequence| {
            let tck_cycles = sequence.tck_cycles & 0x3F;
            let tck_cycles = if tck_cycles == 0 { 64 } else { tck_cycles };

            let mut sequence_info = 0;
            sequence_info |= if tck_cycles == 64 { 0 } else { tck_cycles };
            sequence_info |= (sequence.tms as u8) << 6;
            sequence_info |= (sequence.tdo_capture as u8) << 7;
            buffer[transfer_len_bytes] = sequence_info;
            transfer_len_bytes += 1;

            let byte_count: usize = (tck_cycles as usize + 7) / 8;
            buffer[transfer_len_bytes..(transfer_len_bytes + byte_count)]
                .copy_from_slice(&sequence.data[..byte_count]);
            transfer_len_bytes += byte_count;
        });
        Ok(transfer_len_bytes)
    }

    fn parse_response(&self, buffer: &[u8]) -> Result<Self::Response, SendError> {
        let mut received_len_bytes = 1;
        let status = Status::from_byte(*buffer.first().ok_or(SendError::NotEnoughData)?)?;

        self.sequences.iter().for_each(|&sequence| {
            if sequence.tdo_capture {
                let tck_cycles = sequence.tck_cycles & 0x3F;
                let tck_cycles = if tck_cycles == 0 { 64 } else { tck_cycles };
                let byte_count: usize = (tck_cycles as usize + 7) / 8;
                received_len_bytes += byte_count;
            }
        });

        let response = buffer
            .get(1..received_len_bytes)
            .ok_or(SendError::NotEnoughData)?
            .to_vec();
        Ok(SequenceResponse(status, response))
    }
}

#[derive(Debug)]
pub struct SequenceResponse(pub(crate) Status, pub(crate) Vec<u8>);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_bits_and_padded_bits() {
        let sequence = Sequence::new(41, true, false, [0xFF; 8]).expect("41 cycles is in range");
        assert_eq!(sequence.captured_bits(), Some(41));
        // 41 TDO bits occupy 48 wire bits (six whole bytes).
        assert_eq!(sequence.captured_bits_padded(), 48);

        // Uncaptured sequences reserve no response bits at all.
        let sequence = Sequence::new(41, false, false, [0x00; 8]).expect("41 cycles is in range");
        assert_eq!(sequence.captured_bits(), None);
        assert_eq!(sequence.captured_bits_padded(), 0);

        // Byte-aligned captures carry no padding.
        for cycles in [8u8, 64] {
            let sequence =
                Sequence::new(cycles, true, false, [0x00; 8]).expect("cycles is in range");
            assert_eq!(sequence.captured_bits(), Some(cycles as usize));
            assert_eq!(sequence.captured_bits_padded(), cycles as usize);
        }
    }

    #[test]
    fn wire_bytes_matches_request_layout() {
        // One info byte plus the TDI payload padded to whole bytes.
        for (cycles, expected) in [(1u8, 2usize), (8, 2), (9, 3), (41, 7), (64, 9)] {
            let sequence =
                Sequence::new(cycles, false, false, [0x00; 8]).expect("cycles is in range");
            assert_eq!(sequence.wire_bytes(), expected, "tck_cycles = {cycles}");
        }
    }

    #[test]
    fn short_response_is_an_error_not_a_panic() {
        // A device answering with fewer bytes than the captured sequences
        // occupy must surface NotEnoughData; the parsing layer must never
        // panic on a malformed response.
        let sequence = Sequence::new(41, true, false, [0x00; 8]).expect("cycles is in range");
        let request = SequenceRequest::new(vec![sequence]).expect("one sequence is in range");

        // No payload at all.
        assert!(matches!(
            request.parse_response(&[]),
            Err(SendError::NotEnoughData)
        ));
        // Status byte only: the 48 capture bits (6 bytes) are missing.
        assert!(matches!(
            request.parse_response(&[0x00]),
            Err(SendError::NotEnoughData)
        ));
        // Complete response parses.
        let full = [0x00u8, 1, 2, 3, 4, 5, 6];
        assert!(request.parse_response(&full).is_ok());
    }
}
