use super::super::{CommandId, Request, SendError};

#[derive(Clone, Copy, Debug)]
pub enum ConnectRequest {
    DefaultPort = 0x00,
    Swd = 0x01,
    Jtag = 0x02,
    /// cJTAG (IEEE 1149.7). Vendor extension: the probe firmware performs
    /// the OScan1 activation when this port is selected.
    CJtag = 0x03,
}

impl Request for ConnectRequest {
    const COMMAND_ID: CommandId = CommandId::Connect;

    type Response = ConnectResponse;

    fn to_bytes(&self, buffer: &mut [u8]) -> Result<usize, SendError> {
        buffer[0] = *self as u8;
        Ok(1)
    }

    fn parse_response(&self, buffer: &[u8]) -> Result<Self::Response, SendError> {
        match buffer[0] {
            0 => Ok(ConnectResponse::InitFailed),
            1 => Ok(ConnectResponse::SuccessfulInitForSWD),
            2 => Ok(ConnectResponse::SuccessfulInitForJTAG),
            3 => Ok(ConnectResponse::SuccessfulInitForCJTAG),
            other => Err(SendError::ConnectResponseError(other)),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ConnectResponse {
    InitFailed = 0x00,
    SuccessfulInitForSWD = 0x01,
    SuccessfulInitForJTAG = 0x02,
    SuccessfulInitForCJTAG = 0x03,
}

#[cfg(test)]
mod cjtag_tests {
    use super::*;

    #[test]
    fn cjtag_connect_request_encodes_port_three() {
        let mut buf = [0u8; 8];
        let n = ConnectRequest::CJtag.to_bytes(&mut buf).unwrap();
        assert_eq!(n, 1);
        // DAP_CONNECT payload: wire bytes are [command_id=0x02, port=0x03];
        // the command id byte is prepended by the transport layer, not by
        // to_bytes, so only the port byte is asserted here.
        assert_eq!(buf[0], 0x03);
    }

    #[test]
    fn cjtag_connect_response_parses() {
        assert_eq!(
            ConnectRequest::CJtag.parse_response(&[0x03]).unwrap(),
            ConnectResponse::SuccessfulInitForCJTAG
        );
        assert!(ConnectRequest::CJtag.parse_response(&[0x09]).is_err());
    }

    #[test]
    fn connect_response_rejects_all_out_of_range_values() {
        for value in 0x04..=0xFFu8 {
            assert!(
                ConnectRequest::CJtag.parse_response(&[value]).is_err(),
                "response byte {value:#x} must be rejected"
            );
        }
    }

    #[test]
    fn connect_cjtag_discriminants_pinned() {
        // The port value and the confirmation response are a wire contract
        // with the probe firmware; guard them against accidental renumbering.
        assert_eq!(ConnectRequest::CJtag as u8, 0x03);
        assert_eq!(ConnectResponse::SuccessfulInitForCJTAG as u8, 0x03);
    }
}
