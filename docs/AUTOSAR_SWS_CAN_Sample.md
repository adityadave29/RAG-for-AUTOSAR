# AUTOSAR Classic Platform - Specification of CAN Driver (Sample SWS)

## 1. Introduction and Functional Overview
This specification describes the functionality, API, and configuration of the AUTOSAR Classic Platform CAN Driver module (Can).
The CAN driver is part of the microcontroller abstraction layer (MCAL). It abstracts the CAN hardware controller and provides a standardized interface to the CAN Interface (CanIf) layer.

## 2. CAN Controller State Machine
The CAN Driver module manages state transitions of the CAN hardware controller according to AUTOSAR SWS requirements.

[SWS_Can_00017] The CAN Driver shall provide the function `Can_SetControllerMode(Controller, Transition)` to switch between the following modes:
- `CAN_CS_UNINIT`: Controller uninitialized.
- `CAN_CS_STOPPED`: Controller stopped, no CAN communication possible.
- `CAN_CS_STARTED`: Controller active, participating in CAN bus communication.
- `CAN_CS_SLEEP`: Controller in low power sleep state.

[SWS_Can_00020] When switching to `CAN_CS_STARTED`, the CAN controller shall synchronize with the bus before allowing any transmission or reception of CAN frames.

## 3. Transmission Process
[SWS_Can_00100] The function `Can_Write(Hth, PduInfo)` shall initiate the transmission of a CAN frame.
- `Hth`: Hardware Transmit Handle referencing the mailbox configuration.
- `PduInfo`: Pointer to SDU data buffer and DLC length.

[SWS_Can_00105] If all Hardware Transmit Objects associated with the Hth are busy, `Can_Write` shall return `CAN_BUSY`. Otherwise, it returns `CAN_OK`.
