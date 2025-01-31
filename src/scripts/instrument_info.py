from typing import Final

class VAMPIRESInfo:
    iwas: Final = {
        "20230707_VAMPIRES": 105,
        "20240727_VAMPIRES": 59,
        "20240728_VAMPIRES": 59,
        "20240729_VAMPIRES": 59,
    }


vampires_info = VAMPIRESInfo()


class ZIMPOLInfo:
    pxscale = 3.6e-3  # arc/px


zimpol_info = ZIMPOLInfo()


class NACOInfo:
    ...


naco_info = NACOInfo()


class IRDISInfo:
    pxscale = 12.25e-3  # arc/px


irdis_info = IRDISInfo()


class GPIInfo:
    pxscale = 14.14e-3

gpi_info = GPIInfo()