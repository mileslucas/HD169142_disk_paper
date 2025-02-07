from astropy import time

def label_from_folder(foldername: str) -> str:
    tokens = foldername.split("_")
    date = f"{tokens[0][:4]}/{tokens[0][4:6]}/{tokens[0][6:]}"
    return f"{date} {tokens[1]} ({tokens[2]})"


def time_from_folder(foldername: str) -> time.Time:
    date_raw = foldername.split("_")[0]
    ymd = {
        "year": int(date_raw[:4]),
        "month": int(date_raw[4:6]),
        "day": int(date_raw[6:])
    }
    return time.Time(ymd, format="ymdhms")


def get_time_delta_yr(folder1: str, folder2: str) -> float:
    time1 = time_from_folder(folder1)
    time2 = time_from_folder(folder2)
    return (time2 - time1).jd / 365.25


folders = [
    "20120726_NACO_H",
    "20140425_GPI_J",
    "20150503_IRDIS_J",
    "20150710_ZIMPOL_VBB",
    # "20170918_ALMA_1.3mm",
    "20180715_ZIMPOL_VBB",
    "20210906_IRDIS_Ks",
    # "20230604_CHARIS_JHK",
    "20230707_VAMPIRES_MBI",
    "20240729_VAMPIRES_MBI",
]

pxscales = {
    "20120726_NACO_H": 27e-3,
    "20140425_GPI_J": 14.14e-3,
    "20150503_IRDIS_J": 12.25e-3,
    "20150710_ZIMPOL_VBB": 3.6e-3,
    "20170918_ALMA_1.3mm": 5e-3,
    "20180715_ZIMPOL_VBB": 3.6e-3,
    "20230604_CHARIS_JHK": 15.16e-3,
    "20230707_VAMPIRES_MBI": 5.9e-3,
    "20210906_IRDIS_Ks": 12.25e-3,
    "20240729_VAMPIRES_MBI": 5.9e-3,
}