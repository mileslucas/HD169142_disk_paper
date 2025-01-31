import pandas as pd
import paths
import numpy as np

table_files = paths.data.glob("**/*vampires_table.csv")
tables = [pd.read_csv(fname) for fname in table_files]


def delta_rot(pas):
    angles = np.mod(np.where(pas < 0, pas + 360, pas), 360)
    return np.ptp(angles)


# metrics = [np.load(filename) for filename in sorted((paths.data / "20230707" / "metrics").glob("20230707_HD169142*.npz"))]
# strehls = np.concatenate([met["nvar"] for met in metrics], axis=(0, 1))
def get_stats(table):
    sub = table.query("U_CAMERA == 1")
    used_flc = sub["U_FLCEN"].unique()[0] and sub["U_FLCST"].unique()[0].startswith(
        "IN"
    )
    stats = {
        "DATE": sub["DATE-OBS"].unique()[0],
        "OBJECT": sub["OBJECT"].unique()[0],
        "FILTER": "MBI",
        "Coro.": sub["U_FLDSTP"].unique()[0],
        "Det. Mode": sub["U_DETMOD"].unique()[0].title(),
        "Pol. Mode": "Fast" if used_flc else "Slow",
        "DIT (s)": np.round(sub["EXPTIME"].mean(), 2),
        "NCOADD": int(np.round(sub["DPP COADD NCOADD"].mean())),
        "NFRAMES": len(sub),
        "TINT (min)": int(np.round(sub["DPP COADD TINT"].sum() / 60)),
        "DELTA PA (deg)": int(np.round(delta_rot(sub["D_IMRPAD"].values))),
        "Prop ID": sub["PROP-ID"].unique()[0],
    }
    return stats


table = pd.DataFrame([get_stats(table) for table in tables])
table.sort_values(["DATE", "TINT (min)"], inplace=True, ascending=[True, False])
table.to_csv(paths.data / "obs_log.csv", index=False)
print(table)
