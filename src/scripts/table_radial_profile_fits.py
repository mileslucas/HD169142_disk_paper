import paths
import numpy as np
import pandas as pd
import arviz

dates = ("20230707", "20240729")
names = ["F610", "F670", "F720", "F760"]

param_names = ["r_inner", "A_inner", "g1_inner", "g2_inner"]
param_names += ["r_outer", "A_outer", "g1_outer", "g2_outer"]

if __name__ == "__main__":
    dfs = []

    for i, date in enumerate(dates):
        for wl_idx, filt_name in enumerate(names):
            data = np.load(
                paths.data
                / date
                / f"{date}_HD169142_vampires_{filt_name}_radial_profile_posteriors.npz"
            )
            traces = data["samples"][None, :, :]
            summ = arviz.summary(traces, kind="stats", hdi_prob=0.95)
            summ.index = param_names
            summ.insert(0, "filter", filt_name)
            summ.insert(0, "date", date)
            dfs.append(summ)


    table = pd.concat(dfs)
    table.to_csv(paths.data / "HD169142_vampires_posterior_summary.csv")