"""Parametric-bootstrap intervals using only the history available before final test."""
import numpy as np
import pandas as pd

from .adequacy import recovery
from .data import ROOT, config, save_json
from .models import fit_hmm
from .research import dataset, load_selection


def parameter_intervals():
    cfg, selection = config(), load_selection()
    rows = []
    for asset in cfg["assets"]:
        frame, columns = dataset(asset)
        train = frame.loc[:cfg["validation_end"]].dropna(subset=columns)
        k = selection["choices"][asset]["k"]
        truth = fit_hmm(train, columns, k, cfg["seeds"])
        path = ROOT / f"outputs/research/adequacy/{asset}_parameter_bootstrap.csv"
        if path.exists():
            replicates = pd.read_csv(path)
        else:
            print(f"Parameter bootstrap {asset}: 100 replicates, n={len(train)}", flush=True)
            replicates, _ = recovery(truth, len(train), repetitions=100)
            replicates.to_csv(path, index=False)
        profiles = truth.raw_profiles()[truth.order]
        for state in range(k):
            persistence = truth.transition[state, state]
            estimates = {"persistence": persistence, "duration": 1/(1-persistence),
                         "mean_return": profiles[state, 0],
                         "volatility": profiles[state, len(columns)] * np.sqrt(365)}
            for parameter, estimate in estimates.items():
                values = replicates[f"{parameter}_R{state+1}"].replace([np.inf, -np.inf], np.nan).dropna()
                rows.append({"asset": asset, "state": state, "parameter": parameter, "estimate": estimate,
                    "ci_low": values.quantile(.025), "ci_high": values.quantile(.975),
                    "valid_replicates": len(values), "n_train": len(train), "fit_end": cfg["validation_end"],
                    "interval": "Conditional parametric-bootstrap percentile, Gaussian HMM"})
    output = pd.DataFrame(rows)
    output.to_csv(ROOT / "results/parameter_intervals.csv", index=False)
    save_json(ROOT / "results/parameter_interval_notes.json",
        {"replicates": 100, "initializations_each": 20, "test_data_used": False,
         "limitation": "Conditional Gaussian model and state matching; 100 replicates give coarse tail quantiles. Not a guarantee of coverage under misspecification."})
    return output
