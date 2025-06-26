import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit, logit
from scipy.stats import norm, binom_test, sem
import scipy.stats as stats  
import statsmodels.formula.api as smf
import statsmodels.api as sm
# !pip install seaborn-qqplot
from seaborn_qqplot import pplot
import itertools

from python_section_1 import LAW, Experiment
from python_section_2 import AnotherExperiment
from python_section_3 import evaluate_psi, compute_eic
from python_section_6 import compute_psi_hat_ab, compute_bias_ab, compute_clt, plot_bias_ab
from python_section_7 import estimate_Gbar, estimate_QW, estimate_Qbar
from python_section_8 import compute_gcomp, get_learned_features, update_learned_features, get_psi_hat_de
from python_algorithms import WorkingModelBase, WorkingModelGOne, WorkingModelGTwo, WorkingModelGThree, WorkingModelQOne, KknnAlgo, BoostingTreeAlgo, BoostingLMAlgo

# Helper fn: choose the correct Qbar estimation format 
# (it has been appear in our code so many times, we really don't have to write it over and over)
def choose_Qbar_format(AW, Qbar_hat, algorithm=None, use_predict=False):
    # Create a DataFrame for prediction
    dat = pd.DataFrame({"W": AW["W"], "A": AW["A"], "Y": np.nan})

    if use_predict:# If using a prediction model, apply it to compute Qbar_hat     
        stratify = Qbar_hat.attrs.get("stratify", False)  # Check if stratification is required
        if not stratify:
            fit = Qbar_hat.loc[Qbar_hat["a"] == "both", "fit"].iloc[0]
            if algorithm is not None:
                pred = algorithm.predict(dat, model=fit)
            else:
                pred = fit.predict(dat)
        else:
            fit_one = Qbar_hat.loc[Qbar_hat["a"] == "one", "fit"].iloc[0]
            fit_zero = Qbar_hat.loc[Qbar_hat["a"] == "zero", "fit"].iloc[0]
            pred = np.zeros(len(dat))
            idx_one = dat["A"] == 1

            if algorithm is not None: # if a ML model is used
                if idx_one.sum() > 0:
                    pred[idx_one] = algorithm.predict(dat=dat[idx_one], model=fit_one)
                if (~idx_one).sum() > 0:
                    pred[~idx_one] = algorithm.predict(dat=dat[~idx_one], model=fit_zero)
            else:
                # if a non-ML model is used
                if idx_one.sum() > 0:
                    pred[idx_one] = fit_one.predict(dat=dat[idx_one])
                if (~idx_one).sum() > 0:
                    pred[~idx_one] = fit_zero.predict(dat=dat[~idx_one])
    else: # if the true function is used, e.g. experiment.Qbar()
        pred = Qbar_hat(AW.values) 
    return pred

# 10.2.2 Fluctuating directly
def fluctuate(AW, Qbar, Gbar, h,
              Qbar_algorithm=None,
              Qbar_use_predict=False):
    """
    Create a fluctuation function based on Qbar, Gbar, and h.

    Args:
        Qbar (callable): Function to estimate the conditional expectation of Y given A and W.
        Gbar (callable): Function to estimate the marginal law of A given W.
        h (float): Fluctuation parameter.

    Returns:
        callable: A function that computes the fluctuated Qbar for given A and W.
    """
    # Extract A and W from the input DataFrame AW
    A = AW["A"].values
    W = AW["W"].values
    

    # Compute lGAW (weighted probability of A given W)
    lGAW = A * Gbar(W) + (1 - A) * (1 - Gbar(W))

    # Compute the fluctuated Qbar using the logit and expit transformations
    Qbar_pred = choose_Qbar_format(AW, Qbar, algorithm=Qbar_algorithm, use_predict=Qbar_use_predict)
    out = expit(logit(Qbar_pred) + h * (2 * A - 1) / lGAW)

    return out

def prepare_df(Qbar_truth, Qbar_hat_trees, algorithm):

    # Generate w values
    w = np.linspace(0, 1, int(1e3))

    # Create a DataFrame for w
    df = pd.DataFrame({"w": w})

    # Add columns for truth and trees fluctuations
    df["truth_0_1"] = Qbar_truth(pd.DataFrame({"A": 1, "W": w}).values)
    df["truth_0_0"] = Qbar_truth(pd.DataFrame({"A": 0, "W": w}).values)
    df["trees_0_1"] = choose_Qbar_format(pd.DataFrame({"A": 1, "W": w}), Qbar_hat_trees, algorithm=algorithm, use_predict=True)
    df["trees_0_0"] = choose_Qbar_format(pd.DataFrame({"A": 0, "W": w}), Qbar_hat_trees, algorithm=algorithm, use_predict=True)
    df["truth_hminus_1"] = fluctuate(pd.DataFrame({"A": 1, "W": w}), experiment.Qbar, experiment.Gbar, -1)
    df["truth_hminus_0"] = fluctuate(pd.DataFrame({"A": 0, "W": w}), experiment.Qbar, experiment.Gbar, -1)
    df["trees_hminus_1"] = fluctuate(pd.DataFrame({"A": 1, "W": w}), Qbar_hat_trees, experiment.Gbar, -1,
                                     Qbar_algorithm=algorithm,
                                     Qbar_use_predict=True)
    df["trees_hminus_0"] = fluctuate(pd.DataFrame({"A": 0, "W": w}), Qbar_hat_trees, experiment.Gbar, -1,
                                     Qbar_algorithm=algorithm,
                                     Qbar_use_predict=True)
    df["truth_hplus_1"] = fluctuate(pd.DataFrame({"A": 1, "W": w}), experiment.Qbar, experiment.Gbar, 1)
    df["truth_hplus_0"] = fluctuate(pd.DataFrame({"A": 0, "W": w}), experiment.Qbar, experiment.Gbar, 1)
    df["trees_hplus_1"] = fluctuate(pd.DataFrame({"A": 1, "W": w}), Qbar_hat_trees, experiment.Gbar, 1,
                                    Qbar_algorithm=algorithm,
                                    Qbar_use_predict=True)
    df["trees_hplus_0"] = fluctuate(pd.DataFrame({"A": 0, "W": w}), Qbar_hat_trees, experiment.Gbar, 1,
                                    Qbar_algorithm=algorithm,
                                    Qbar_use_predict=True)
    
    # Pivot the DataFrame to long format
    df_long = df.melt(id_vars=["w"], var_name="f", value_name="value")

    # Extract components from the column names
    df_long[["f", "h", "a"]] = df_long["f"].str.extract(r"([^_]+)_([^_]+)_([01]+)")

    # Modify columns for visualization
    df_long["f"] = df_long["f"].replace({"truth": "Q_0", "trees": "Q_n"})
    df_long["h"] = df_long["h"].replace({"0": 0, "hplus": 1, "hminus": -1}).astype(int)
    df_long["a"] = "a=" + df_long["a"]
    df_long["fh"] = "(" + df_long["f"] + "," + df_long["h"].astype(str) + ")"
    
    return df_long

def plot_fluctuation(df_long):
    # Define a custom color palette for "hue"
    custom_palette = {0: "blue", 1: "green", -1: "red"}  # Assign colors to hue values

    # Facet by 'a'
    g = sns.FacetGrid(df_long, col="a", height=5, aspect=1.5)
    g.map_dataframe(
        sns.lineplot,
        x="w",
        y="value",
        hue="h",
        style="f",
        linewidth=3,
        units="fh",
        estimator=None,
        palette=custom_palette  # Apply the custom palette
    )
    g.set_titles("{col_name}")
    g.set_axis_labels("w", r"$f_h(a, w)$")
    g.add_legend()

    plt.show()

if __name__ == "__main__":
    # initialize an experiment
    experiment = Experiment()
    B=int(1e6)
    obs = experiment.sample_from(n=B)
    obs_df = pd.DataFrame(obs, columns=["W", "A", "Y"])

    # 10.2.2 Fluctuating directly
    Qbar_hminus = fluctuate(obs_df[["A", "W"]], experiment.Qbar, experiment.Gbar, -1)
    Qbar_hplus = fluctuate(obs_df[["A", "W"]], experiment.Qbar, experiment.Gbar, 1)

    boosting_tree_algo = BoostingTreeAlgo()
    Qbar_hat_trees = estimate_Qbar(dat=obs[:int(1e3)], algorithm=boosting_tree_algo, verbose=True)
    Qbar_trees_hminus = fluctuate(obs_df[["A", "W"]], Qbar_hat_trees, experiment.Gbar, -1,
                                  Qbar_algorithm=boosting_tree_algo,
                                  Qbar_use_predict=True)
    Qbar_trees_hplus = fluctuate(obs_df[["A", "W"]], Qbar_hat_trees, experiment.Gbar, 1,
                                 Qbar_algorithm=boosting_tree_algo,
                                 Qbar_use_predict=True)

    df_fluct = prepare_df(experiment.Qbar, Qbar_hat_trees, boosting_tree_algo)
    plot_fluctuation(df_fluct)

    
    


