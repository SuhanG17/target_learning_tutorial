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
from matplotlib.ticker import FuncFormatter

from python_section_1 import LAW, Experiment
from python_section_2 import AnotherExperiment
from python_section_3 import evaluate_psi, compute_eic
from python_section_6 import compute_psi_hat_ab, compute_bias_ab, compute_clt, plot_bias_ab
from python_section_7 import estimate_Gbar, estimate_QW, estimate_Qbar
from python_section_8 import compute_gcomp, get_learned_features, update_learned_features, get_psi_hat_de
from python_section_9 import apply_one_step_correction
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

def choose_Gbar_format(W, Gbar_hat, use_predict=False):
    # create a DataFrame for prediction
    dat = pd.DataFrame({"W": W, "A": np.nan, "Y": np.nan})

    if use_predict:  # If using a prediction model, apply it to compute Gbar_hat
        GW = Gbar_hat.predict(dat)
    else:
        # If using the true function, e.g. experiment.Gbar()
        GW = Gbar_hat(W)

    return GW

#  This function first appear in 10.2.4 Targeted roaming of a fluctuation.
#  But we think it can be combined with others, so we put it here.
def compute_lGbar_hatAW(GW, A, threshold=0.05):
    # Ensure threshold is within valid range
    if not (0 <= threshold <= 0.5):
        raise ValueError("Threshold must be between 0 and 0.5.")

    # Compute lGAW
    lGAW = A * GW + (1 - A) * (1 - GW)

    # Apply thresholding
    pred = np.minimum(1 - threshold, np.maximum(lGAW, threshold))

    return pred 


# 10.2.2 Fluctuating directly
def fluctuate(AW, Qbar, Gbar, h,
              Qbar_algorithm=None,
              Qbar_use_predict=False,
              Gbar_use_predict=False,
              lGAW_threshold=0):
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
    # lGAW = A * Gbar(W) + (1 - A) * (1 - Gbar(W))
    GW_pred = choose_Gbar_format(W, Gbar, use_predict=Gbar_use_predict)
    lGAW = compute_lGbar_hatAW(GW_pred, A, threshold=lGAW_threshold)

    # Compute the fluctuated Qbar using the logit and expit transformations
    QAW_pred = choose_Qbar_format(AW, Qbar, algorithm=Qbar_algorithm, use_predict=Qbar_use_predict)
    out = expit(logit(QAW_pred) + h * (2 * A - 1) / lGAW)

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

# 10.2.4 Targeted roaming of a fluctuation
# Notes
# compute_lGbar_hatAW() and get_risk_per_h() are adapted into Python grammar.
# The most accurate translation from R code should be find under snippets


def get_risk_per_h(candidates, obs_df, Qbar, Gbar,
                   Qbar_algorithm=None,
                   Qbar_use_predict=False,
                   Gbar_use_predict=False,
                   lGAW_threshold=0.05): 
    """
    Compute the risk for each candidate value given lGAW and QAW.

    Args:
        candidates (np.ndarray): Array of candidate values for h.
        lGAW (np.ndarray): Weighted probability of A given W.
        QAW (np.ndarray): Estimated conditional expectation of Y given A and W.

    Returns:
        np.ndarray: Risk values for each candidate.
    """    
    Y = obs_df["Y"].values

    risk = []
    for h in candidates:
        fluctuation_out = fluctuate(obs_df[["A", "W"]], Qbar, Gbar, h, 
                                    Qbar_algorithm=Qbar_algorithm,
                                    Qbar_use_predict=Qbar_use_predict,
                                    Gbar_use_predict=Gbar_use_predict,
                                    lGAW_threshold=lGAW_threshold)
        risk.append(-np.mean(Y * np.log(fluctuation_out) + (1 - Y) * np.log(1 - fluctuation_out)))
    risk = np.array(risk)
    
    # Find the index of the minimum risk
    idx_min = np.argmin(risk)
    
    # Find the index of the candidate closest to zero
    idx_zero = np.argmin(np.abs(candidates))
    
    # Define labels for plotting
    labels = [
        r"$R_n(\bar{Q}_{n, h_n}^o)$",
        r"$R_n(\bar{Q}_{n, 0}^o)$"
    ]

    # Create a DataFrame for risk and candidates
    risk_df = pd.DataFrame({"value": risk, "h": candidates})

    # Filter rows based on the condition
    filtered_risk_df = risk_df[
        abs(risk_df["h"] - candidates[idx_min]) <= abs(candidates[idx_min])
    ]

    return risk, filtered_risk_df, idx_min, idx_zero, labels

def plot_risk(risk, filtered_risk_df, candidates, idx_min, idx_zero, labels):
    # Plot using matplotlib
    plt.figure(figsize=(12, 8))

    # Scatter plot for risk values
    plt.scatter(filtered_risk_df["h"], filtered_risk_df["value"], color="#CC6666", label="Risk Points")

    # Vertical lines at h = 0 and h = candidates[idx_min]
    plt.axvline(x=0, color="black", linestyle="--", label="h = 0")
    plt.axvline(x=candidates[idx_min], color="black", linestyle="--", label=f"h = {candidates[idx_min]:.4f}")

    # Horizontal lines at risk[idx_min] and risk[idx_zero]
    plt.axhline(y=risk[idx_min], color="blue", linestyle="--", label=f"Risk at h_min = {risk[idx_min]:.4f}")
    plt.axhline(y=risk[idx_zero], color="green", linestyle="--", label=f"Risk at h_zero = {risk[idx_zero]:.4f}")

    # Customize y-axis
    plt.yscale("log")  # Apply log transformation to the y-axis
    plt.ylabel(r"Empirical logistic risk, $R_n(\bar{Q}_{n,h}^o)$")

    # Add secondary y-axis with custom labels
    # def secondary_y_axis_formatter(value, pos):
    #     if value == risk[idx_min]:
    #         return labels[0]
    #     elif value == risk[idx_zero]:
    #         return labels[1]
    #     return ""

    sec_ax = plt.gca().secondary_yaxis("right")
    sec_ax.set_ticks([risk[idx_min], risk[idx_zero]])
    sec_ax.set_yticklabels(labels)

    # Add labels and legend
    plt.xlabel("h")
    plt.legend()
    plt.title("Empirical Logistic Risk Visualization")

    # Show the plot
    plt.tight_layout()
    plt.show()

# 10.2.6 Alternative fluctuation
def fluctuate_a(AW, Qbar, h,
                Qbar_algorithm=None,
                Qbar_use_predict=False):
    # Extract A and W from the input DataFrame AW
    # A = AW["A"].values
    # W = AW["W"].values

    Qbar_pred = choose_Qbar_format(AW, Qbar, algorithm=Qbar_algorithm, use_predict=Qbar_use_predict)
    out = expit(logit(Qbar_pred) + h)
    return out

def fluctuate_alt(AW, Qbar, h0, h1,
                  Qbar_algorithm=None,
                  Qbar_use_predict=False):
    """    Create an alternative fluctuation function based on Qbar, h0, and h1. """
    
    A = AW["A"].values
    AW_0 = pd.DataFrame({"A": 0, "W": AW["W"]})
    AW_1 = pd.DataFrame({"A": 1, "W": AW["W"]})

    out = A * fluctuate_a(AW_1, Qbar, h1, Qbar_algorithm, Qbar_use_predict) + \
            (1 - A) * fluctuate_a(AW_0, Qbar, h0, Qbar_algorithm, Qbar_use_predict)
    
    return out

def prepare_df_alt(Qbar_truth, Qbar_hat_trees, algorithm, 
                   h0=[-1, 1, 1.5], h1=[1.5, 1, -1]):

    # Generate w values
    w = np.linspace(0, 1, int(1e3))

    # Create a DataFrame for w
    df = pd.DataFrame({"w": w})

    # Add columns for truth and trees fluctuations
    # first set of h0 and h1
    df["truth_h0={}_h1={}_1".format(h0[0], h1[0])] = fluctuate_alt(pd.DataFrame({"A": 1, "W": w}), Qbar_truth, h0=h0[0], h1=h1[0])
    df["truth_h0={}_h1={}_0".format(h0[0], h1[0])] = fluctuate_alt(pd.DataFrame({"A": 0, "W": w}), Qbar_truth, h0=h0[0], h1=h1[0])
    df["trees_h0={}_h1={}_1".format(h0[0], h1[0])] = fluctuate_alt(pd.DataFrame({"A": 1, "W": w}), Qbar_hat_trees, h0=h0[0], h1=h1[0],
                                                                   Qbar_algorithm=algorithm,
                                                                   Qbar_use_predict=True)
    df["trees_h0={}_h1={}_0".format(h0[0], h1[0])] = fluctuate_alt(pd.DataFrame({"A": 0, "W": w}), Qbar_hat_trees, h0=h0[0], h1=h1[0], 
                                                                   Qbar_algorithm=algorithm,
                                                                   Qbar_use_predict=True)
    
    
    # second set of h0 and h1
    df["truth_h0={}_h1={}_1".format(h0[1], h1[1])] = fluctuate_alt(pd.DataFrame({"A": 1, "W": w}), Qbar_truth, h0=h0[1], h1=h1[1])
    df["truth_h0={}_h1={}_0".format(h0[1], h1[1])] = fluctuate_alt(pd.DataFrame({"A": 0, "W": w}), Qbar_truth, h0=h0[1], h1=h1[1])
    df["trees_h0={}_h1={}_1".format(h0[1], h1[1])] = fluctuate_alt(pd.DataFrame({"A": 1, "W": w}), Qbar_hat_trees, h0=h0[1], h1=h1[1],
                                                                   Qbar_algorithm=algorithm,
                                                                   Qbar_use_predict=True)
    df["trees_h0={}_h1={}_0".format(h0[1], h1[1])] = fluctuate_alt(pd.DataFrame({"A": 0, "W": w}), Qbar_hat_trees, h0=h0[1], h1=h1[1],
                                                                   Qbar_algorithm=algorithm,
                                                                   Qbar_use_predict=True)
    
    # third set of h0 and h1
    df["truth_h0={}_h1={}_1".format(h0[2], h1[2])] = fluctuate_alt(pd.DataFrame({"A": 1, "W": w}), Qbar_truth, h0=h0[2], h1=h1[2])
    df["truth_h0={}_h1={}_0".format(h0[2], h1[2])] = fluctuate_alt(pd.DataFrame({"A": 0, "W": w}), Qbar_truth, h0=h0[2], h1=h1[2])
    df["trees_h0={}_h1={}_1".format(h0[2], h1[2])] = fluctuate_alt(pd.DataFrame({"A": 1, "W": w}), Qbar_hat_trees, h0=h0[2], h1=h1[2],
                                                                   Qbar_algorithm=algorithm,
                                                                   Qbar_use_predict=True)
    df["trees_h0={}_h1={}_0".format(h0[2], h1[2])] = fluctuate_alt(pd.DataFrame({"A": 0, "W": w}), Qbar_hat_trees, h0=h0[2], h1=h1[2],
                                                                   Qbar_algorithm=algorithm,
                                                                   Qbar_use_predict=True)
    
    # return df
    # Pivot the DataFrame to long format
    df_long = df.melt(id_vars=["w"], var_name="f", value_name="value")

    # Extract components from the column names
    df_long[["f", "h0", "h1", "a"]] = df_long["f"].str.extract(r"([^_]+)_h0=(-?\d+\.?\d*)_h1=(-?\d+\.?\d*)_([01]+)")

    # Convert h0 and h1 to numeric types for further processing
    df_long["h0"] = pd.to_numeric(df_long["h0"])
    df_long["h1"] = pd.to_numeric(df_long["h1"])

    # Modify columns for visualization
    df_long["f"] = df_long["f"].replace({"truth": "Q_0", "trees": "Q_n"})
    df_long["a"] = "a=" + df_long["a"]
    df_long["fh"] = "(" + df_long["f"] + ", h0=" + df_long["h0"].astype(str) + ", h1=" + df_long["h1"].astype(str) + ")"

    return df_long

def plot_fluctuation_alt(df_long):
    """
    Plot fluctuations using the output from prepare_df_alt(), treating h0 and h1 as a tuple for hue.

    Args:
        df_long (pd.DataFrame): DataFrame containing columns extracted and prepared by prepare_df_alt().

    Returns:
        None: Displays the plot.
    """
    # Define a custom color palette for "hue"
    custom_palette = {(-1.0, 1.5): "blue", (1.0, 1.0): "green", (1.5, -1.0): "red"}  # Assign colors to hue values

    # Create a new column for the tuple (h0, h1)
    df_long["h_tuple"] = list(zip(df_long["h0"], df_long["h1"]))

    # Facet by 'a'
    g = sns.FacetGrid(df_long, col="a", height=6, aspect=2)  # Adjust height and aspect for better visualization
    g.map_dataframe(
        sns.lineplot,
        x="w",
        y="value",
        hue="h_tuple",  # Use the tuple (h0, h1) for color coding
        style="f",  # Use 'f' for different line styles
        linewidth=3,
        estimator=None,
        palette=custom_palette  # Apply the custom palette
    )
    g.set_titles("{col_name}")  # Set facet titles based on 'a'
    g.set_axis_labels("w", r"$f_{(h_0, h_1)}(a, w)$")  # Customize axis labels
    g.add_legend(title="Legend")  # Add a legend

    # Show the plot
    plt.tight_layout()
    plt.show()

# Practice Problem 10.2.6.4
def get_risk_per_h0_h1(obs_df, Qbar_hat, Qbar_algorithm, Qbar_use_predict, lGAW,
                       h0_list, h1_list, ha_of_interest=0):
    Y = obs_df["Y"].values
    A = obs_df["A"].values
    risk_ls = []
    for id, (h0, h1) in enumerate(zip(h0_list, h1_list)):
        # Compute \bar{Q}^{a})_{n, h0} and # \bar{Q}^{a})_{n, h1} for each treatment arm
        Qbar_a0 = fluctuate_a(obs_df[["A", "W"]], Qbar_hat, h0,
                              Qbar_algorithm=Qbar_algorithm,
                              Qbar_use_predict=Qbar_use_predict)

        Qbar_a1 = fluctuate_a(obs_df[["A", "W"]], Qbar_hat, h1,
                              Qbar_algorithm=Qbar_algorithm,
                              Qbar_use_predict=Qbar_use_predict)
        
        # Compute the empirical risk for each treatment arm with weights
        risk_a0 = -np.mean(Y * np.log(Qbar_a0) + (1 - Y) * np.log(1 - Qbar_a0) * ( (A==0) / lGAW ))
        risk_a1 = -np.mean(Y * np.log(Qbar_a1) + (1 - Y) * np.log(1 - Qbar_a1) * ( (A==1) / lGAW ))

        risk = risk_a0 + risk_a1
        # print(f'risk is {risk} for h0={h0}, h1={h1}, id={id}')
        risk_ls.append(risk)

    risk = np.array(risk_ls)

    # Find the index of the minimum risk
    idx_min = np.argmin(risk)

    # Find the index of the candidate closest to zero
    if ha_of_interest == 0: # which h_ are we we looking at, the other h_ is set to a fixed value e.g. 0
        idx_zero = np.argmin(np.abs(h0_list))
    else:
        idx_zero = np.argmin(np.abs(h1_list))
    
    # Define labels for plotting
    if ha_of_interest == 0:
        labels = [
            r"$R_n(\bar{Q}_{n, h_0}^o)$",
            r"$R_n(\bar{Q}_{n, 0}^o)$"
        ]
    else:
        labels = [
            r"$R_n(\bar{Q}_{n, h_1}^o)$",
            r"$R_n(\bar{Q}_{n, 0}^o)$"
        ]
    
    # Create a DataFrame for risk and candidates
    candidates = h0_list if ha_of_interest == 0 else h1_list
    risk_df = pd.DataFrame({"value": risk, "h": candidates})

    # Filter rows based on the condition
    filtered_risk_df = risk_df[
        abs(risk_df["h"] - candidates[idx_min]) <= abs(candidates[idx_min])
    ]

    return risk, filtered_risk_df, idx_min, idx_zero, labels


# 10.4.1 A first numerical application
# Preliminary calculations
def preliminary(obs, Qbar_hat, Gbar_hat, 
                Qbar_algorithm=None,
                Qbar_use_predict=False,
                Gbar_use_predict=False,
                threshold=5e-2):
    if not all(col in obs.columns for col in ["W", "A", "Y"]):
        raise ValueError("Argument 'dat' must contain columns 'W', 'A', and 'Y'.")
    
    QAW = choose_Qbar_format(obs[["A", "W"]], Qbar_hat, 
                             algorithm=Qbar_algorithm, use_predict=Qbar_use_predict)
    QoneW = choose_Qbar_format(pd.DataFrame({"A": 1, "W": obs["W"]}), Qbar_hat, 
                               algorithm=Qbar_algorithm, use_predict=Qbar_use_predict)
    QzeroW = choose_Qbar_format(pd.DataFrame({"A": 0, "W": obs["W"]}), Qbar_hat,
                                 algorithm=Qbar_algorithm, use_predict=Qbar_use_predict)
    Gbar_out = choose_Gbar_format(obs["W"].values, Gbar_hat, use_predict=Gbar_use_predict)
    GW = np.clip(Gbar_out, threshold, 1 - threshold)
    HW = obs["A"] / GW - (1 - obs["A"]) / (1 - GW)
    
    return pd.DataFrame({"QAW": QAW, "QoneW": QoneW, "QzeroW": QzeroW, "GW": GW, "HW": HW})

def apply_targeting_step(dat, Qbar_hat, Gbar_hat, 
                         Qbar_algorithm=None,
                         Qbar_use_predict=False,
                         Gbar_use_predict=False,
                         threshold=5e-2, epsilon=None):
    """
    Apply the targeting step to adjust Qbar using the targeting parameter epsilon.

    Args:
        dat (pd.DataFrame): DataFrame containing columns 'W', 'A', and 'Y'.
        Gbar (callable): Function to estimate G(W).
        Qbar (callable): Function to estimate Q(A, W).
        threshold (float): Threshold value for probabilities (default is 0.05).
        epsilon (float, optional): Targeting parameter. If None, it is estimated.

    Returns:
        pd.DataFrame: DataFrame containing psi_n, sig_n, and crit_n.
    """
    # Validate threshold
    if not (1e-3 <= threshold <= 1 - 1e-3):
        raise ValueError("Threshold must be between 0.001 and 0.999.")

    tib = preliminary(dat, Qbar_hat, Gbar_hat, 
                      Qbar_algorithm=Qbar_algorithm, Qbar_use_predict=Qbar_use_predict, 
                      Gbar_use_predict=Gbar_use_predict, threshold=threshold)

    # Estimate epsilon if not provided
    if epsilon is None:
        glm_model = smf.glm(formula="Y ~ HW - 1", data=pd.DataFrame({"Y":dat["Y"], "HW":tib["HW"]}),
                            family=sm.families.Binomial(),
                            offset=logit(tib["QAW"])
                        )
        fit = glm_model.fit()
        epsilon = fit.params[0]

    # Update Qbar with epsilon
    QoneW_epsilon = expit(logit(tib["QoneW"]) + epsilon / tib["GW"])
    QzeroW_epsilon = expit(logit(tib["QzeroW"]) - epsilon / (1 - tib["GW"]))
    QAW_epsilon = dat["A"] * QoneW_epsilon + (1 - dat["A"]) * QzeroW_epsilon

    # Compute psi_n, sig_n, and crit_n
    psi_n = np.mean(QoneW_epsilon - QzeroW_epsilon)
    eic_dat = (dat["Y"] - QAW_epsilon) * tib["HW"] + QoneW_epsilon - QzeroW_epsilon - psi_n
    sig_n = np.std(eic_dat) / np.sqrt(len(dat))
    crit_n = np.mean(eic_dat)

    return pd.DataFrame({"psi_n": [psi_n], "sig_n": [sig_n], "crit_n": [crit_n]})

# Practice Problem 10.4.2.2
def get_psi_and_crit_by_epsilon(dat, Qbar_hat, Gbar_hat, 
                                Qbar_algorithm=None,
                                Qbar_use_predict=False,
                                Gbar_use_predict=False,
                                threshold=5e-2, epsilon=None):
    psi_n = []
    crit_n = []
    for h in epsilon:
        result = apply_targeting_step(dat, Qbar_hat, Gbar_hat, 
                                      Qbar_algorithm=Qbar_algorithm,
                                      Qbar_use_predict=Qbar_use_predict,
                                      Gbar_use_predict=Gbar_use_predict,
                                      threshold=threshold, epsilon=h)
        psi_n.append(result["psi_n"].values[0])
        crit_n.append(result["crit_n"].values[0])
    psi_Qbar_epsilon = pd.DataFrame({"psi_n": psi_n, "crit_n": crit_n})
    idx_Qbar = np.argmin(np.abs(psi_Qbar_epsilon["crit_n"]))

    return psi_Qbar_epsilon, idx_Qbar 
    


def prepare_psi_crit_data(dat, Qbar_hat_dict, Gbar_hat_dict, 
                            Qbar_algorithm_dict=None,
                            Qbar_use_predict_dict=False,
                            Gbar_use_predict_dict=False,
                            threshold_dict=5e-2):
    # Define epsilon range
    epsilon = np.linspace(-1e-2, 1e-2, int(1e2))

    # Compute psi_trees_epsilon
    psi_trees_epsilon, idx_trees = get_psi_and_crit_by_epsilon(dat, 
                                                              Qbar_hat_dict["trees"], 
                                                              Gbar_hat_dict["trees"],
                                                              Qbar_algorithm=Qbar_algorithm_dict["trees"],
                                                              Qbar_use_predict=Qbar_use_predict_dict["trees"],
                                                              Gbar_use_predict=Gbar_use_predict_dict["trees"],
                                                              threshold=threshold_dict["trees"],
                                                              epsilon=epsilon)

    # Compute psi_kknn_epsilon
    psi_kknn_epsilon, idx_kknn = get_psi_and_crit_by_epsilon(dat,
                                                            Qbar_hat_dict["kknn"], 
                                                            Gbar_hat_dict["kknn"],
                                                            Qbar_algorithm=Qbar_algorithm_dict["kknn"],
                                                            Qbar_use_predict=Qbar_use_predict_dict["kknn"],
                                                            Gbar_use_predict=Gbar_use_predict_dict["kknn"],
                                                            threshold=threshold_dict["kknn"],
                                                            epsilon=epsilon)


    # Combine results into a single DataFrame
    results_df = pd.concat([psi_trees_epsilon, psi_kknn_epsilon], axis=0).reset_index(drop=True)
    
    # Add a 'type' column to indicate the source of each row
    results_df["type"] = np.repeat(["trees", "kknn"], len(epsilon))
    
    return results_df, idx_trees, idx_kknn, psi_trees_epsilon, psi_kknn_epsilon


def plot_psi_crit_data(results_df, idx_trees, idx_kknn, psi_trees_epsilon, psi_kknn_epsilon):
    # Plot using seaborn
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=results_df, x="crit_n", y="psi_n", hue="type", palette="Set1")

    # Add vertical and horizontal lines
    plt.axvline(x=0, color="black", linestyle="--", label="crit_n = 0")
    plt.axhline(y=psi_trees_epsilon["psi_n"][idx_trees], color="blue", linestyle="--", label="psi_trees_at_crit_min")
    plt.axhline(y=psi_kknn_epsilon["psi_n"][idx_kknn], color="green", linestyle="--", label="psi_kknn_at_crit_min")

    # Customize labels
    plt.xlabel(r"$P_n D^*(P_{n,h}^o)$")
    plt.ylabel(r"$\Psi(P_{n,h}^o)$")
    plt.legend(title="Type")
    plt.title("Empirical Investigation of Targeting Step")

    # Show the plot
    plt.tight_layout()
    plt.show()

# 10.4.3 Empirical investigation
def update_learned_features(learned_features, algorithm_d=None, algorithm_e=None,
                            Gbar_algorithm=None):
    # Assuming learned_features_fixed_sample_size is a pandas DataFrame
    for key, entry in learned_features.items():
        entry["Qbar_hat_d"] = estimate_Qbar(entry["obs"], algorithm=algorithm_d)
        entry["Qbar_hat_e"] = estimate_Qbar(entry["obs"], algorithm=algorithm_e)
        Gbar_hat = estimate_Gbar(entry["obs"], algorithm=Gbar_algorithm)
        entry["est_d"] = apply_targeting_step(entry["obs"], entry["Qbar_hat_d"], 
                                                Gbar_hat=Gbar_hat,
                                                Qbar_algorithm=algorithm_d,
                                                Qbar_use_predict=True,
                                                Gbar_use_predict=True,
                                                threshold=0.05, epsilon=None)
        entry["est_e"] = apply_targeting_step(entry["obs"], entry["Qbar_hat_e"],
                                                Gbar_hat=Gbar_hat,
                                                Qbar_algorithm=algorithm_e,
                                                Qbar_use_predict=True,
                                                Gbar_use_predict=True,
                                                threshold=0.05, epsilon=None)
    return learned_features

def plot_estimation_bias_de(psi_hat_de_df, bias_de):    
    # Generate the standard normal distribution
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)
    
    # Create the plot
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))

    for id_norm, ((if_norm_, norm_group), (if_norm_bias, norm_bias)) in enumerate(zip(psi_hat_de_df.groupby("auto_renormalization"), 
                                                        bias_de.groupby("auto_renormalization"))): 

        # Plot the standard normal distribution
        sns.lineplot(x=x, y=y, ax=ax[id_norm], linestyle='-', color="grey", alpha=0.5, label="Standard Normal")
        
        # Plot density for clt
        color_index = {}
        for id_type, (type_, group) in enumerate(norm_group.groupby("type")): 
            sns.kdeplot(group["clt"], 
                        fill=True, 
                        alpha=0.1, 
                        label=f"{type_}_targeted", 
                        ax=ax[id_norm])
            color_index[type_] = id_type
            
        
        for _, row in norm_bias.iterrows():
            color_id = color_index[row["type"]]
            ax[id_norm].axvline(x=row["bias"], 
                                color=sns.color_palette()[color_id], 
                                linestyle="--", 
                                linewidth=1.5, 
                                alpha=0.5, 
                                label=f"Bias (Type {row['type']})")
        
        ax[id_norm].set_title("auto-renormalization:{}".format(if_norm_))
        if not if_norm_:
            ax[id_norm].set_xlim(-4, 4)
            ax[id_norm].set_ylim(0, 0.5)
            ax[id_norm].legend(title="Type", loc='upper left')
        else:
            ax[id_norm].set_xlim(-4, 4)
            ax[id_norm].set_ylim(0, 0.5)
            ax[id_norm].legend(title="Type", loc='upper right')
        
    fig.suptitle(r"$\sqrt{n/v_n^{d, e}} * (\psi_n^{d, e} - \psi_0)$", fontsize=12)
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

    # 10.2.4 Targeted roaming of a fluctuation
    # Generate candidates array
    candidates = np.linspace(-0.01, 0.01, int(1e4))  # Equivalent to seq(-0.01, 0.01, length.out = 1e4)
    
    # Subset the obs DataFrame for W, A, and Y
    W = obs_df.loc[1:int(1e3), "W"]  # Select the first 1000 rows of column "W"
    A = obs_df.loc[1:int(1e3), "A"]  # Select the first 1000 rows of column "A"
    Y = obs_df.loc[1:int(1e3), "Y"]  # Select the first 1000 rows of column "Y"

    working_model_G_one = WorkingModelGOne()
    Gbar_hat = estimate_Gbar(dat=obs[:int(1e3)], algorithm=working_model_G_one)
    
    risk, filtered_risk_df, idx_min, idx_zero, labels = get_risk_per_h(candidates, pd.DataFrame({"A": A, "W": W, "Y": Y}),
                                                                       Qbar_hat_trees, Gbar_hat,
                                                                       Qbar_algorithm=boosting_tree_algo,
                                                                       Qbar_use_predict=True,
                                                                       Gbar_use_predict=True,
                                                                       lGAW_threshold=0.05)
    plot_risk(risk, filtered_risk_df, candidates, idx_min, idx_zero, labels)

    # 10.2.6 Alternative fluctuation
    # Practice Problem 10.2.6.1
    """The alternative fluctuation model differs from the canonical one in that it applies separate one-dimensional fluctuations 
    to the treatment-specific regressions \bar{Q}(0, W) and \bar{Q}(1, W), leading to a more flexible two-parameter model. 
    This allows each treatment group to be updated independently, which may lead to better empirical risk minimization. 
    The loss function is also modified: rather than a unified logistic regression, it applies weighted losses separately 
    for each treatment arm, weighted by the inverse of the known propensity score \ell \bar{G}(A, W). 
    In contrast, the earlier fluctuation model used a global score-based update to adjust all predictions jointly."""

    # Practice Problem 10.2.6.2
    h0, h1 = -1, 1
    Qbar_alt = fluctuate_alt(pd.DataFrame({"A": A, "W": W}), Qbar_hat_trees, h0, h1,
                             Qbar_algorithm=boosting_tree_algo,
                             Qbar_use_predict=True)
    
    df_fluct_alt = prepare_df_alt(experiment.Qbar, Qbar_hat_trees, boosting_tree_algo, h0=[-1, 1, 1.5], h1=[1.5, 1, -1])
    plot_fluctuation_alt(df_fluct_alt)

    # Practice Problem 10.2.6.3 
    '''
    We are given the empirical risk function:
    (h_0, h_1) \mapsto \sum_{a=0,1} \mathbb{P}n\left( L^a{Y, \bar{G}} \left( \bar{Q}^{a}_{h_a} \right)(O) \right)

    This expands to:
    \text{EmpiricalRisk}(h_0, h_1) = \mathbb{P}n\left( L^0{Y, \bar{G}}(\bar{Q}^0_{h_0})(O) \right) + \mathbb{P}n\left( L^1{Y, \bar{G}}(\bar{Q}^1_{h_1})(O) \right)

    Let us define:
        •	R_0(h_0) := \mathbb{P}n\left( L^0{Y, \bar{G}}(\bar{Q}^0_{h_0})(O) \right),
        •	R_1(h_1) := \mathbb{P}n\left( L^1{Y, \bar{G}}(\bar{Q}^1_{h_1})(O) \right),

    Then the empirical risk becomes:
    R(h_0, h_1) = R_0(h_0) + R_1(h_1)

    This shows:
        •	R_0(h_0) depends only on h_0,
        •	R_1(h_1) depends only on h_1,
        •	There is no interaction between h_0 and h_1 in the risk function.

    ⸻

    Therefore, we can optimize independently:

    We want to solve:
    (h_0^\star, h_1^\star) = \arg\min_{(h_0, h_1) \in \mathbb{R}^2} \left[ R_0(h_0) + R_1(h_1) \right]

    But since the two terms are independent, this separates cleanly into:
    h_0^\star = \arg\min_{h_0 \in \mathbb{R}} R_0(h_0), \quad
    h_1^\star = \arg\min_{h_1 \in \mathbb{R}} R_1(h_1)

    The optimization problem over \mathbb{R}^2 decomposes into two independent scalar optimization problems because:
	•	The total risk is an additive sum over a = 0, 1,
	•	Each summand depends only on a single parameter, h_0 or h_1, through the fluctuation of \bar{Q}^a,
	•	There is no coupling between h_0 and h_1 anywhere in the objective function.

    This is a standard property of separable objective functions in optimization: they can be minimized coordinate-wise.
    '''

    # Practice Problem 10.2.6.4
    # look at h0
    h0_list = np.linspace(-0.1, 0.1, int(1e4))  # Equivalent to seq(-0.01, 0.01, length.out = 1e4)
    h1_list = np.zeros_like(h0_list)  # Set h1 to zero for this problem

    GW = choose_Gbar_format(W, experiment.Gbar, use_predict=False)
    lGAW = compute_lGbar_hatAW(GW, A, threshold=0.05)
    risk, filtered_risk_df, idx_min, idx_zero, labels = get_risk_per_h0_h1(pd.DataFrame({"A": A, "W": W, "Y":Y}), 
                                                                           Qbar_hat_trees, boosting_tree_algo, 
                                                                           Qbar_use_predict=True,
                                                                           lGAW=lGAW,
                                                                           h0_list=h0_list,
                                                                           h1_list=h1_list,
                                                                           ha_of_interest=0)

    plot_risk(risk, filtered_risk_df, h0_list, idx_min, idx_zero, labels)
    # look at h1
    h1_list = np.linspace(-0.1, 0.1, int(1e4))
    h0_list = np.zeros_like(h1_list)  # Set h0 to zero for this problem

    GW = choose_Gbar_format(W, experiment.Gbar, use_predict=False)
    lGAW = compute_lGbar_hatAW(GW, A, threshold=0.05)
    risk, filtered_risk_df, idx_min, idx_zero, labels = get_risk_per_h0_h1(pd.DataFrame({"A": A, "W": W, "Y":Y}), 
                                                                           Qbar_hat_trees, boosting_tree_algo, 
                                                                           Qbar_use_predict=True,
                                                                           lGAW=lGAW,
                                                                           h0_list=h0_list,
                                                                           h1_list=h1_list,
                                                                           ha_of_interest=1)

    plot_risk(risk, filtered_risk_df, h0_list, idx_min, idx_zero, labels)

    # Practice Problem 10.2.6.5
    """
    TL;DR: 1) The inverse propensity weights does not change the optimum of the loss function. 
    2) Though the loss function is expanded to two parts, the optimal still sits at the same point as before.

    Recall:
    L^a_{Y, \bar{G}}(f)(O) = \frac{\mathbf{1}\{A = a\}}{\ell \bar{G}(A, W)} L_Y(f)(O)
    and
    L_Y(f)(O) = -Y \log f(W) - (1 - Y) \log (1 - f(W))
    is the log-loss for binary regression.

    So,
    \mathbb{E}{P_0} \left[ L^a{Y, \bar{G}}(f)(O) \right] = \mathbb{E}_{P_0} \left[ \frac{\mathbf{1}\{A = a\}}{\ell \bar{G}(a, W)} \cdot L_Y(f)(O) \right]

    Now apply inverse probability weighting:
    = \mathbb{E}{P_0} \left[ \mathbb{E}{P_0} \left[ \frac{\mathbf{1}\{A = a\}}{\ell \bar{G}(a, W)} \cdot L_Y(f)(O) \Big| W \right] \right] = \mathbb{E}{P_0} \left[ \mathbb{E}{P_0} \left[ L_Y(f)(O) \mid A = a, W \right] \right]

    This means we are minimizing:
    \mathbb{E}{P_0} \left[ \mathbb{E}{P_0} \left[ -Y \log f(W) - (1 - Y) \log (1 - f(W)) \mid A = a, W \right] \right]

    This is exactly minimized when:
    f(W) = \mathbb{E}[Y \mid A = a, W] = \bar{Q}^a_0(W)

    """

    # Practice Problem 10.2.6.6
    """
    Since:
	•	\bar{Q}_0(a, w) = \bar{Q}^a_0(w), and
	•	Each L^a_{Y, \bar{G}} is minimized at f = \bar{Q}^a_0 individually,

    Then the summed risk:
    \mathbb{E}{P_0} \left[ \sum_{a=0}^1 L^a_{Y, \bar{G}}(f^a)(O) \right]
    is minimized when f^a = \bar{Q}_0^a, i.e., when we plug in:
    \bar{Q}(a, w) = \bar{Q}_0(a, w)

    Therefore, the total risk using:
    L_{Y, \bar{G}}^{\text{sum}}(f) = \sum_{a=0}^1 L^a_{Y, \bar{G}}(f)(O)
    is valid for learning the full function \bar{Q}_0(a, w).

    """

    # Practice Problem 10.2.6.7
    """
    Equation (10.8) is the original empirical risk minimization (ERM) problem in TMLE:
    \arg\min_{\varepsilon \in \mathbb{R}} \mathbb{E}{P_n}\left[L_Y\left( \bar{Q}\varepsilon \right)\right]
    where \bar{Q}_\varepsilon is a fluctuated version of \bar{Q}_n, e.g. by a logistic tilting model.

    In your alternative fluctuation, you write:
    \bar{Q}^{\text{alt}}_{h_0, h_1}(a, w) = a \cdot \bar{Q}^1_{h_1}(w) + (1 - a) \cdot \bar{Q}^0_{h_0}(w)
    with:
    \[
    \bar{Q}^a_{h}(w) = \expit\left( \text{logit}(\bar{Q}^a(w)) + h \right)
    \]

    The empirical risk function becomes:
    (h_0, h_1) \mapsto \sum_{a=0}^1 \mathbb{E}{P_n} \left[ L^a{Y, \bar{G}} \left( \bar{Q}^a_{h_a} \right)(O) \right]

    ⸻

    Mutatis mutandis argument:

    Just like in (10.8), where we fluctuate \bar{Q} via a single parameter \varepsilon, here we fluctuate two components \bar{Q}^0 and \bar{Q}^1 separately by h_0 and h_1.
    We minimize empirical risk of a loss function (log-loss) weighted by inverse propensity scores.

    Thus:
        •	The fluctuation model is analogous,
        •	The optimization step mirrors that of TMLE,
        •	The justification follows from the same logic.

    Therefore, the alternative fluctuation and summed loss function form a valid TMLE submodel under the same principles as the original formulation.
    """

    # 10.4 Empirical investigation
    # 10.4.1 A first numerical application
    ## code adapted from  section 9
    QW_hat = estimate_QW(obs_df[:int(1e3)])
    kknn_algo = KknnAlgo()
    Qbar_hat_kknn = estimate_Qbar(dat=obs_df[:int(1e3)], algorithm=kknn_algo)
    psin_kknn = compute_gcomp(QW_hat, Qbar_hat_kknn, algorithm=kknn_algo, use_predict=True)
    print(psin_kknn)
    
    working_model_G_one = WorkingModelGOne()
    Gbar_hat = estimate_Gbar(dat=obs[:int(1e3)], algorithm=working_model_G_one)

    psin_kknn_os = apply_one_step_correction(obs_df[:int(1e3)], Gbar_hat, Qbar_hat_kknn, psin_kknn["psi_n"][0],
                                              Gbar_use_predict=True, Qbar_use_predict=True,
                                              Gbar_algorithm=working_model_G_one, Qbar_algorithm=kknn_algo)
    print(psin_kknn_os)

    psin_kknn_tmle = apply_targeting_step(obs_df[:int(1e3)], Qbar_hat_kknn, Gbar_hat, 
                                          Qbar_algorithm=kknn_algo, Qbar_use_predict=True,
                                          Gbar_use_predict=True, threshold=5e-2, epsilon=None)
    print(psin_kknn_tmle)

    # 10.4.2 A computational exploration
    # Practice Problem 10.4.2.1
    """Epsilon it the optimal fluctuation parameter"""

    # Practice Problem 10.4.2.2
    results_df, idx_trees, idx_kknn, psi_trees_epsilon, psi_kknn_epsilon = prepare_psi_crit_data(obs_df[:int(1e3)], 
                                                                                                 Qbar_hat_dict={"trees": Qbar_hat_trees, "kknn": Qbar_hat_kknn},
                                                                                                 Gbar_hat_dict={"trees": Gbar_hat, "kknn": Gbar_hat},
                                                                                                 Qbar_algorithm_dict={"trees": boosting_tree_algo, "kknn": kknn_algo},
                                                                                                 Qbar_use_predict_dict={"trees": True, "kknn": True},
                                                                                                 Gbar_use_predict_dict={"trees": True, "kknn": True},
                                                                                                 threshold_dict={"trees": 5e-2, "kknn": 5e-2})
    plot_psi_crit_data(results_df, idx_trees, idx_kknn, psi_trees_epsilon, psi_kknn_epsilon)

    # Practice Problem 10.4.2.3
    """The colored curves in Figure 10.3 look like line segments because the targeting step induces a one-dimensional fluctuation path in the model \bar{Q}_{n,h}, and both:
	•	the parameter \Psi(P_{n,h}), and
	•	the influence curve average P_n D^*(P_{n,h})
    are (approximately) linear in h. This results in a linear trajectory in the 2D plot, forming straight segments for each method (trees, kknn)."""

    # 10.4.3 Empirical investigation
    iter = 1000
    learned_features_fixed_sample_size = get_learned_features(obs_df, iter, working_model_G_one)
    updated_features = update_learned_features(learned_features_fixed_sample_size, 
                                               algorithm_d=kknn_algo, algorithm_e=boosting_tree_algo,
                                               Gbar_algorithm=working_model_G_one)
    psi_hat_de_df, bias_de = get_psi_hat_de(updated_features, psi_zero=evaluate_psi(experiment))
    print(bias_de)
    plot_estimation_bias_de(psi_hat_de_df, bias_de)




# Snippet
# Some code is translated from R and renderes redundant in python grammar, hence, we put them here for reference.

# 10.2.4 Targeted roaming of a fluctuation
# def compute_lGbar_hatAW(A, W, Gbar_hat, threshold=0.05):
#     """
#     Compute lGbar_hatAW with thresholding.

#     Args:
#         A (np.ndarray or pd.Series): Binary treatment indicator (0 or 1).
#         W (np.ndarray or pd.Series): Covariates.
#         Gbar_hat (callable): A model object with a `predict` method to estimate Gbar.
#         threshold (float): Threshold value for predictions (default is 0.05).

#     Returns:
#         np.ndarray: Thresholded predictions for lGAW.
#     """
#     # Ensure threshold is within valid range
#     if not (0 <= threshold <= 0.5):
#         raise ValueError("Threshold must be between 0 and 0.5.")

#     # Create a DataFrame for A and W
#     dat = pd.DataFrame({"A": A, "W": W})

#     # Predict GW using Gbar_hat
#     GW = Gbar_hat.predict(dat)

#     # Handle matrix-like predictions
#     if isinstance(GW, np.ndarray) and GW.ndim == 2:
#         if GW.shape[1] == 2:
#             GW = GW[:, 1]  # Use the second column
#         else:
#             raise ValueError("Object 'GW' is neither a vector nor a two-column matrix.")
#     elif not isinstance(GW, (np.ndarray, pd.Series)):
#         raise ValueError("Object 'GW' is neither a vector nor a two-column matrix.")

#     # Compute lGAW
#     lGAW = A * GW + (1 - A) * (1 - GW)

#     # Apply thresholding
#     pred = np.minimum(1 - threshold, np.maximum(lGAW, threshold))

#     return pred

# def get_risk_per_h(candidates, lGAW, QAW, A, Y):
#     """
#     Compute the risk for each candidate value given lGAW and QAW.

#     Args:
#         candidates (np.ndarray): Array of candidate values for h.
#         lGAW (np.ndarray): Weighted probability of A given W.
#         QAW (np.ndarray): Estimated conditional expectation of Y given A and W.

#     Returns:
#         np.ndarray: Risk values for each candidate.
#     """    

#     # Compute risk for each candidate
#     risk = np.array([
#         -np.mean(Y * np.log(expit(logit(QAW) + h * (2 * A - 1) / lGAW)) +
#                  (1 - Y) * np.log(1 - expit(logit(QAW) + h * (2 * A - 1) / lGAW)))
#         for h in candidates
#     ])
    
#     # Find the index of the minimum risk
#     idx_min = np.argmin(risk)
    
#     # Find the index of the candidate closest to zero
#     idx_zero = np.argmin(np.abs(candidates))
    
#     # Define labels for plotting
#     labels = [
#         r"$R_n(\bar{Q}_{n, h_n}^o)$",
#         r"$R_n(\bar{Q}_{n, 0}^o)$"
#     ]

#     # Create a DataFrame for risk and candidates
#     risk_df = pd.DataFrame({"value": risk, "h": candidates})

#     # Filter rows based on the condition
#     filtered_risk_df = risk_df[
#         abs(risk_df["h"] - candidates[idx_min]) <= abs(candidates[idx_min])
#     ]

#     return risk, filtered_risk_df, idx_min, idx_zero, labels

## This is how you get lGAW and QAW, if you are using the code above
# Generate candidates array
# candidates = np.linspace(-0.01, 0.01, int(1e4))  # Equivalent to seq(-0.01, 0.01, length.out = 1e4)

# # Subset the obs DataFrame for W, A, and Y
# W = obs_df.loc[1:int(1e3), "W"]  # Select the first 1000 rows of column "W"
# A = obs_df.loc[1:int(1e3), "A"]  # Select the first 1000 rows of column "A"
# Y = obs_df.loc[1:int(1e3), "Y"]  # Select the first 1000 rows of column "Y"

# working_model_G_one = WorkingModelGOne()
# Gbar_hat = estimate_Gbar(dat=obs[:int(1e3)], algorithm=working_model_G_one)
# lGAW = compute_lGbar_hatAW(A, W, Gbar_hat, threshold=0.05)
# QAW = choose_Qbar_format(pd.DataFrame({"A": A, "W": W}), Qbar_hat_trees, algorithm=boosting_tree_algo, use_predict=True)
    