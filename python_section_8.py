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
from python_algorithms import WorkingModelGOne, WorkingModelGTwo, WorkingModelGThree, WorkingModelQOne, KknnAlgo, BoostingTreeAlgo, BoostingLMAlgo

# 8.2.1 Construction and computation
# compute_iptw is already defined in python_section_6.py, 
# but we need to introduce use_predict=False flag for algorithms
# Hence, reinventing the function here
def compute_iptw(dat, Gbar, use_predict=False, threshold=0.05):
    """
    Compute the IPTW estimate and its standard error.

    Parameters:
        dat (pd.DataFrame): Input data containing columns 'W', 'A', and 'Y'.
        Gbar (function): A function that computes the probability of treatment given W.
        threshold (float): Minimum value for the stabilized weights (default is 0.05).

    Returns:
        pd.DataFrame: A DataFrame with columns 'psi_n' (IPTW estimate) and 'sig_n' (standard error).
    """
    # Extract columns
    W = dat["W"]
    A = dat["A"]
    Y = dat["Y"]

    # Compute stabilized weights
    if use_predict:
        # If using a prediction model, apply it to compute Gbar
        Gbar_values = Gbar.predict(W)
        lGAW = A * Gbar_values + (1 - A) * (1 - Gbar_values)
    else:
        lGAW = A * Gbar(W) + (1 - A) * (1 - Gbar(W))
    lGAW = np.maximum(threshold, lGAW)  # Apply threshold

    # Compute IPTW estimate (psi_n)
    psi_n = np.mean(Y * (2 * A - 1) / lGAW)

    # Compute standard error (sig_n)
    sig_n = np.std(Y * (2 * A - 1) / lGAW, ddof=1) / np.sqrt(len(dat))

    # Return results as a DataFrame
    return pd.DataFrame({"psi_n": [psi_n], "sig_n": [sig_n]})

# 8.2.3 Empirical investigation
def get_learned_features(obs_df, iter, algorithm):
    # Step 1: Create learned_features_fixed_sample_size
    obs_df["id"] = (np.arange(len(obs_df)) % iter)  # Create id column
    grouped = obs_df.groupby("id").apply(lambda group: group[["W", "A", "Y"]]).reset_index(level=0)
    nested_obs = grouped.groupby("id").apply(lambda x: x.reset_index(drop=True)).reset_index(level=0, drop=True)

    # Store models and data in a dictionary
    learned_features = {}
    for id_, group in nested_obs.groupby("id"):
        Gbar_hat = estimate_Gbar(group, algorithm=algorithm)
        learned_features[id_] = {
            "obs": group,
            "Gbar_hat": Gbar_hat, 
            "est_c": compute_iptw(group, Gbar_hat, use_predict=True) 
        }

    return learned_features

def get_psi_hat_abc(learned_features_fixed_sample_size, experiment, obs_df, iter=1000):
    psi_zero = evaluate_psi(experiment)
    psi_hat_ab = compute_psi_hat_ab(obs_df, experiment, iter)
    psi_hat_ab = compute_clt(psi_hat_ab, psi_zero)

    # # Compute est_c for each entry in the dictionary
    # for key, entry in learned_features_fixed_sample_size.items():
    #     learned_features_fixed_sample_size[key]["est_c"] = compute_est_c(entry)

    # Step 3: Unnest and compute clt
    psi_hat_abc = []
    for key, entry in learned_features_fixed_sample_size.items():
        est_c = entry["est_c"]
        psi_n = est_c["psi_n"].values[0]  # Assuming est_c is a DataFrame
        sig_n = est_c["sig_n"].values[0]  # Assuming est_c is a DataFrame
        clt = (psi_n - psi_zero) / sig_n
        psi_hat_abc.append({
            "id": key,
            "psi_n": psi_n,
            "sig_n": sig_n,
            "clt": clt,
            "type": "c"
        })

    # Convert psi_hat_abc to a list of dictionaries
    psi_hat_abc.extend(psi_hat_ab.to_dict(orient="records"))  # Assuming psi_hat_ab is a DataFrame

    # Assuming psi_hat_abc is a list of dictionaries, convert it to a DataFrame
    psi_hat_abc_df = pd.DataFrame(psi_hat_abc)

    # Group by 'type' and calculate the mean of 'clt'
    bias_abc_df = psi_hat_abc_df.groupby("type", as_index=False).agg(bias=("clt", "mean"))

    # Convert the result to a dictionary if needed
    bias_abc = bias_abc_df.to_dict(orient="records")
    bias_abc = pd.DataFrame(bias_abc)

    return bias_abc, psi_hat_abc_df


def plot_estimation_bias(psi_hat_abc_df, bias_abc_df):
    # Assuming psi_hat_abc_df and bias_abc_df are already defined as DataFrames

    # Create standard normal density data
    x = np.linspace(-3, 3, int(1e3))
    y = norm.pdf(x)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot standard normal density
    plt.plot(x, y, linestyle="-", color='grey', alpha=0.5, label="Standard Normal Density")

    # Plot density of clt values for each type
    for type_, group in psi_hat_abc_df.groupby("type"):
        sns.kdeplot(group["clt"], fill=True, alpha=0.1, label=f"Type {type_}")

    # Add vertical lines for bias
    for _, row in bias_abc_df.iterrows():
        plt.axvline(x=row["bias"], color=sns.color_palette()[int(row.name)], 
                    linestyle="--", linewidth=1.5, alpha=0.5, label=f"Bias (Type {row['type']})")


    # Set x-axis limits
    plt.xlim(-3, 4)
    
    # Add labels
    plt.xlabel(r"$\sqrt{n/v_n^{(a, b, c)}} (\psi_n^{(a, b, c)} - \psi_0)$", fontsize=12)
    plt.ylabel("")
    plt.legend(title="Type")
    plt.grid(True)
    plt.title("Bias and Density Plot")
    
    # Show the plot
    plt.show()


def plot_qq(psi_hat_abc_df):
    # Create a Q-Q plot for each type
    # fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize an empty DataFrame to store all Q-Q data
    qq_data = pd.DataFrame()

    # Loop through each type and plot Q-Q plots
    for t in psi_hat_abc_df["type"].unique():
        subset = psi_hat_abc_df[psi_hat_abc_df["type"] == t]
        out = stats.probplot(subset["clt"], dist="norm", plot=None)  # Get the output without plotting

        # Unpack the theoretical and sample quantiles from stats.probplot
        theoretical_quantiles, sample_quantiles = out[0]

        # Create a DataFrame with the correct structure and add the type column
        out_df = pd.DataFrame({
            "theoretical_quantiles": theoretical_quantiles,
            "sample_quantiles": sample_quantiles,
            "type": t  # Add the type column
        })

        # Append the data to the main DataFrame
        qq_data = pd.concat([qq_data, out_df], ignore_index=True)

    # Use seaborn_qqplot to plot the combined Q-Q data
    pplot(qq_data, 
            x="theoretical_quantiles", 
            y="sample_quantiles", 
            hue="type",  # Use the type column for coloring
            kind='qq', height=4, aspect=2, display_kws={"identity": True})

# plot empirical bias and estimation
def get_estimation_bias_empirical(psi_hat_abc_df, est_type="c"): 
    # Step 1: Calculate emp_sig_n (empirical standard deviation of psi_n for type "c")
    emp_sig_n = psi_hat_abc_df[psi_hat_abc_df["type"] == est_type]["psi_n"].std()
    
    # Step 2: Create a copy of the dataset and adjust the clt column only for type "c"
    psi_hat_abc_df = psi_hat_abc_df.copy()
    psi_hat_abc_df.loc[psi_hat_abc_df["type"] == est_type, "clt"] = (
        psi_hat_abc_df.loc[psi_hat_abc_df["type"] == est_type, "clt"]
        * psi_hat_abc_df.loc[psi_hat_abc_df["type"] == est_type, "sig_n"]
        / emp_sig_n
    )    
    return psi_hat_abc_df, emp_sig_n
    
# Practice Problem 8.3.1: calculate confidence interval
def evaluate_CI(psi_hat_abc_df_emp_, alpha=0.05, est_type="b", psi_zero=None):
    """Evaluate confidence intervals for the slope estimate."""
    # Filter the DataFrame for the specified type
    tmp = psi_hat_abc_df_emp_[psi_hat_abc_df_emp_["type"] == est_type]

    # Compute confidence intervals for each psi_n
    slope_CI = tmp["psi_n"].values[:, None] + np.array([-1, 1]) * np.sqrt(alpha) * tmp["sig_n"].values[:, None] / np.sqrt(1000)

    if psi_zero is not None:
        # Check if psi_zero is within every entry of slope_CI
        in_CI = (slope_CI[:, 0] <= psi_zero) & (psi_zero <= slope_CI[:, 1])
        print(f"psi_zero is within the confidence interval: {np.sum(in_CI)} out of {len(in_CI)}")
    else:
        in_CI = None

    return slope_CI, in_CI

def perform_binom_test(in_CI, n=1000, p=0.95, alternative='greater'):
    """Perform a binomial test to evaluate the confidence intervals."""
    result = binom_test(x=np.sum(in_CI),  # number of successes
                        n=n,  # number of trials
                        p=p,  # probability of success under the null hypothesis
                        alternative=alternative)  # alternative hypothesis
    return result


# 8.4.1 Construction and computation
def compute_gcomp(QW, Qbar_hat, algorithm=None, use_predict=False, n=int(1e3)):
    """
    Compute the G-computation estimator and its standard error.

    Parameters:
        QW (pd.DataFrame): A DataFrame with columns 'value' (W values) and 'weight' (weights).
        Qbar_hat (function): A function that computes the conditional mean given A and W.
        n (int): Number of observations.
        algorithm (object, optional): An algorithm object for prediction (default is None), only used for ML algorithms.
        use_predict (bool): Whether to use the prediction model (default is False), 
                            used for both ML and non-ML algorithms. Set to False if we are using the experiment.Qbar_hat().

    Returns:
        pd.DataFrame: A DataFrame with columns 'psi_n' (estimator) and 'sig_n' (standard error).
    """
    # Validate inputs
    if not isinstance(n, int) or n < 1:
        raise ValueError("nobs must be an integer greater than or equal to 1.")
    if not set(QW.columns) == {"value", "weight"}:
        raise ValueError("QW must have columns 'value' and 'weight'.")

    # Extract W values
    W = QW["value"].values

    # Compute the difference in conditional means
    if use_predict:# If using a prediction model, apply it to compute Qbar_hat     
        stratify = Qbar_hat.attrs.get("stratify", False)  # Check if stratification is required
        if not stratify:
            fit = Qbar_hat.loc[Qbar_hat["a"] == "both", "fit"].iloc[0]
            if algorithm is not None:
                out = algorithm.predict(pd.DataFrame({"W": W, "A": 1, "Y": np.nan}), model=fit) - \
                    algorithm.predict(pd.DataFrame({"W": W, "A": 0, "Y": np.nan}), model=fit)
            else:
                out = fit.predict(pd.DataFrame({"W": W, "A": 1, "Y": np.nan})) - \
                        fit.predict(pd.DataFrame({"W": W, "A": 0, "Y": np.nan}))
        else:
            fit_one = Qbar_hat.loc[Qbar_hat["a"] == "one", "fit"].iloc[0]
            fit_zero = Qbar_hat.loc[Qbar_hat["a"] == "zero", "fit"].iloc[0]
            if algorithm is not None: # if a ML model is used
                out = algorithm.predict(pd.DataFrame({"W": W, "A": 1, "Y": np.nan}), model=fit_one) - \
                        algorithm.predict(pd.DataFrame({"W": W, "A": 0, "Y": np.nan}), model=fit_zero)
            else:
                # if a non-ML model is used
                out = fit_one.predict(pd.DataFrame({"W": W, "A": 1, "Y": np.nan})) - \
                        fit_zero.predict(pd.DataFrame({"W": W, "A": 0, "Y": np.nan}))
    else: # if the true function is used, e.g. experiment.Qbar()
        out = Qbar_hat(pd.DataFrame({"W": W, "A": 1, "Y": np.nan})) - Qbar_hat(pd.DataFrame({"W": W, "A": 0, "Y": np.nan}))

    # Compute the G-computation estimator (psi_n)
    weights = QW["weight"].values
    if np.isnan(out).any():
        masked_data = np.ma.masked_array(out, mask=np.isnan(out))
        psi_n = np.ma.average(masked_data, weights=weights)
    else:
        psi_n = np.average(out, weights=weights)

    # Compute the standard error (sig_n)
    if np.isnan(out).any():
        masked_data = np.ma.masked_array(out, mask=np.isnan(out))
        variance = np.ma.average((masked_data - psi_n) ** 2, weights=weights)
    else:
        variance = np.average((out - psi_n) ** 2, weights=weights)
    sig_n = np.sqrt(variance) / np.sqrt(n)

    # Return the results as a DataFrame
    return pd.DataFrame({"psi_n": [psi_n], "sig_n": [sig_n]})

# 8.4.3 Empirical investigation, fixed sample size
def update_learned_features(learned_features, algorithm_d=None, algorithm_e=None):
    # Assuming learned_features_fixed_sample_size is a pandas DataFrame
    for key, entry in learned_features.items():
        entry["Qbar_hat_d"] = estimate_Qbar(entry["obs"], algorithm=algorithm_d)
        entry["Qbar_hat_e"] = estimate_Qbar(entry["obs"], algorithm=algorithm_e)
        QW = estimate_QW(entry["obs"])
        entry["est_d"] = compute_gcomp(QW, entry["Qbar_hat_d"], algorithm=None, use_predict=True)
        entry["est_e"] = compute_gcomp(QW, entry["Qbar_hat_e"], algorithm=algorithm_e, use_predict=True)

    return learned_features

def get_psi_hat_de(updated_features, psi_zero):
    # Assuming learned_features_fixed_sample_size is a pandas DataFrame
    psi_hat_de = []
    for key, entry in updated_features.items():
        est_d = entry["est_d"]
        psi_hat_de.append({
            "id": key,
            "psi_n": est_d["psi_n"].values[0],  # Assuming est_d is a DataFrame
            "sig_n": est_d["sig_n"].values[0],  # Assuming est_d is a DataFrame
            "type": "d"
        })

        est_e = entry["est_e"]
        psi_hat_de.append({
            "id": key,
            "psi_n": est_e["psi_n"].values[0],  # Assuming est_e is a DataFrame
            "sig_n": est_e["sig_n"].values[0],  # Assuming est_e is a DataFrame
            "type": "e"
        })
    psi_hat_de = pd.DataFrame(psi_hat_de)

    psi_hat_de = psi_hat_de.groupby("type").apply(
        lambda group: group.assign(sig_alt=np.std(group["psi_n"]))
    ).reset_index(drop=True)
    
    # Add clt_ and clt_alt columns
    psi_hat_de["clt_"] = (psi_hat_de["psi_n"] - psi_zero) / psi_hat_de["sig_n"]
    psi_hat_de["clt_alt"] = (psi_hat_de["psi_n"] - psi_zero) / psi_hat_de["sig_alt"]

    # Step 1: Pivot longer
    psi_hat_de = psi_hat_de.melt(
        id_vars=["id", "psi_n", "sig_n", "sig_alt", "type"],  # Keep the "id" column
        value_vars=["clt_", "clt_alt"],  # Columns to pivot
        var_name="key",  # New column for variable names
        value_name="clt"  # New column for values
    )
    
    # Step 2: Extract the part after the underscore
    psi_hat_de["key"] = psi_hat_de["key"].str.extract(r"_(.*)$")[0]
    
    # Step 3: Convert key to boolean
    psi_hat_de["key"] = psi_hat_de["key"].apply(lambda x: True if x == "" else False)
    
    # Step 4: Rename the key column
    psi_hat_de = psi_hat_de.rename(columns={"key": "auto_renormalization"})

    # Step 5: Group by type and auto_renormalization, then compute the bias
    bias_de = (
        psi_hat_de.groupby(["type", "auto_renormalization"])
        .agg(bias=("clt", "mean"))  # Compute the mean of the "clt" column
        .reset_index()  # Reset the index to flatten the DataFrame
    )

    return psi_hat_de, bias_de

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
                        label=f"Type {type_}", 
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
            ax[id_norm].set_xlim(-40, 40)
            ax[id_norm].set_ylim(0, 0.5)
            ax[id_norm].legend(title="Type", loc='upper right')
        
    fig.suptitle(r"$\sqrt{n/v_n^{d, e}} * (\psi_n^{d, e} - \psi_0)$", fontsize=12)
    plt.show()

# 8.4.4 Empirical investigation, varying sample size
def get_feature_sample_size(obs_df, sample_size, algorithm_d=None, algorithm_e=None):
    """
    Create a dictionary where each block corresponds to a subset of the data.

    Args:
        obs_df (pd.DataFrame): The input DataFrame containing columns W, A, and Y.
        sample_size (list): A list of integers specifying the sample sizes for each block.

    Returns:
        dict: A dictionary with block indices as keys and corresponding DataFrames as values.
    """
    # Ensure the total number of rows is a multiple of the block size
    block_size = sum(sample_size)
    n_rows = (len(obs_df) // block_size) * block_size
    obs_df = obs_df.head(n_rows)

    # Repeat the sample_size pattern to cover all rows
    block_sizes = np.tile(sample_size, n_rows // sum(sample_size) + 1)
    # Generate labels for each block
    labels = np.repeat(np.arange(len(block_sizes)), block_sizes)[:n_rows]

    # Create block labels
    obs_df["block"] = labels

    # Group by block and create the dictionary
    learned_features = {
        block: {'obs': group.drop(columns=["block"])} for block, group in obs_df.groupby("block")
    }

    for key, entry in learned_features.items():
        entry["Qbar_hat_d"] = estimate_Qbar(entry["obs"], algorithm=algorithm_d)
        entry["Qbar_hat_e"] = estimate_Qbar(entry["obs"], algorithm=algorithm_e)
        QW = estimate_QW(entry["obs"])
        entry["est_d"] = compute_gcomp(QW, entry["Qbar_hat_d"], algorithm=None, use_predict=True)
        entry["est_e"] = compute_gcomp(QW, entry["Qbar_hat_e"], algorithm=algorithm_e, use_predict=True)

    return learned_features

def get_root_n_bias(learned_features_varying_sample_size, psi_zero):
    psi_hat_de = []
    for key, entry in learned_features_varying_sample_size.items():
        est_d = entry["est_d"]
        psi_hat_de.append({
            "block": key,
            "sample_size": entry["obs"].shape[0],  # Assuming entry["obs"] is a DataFrame
            "psi_n": est_d["psi_n"].values[0],  # Assuming est_d is a DataFrame
            "sig_n": est_d["sig_n"].values[0],  # Assuming est_d is a DataFrame
            "type": "d"
        })

        est_e = entry["est_e"]
        psi_hat_de.append({
            "block": key,
            "sample_size": entry["obs"].shape[0],  # Assuming entry["obs"] is a DataFrame
            "psi_n": est_e["psi_n"].values[0],  # Assuming est_e is a DataFrame
            "sig_n": est_e["sig_n"].values[0],  # Assuming est_e is a DataFrame
            "type": "e"
        })
    psi_hat_de = pd.DataFrame(psi_hat_de)

    psi_hat_de = psi_hat_de.groupby(["sample_size", "type"]).apply(
        lambda group: group.assign(sig_alt=np.std(group["psi_n"]))
    ).reset_index(drop=True)

    # Add clt_ and clt_alt columns
    psi_hat_de["clt_"] = (psi_hat_de["psi_n"] - psi_zero) / psi_hat_de["sig_n"]
    psi_hat_de["clt_alt"] = (psi_hat_de["psi_n"] - psi_zero) / psi_hat_de["sig_alt"]

    # Step 1: Pivot longer
    psi_hat_de = psi_hat_de.melt(
        id_vars=["block", "psi_n", "sig_n", "sig_alt", "type", "sample_size"],  # Keep the "id" column
        value_vars=["clt_", "clt_alt"],  # Columns to pivot
        var_name="key",  # New column for variable names
        value_name="clt"  # New column for values
    )
    
    # Step 2: Extract the part after the underscore
    psi_hat_de["key"] = psi_hat_de["key"].str.extract(r"_(.*)$")[0]
    
    # Step 3: Convert key to boolean
    psi_hat_de["key"] = psi_hat_de["key"].apply(lambda x: True if x == "" else False)
    
    # Step 4: Rename the key column
    psi_hat_de = psi_hat_de.rename(columns={"key": "auto_renormalization"})

    # Step 5: Group by type and auto_renormalization, then compute the bias
    bias_de_root_n = (
        psi_hat_de.groupby(["type", "auto_renormalization", "sample_size"])
        .agg(bias=("clt", "mean"))  # Compute the mean of the "clt" column
        .reset_index()  # Reset the index to flatten the DataFrame
    )

    return psi_hat_de, bias_de_root_n

def plot_root_n_bias(psi_hat_de_df_combined, psi_zero):
        # Filter the data for auto-renormalization
        filtered_data = psi_hat_de_df_combined[psi_hat_de_df_combined["auto_renormalization"]]
        
        # Add a new column for root-n bias (rnb)
        filtered_data = filtered_data.assign(
            rnb=np.sqrt(filtered_data["sample_size"]) * (filtered_data["psi_n"] - psi_zero)
        )
        
        # Group by sample_size and type
        grouped_data = filtered_data.groupby(["sample_size", "type"])
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        
        # Plot mean and standard error (similar to stat_summary in ggplot2)
        for (sample_size, type_), group in grouped_data:
            # print(sample_size)
            # print(type_)
            # print()

            x_axis = sample_size-50 if type_ == "d" else sample_size + 50

            # Compute mean and standard error
            mean_rnb = group["rnb"].mean()
            se_rnb = sem(group["rnb"])
        
            # Plot error bars
            plt.errorbar(
                x=x_axis,  # Adjust x-axis position for better visibility
                # x=[sample_size],
                y=[mean_rnb],
                yerr=[se_rnb],
                fmt="o",
                label=f"{type_} (n={sample_size})",
                capsize=5,
                markersize=7,
                color="blue" if type_ == "d" else "orange",
            )
        
            # Add scatter points for individual data points
            sns.scatterplot(data=filtered_data,
                            x=x_axis,
                            y="rnb",
                            alpha=0.1,
                            color="blue" if type_ == "d" else "orange",
                        )
        
        # Customize the x-axis
        plt.xticks(
            ticks=np.unique(np.concatenate([[B / iter], filtered_data["sample_size"].unique()])),
            labels=np.unique(np.concatenate([[B / iter], filtered_data["sample_size"].unique()])),
        )
        
        # Add labels and title
        plt.xlabel("Sample size n")
        plt.ylabel(r"$\sqrt{n} (\psi_n^{d,e} - \psi_0)$")
        plt.title("Root-n Bias vs Sample Size")
        plt.legend(title="Type")
        plt.grid(True)
        
        # Show the plot
        plt.tight_layout()
        plt.show()

# Practice Problem 8.5.1
class WorkingModelQTwo(WorkingModelQOne):
    def __init__(self):
        """
        Initialize the WorkingModelQTwo with optional powers.
        """
        super().__init__()

    def fit(self, dat, formula=None, **kwargs):
        """
        Fit the model to the data.

        Parameters:
            dat: pandas DataFrame containing the data.
            formula: Optional formula to override the default formula.

        Returns:
            fit: The fitted model.
        """
        if isinstance(dat, np.ndarray):
            dat = pd.DataFrame(dat, columns=["W", "A", "Y"])

        if formula is None:
            formula = self.formula

        self.model = smf.glm(formula=formula, data=dat, 
                             family=sm.families.Gaussian(), **kwargs).fit()
        return self.model

# Practice Problem 8.5.3
def estimate_QW_new(sample_size=1000):
    """
    Estimate QW by transforming the input data.

    Parameters:
        dat: pandas DataFrame containing a column "W".

    Returns:
        result: pandas DataFrame with columns "value" and "weight".
    """
    # Convert to DataFrame if necessary
    # Ensure W is a numpy array
    W = np.random.normal(loc=5, scale=2, size=sample_size) 

    # Estimate the mean and variance using MLE
    mean_est = np.mean(W)
    variance_est = np.var(W, ddof=0)  # Use population variance (ddof=0)
    density = norm.pdf(W, loc=mean_est, scale=np.sqrt(variance_est))

    # Select "W" column and compute weights
    result = pd.DataFrame({"value": W})
    result["weight"] = density

    return result



if __name__ == "__main__":
    
    # initialize an experiment
    experiment = Experiment()
    B=int(1e6)
    obs = experiment.sample_from(n=B)
    obs_df = pd.DataFrame(obs, columns=["W", "A", "Y"])

    # 8.2.1 Construction and computation
    # Truth example
    out = compute_iptw(obs_df[:int(1e3)], experiment.Gbar)
    print("psi_n is {} and sig_n is {}".format(out["psi_n"].values[0], out["sig_n"].values[0]))

    # Esimation example
    working_model_G_one = WorkingModelGOne()
    Gbar_hat = estimate_Gbar(dat=obs_df[:int(1e3)], algorithm=working_model_G_one)
    out = compute_iptw(obs_df[:int(1e3)], Gbar_hat, use_predict=True)
    print("psi_n is {} and sig_n is {}".format(out["psi_n"].values[0], out["sig_n"].values[0]))

    # 8.2.3 Empirical investigation
    # Initialize parameters
    iter = 1000  # Number of iterations
    working_model_G_one = WorkingModelGOne()
    learned_features_fixed_sample_size = get_learned_features(obs_df, iter, working_model_G_one)
    bias_abc, psi_hat_abc_df = get_psi_hat_abc(learned_features_fixed_sample_size, experiment, obs_df, iter)
    bias_abc

    # plot bias and estimation
    plot_estimation_bias(psi_hat_abc_df, bias_abc)

    # plot Q-Q plot
    plot_qq(psi_hat_abc_df)

    # estimate with the empirical variance
    # Filter rows where type == "c"
    filtered_df = psi_hat_abc_df[psi_hat_abc_df["type"] == "c"]
    
    # Calculate the empirical standard deviation of psi_n
    emp_sig_n = filtered_df["psi_n"].std()
    print(f"Empirical standard deviation of psi_n: {emp_sig_n:.4f}")
    
    # Summarize the sig_n column
    summ_sig_n = filtered_df["sig_n"].describe()
    print("Summary of sig_n:")
    print(summ_sig_n)

    # Plot empirical bias and estimation
    psi_hat_abc_df_emp_c, emp_sig_n_c = get_estimation_bias_empirical(psi_hat_abc_df)
    plot_estimation_bias(psi_hat_abc_df_emp_c, bias_abc)

    # Practice Problem 8.3.1
    psi_hat_abc_df_emp_b, emp_sig_n_b = get_estimation_bias_empirical(psi_hat_abc_df, est_type="b")
    slope_CI_b, in_CI_b = evaluate_CI(psi_hat_abc_df_emp_b, alpha=0.05, est_type="b", psi_zero=evaluate_psi(experiment))
    p_value_b_emp_greater = perform_binom_test(in_CI_b, n=1000, p=0.95, alternative='greater')
    print(f"Empirical one-sided p-value for type 'b' (greater): {p_value_b_emp_greater}")
    p_value_b_emp_less = perform_binom_test(in_CI_b, n=1000, p=0.05, alternative='less')
    print(f"Empirical one-sided p-value for type 'b' (less): {p_value_b_emp_less}")
    p_value_b_emp_two_sided = perform_binom_test(in_CI_b, n=1000, p=0.95, alternative='two-sided')
    print(f"Empirical two-sided p-value for type 'b': {p_value_b_emp_two_sided}")


    # Practice Problem 8.3.2
    working_model_G_two = WorkingModelGTwo()
    learned_features_fixed_sample_size = get_learned_features(obs_df, iter, working_model_G_two)
    bias_abc, psi_hat_abc_df = get_psi_hat_abc(learned_features_fixed_sample_size, experiment, obs_df, iter)
    plot_estimation_bias(psi_hat_abc_df, bias_abc)
    plot_qq(psi_hat_abc_df)

     
    # Practice Problem 8.3.3
    powers=np.repeat(np.arange(0.4, 3.75, 0.25), 2)
    working_model_G_two = WorkingModelGTwo(powers=powers)
    learned_features_fixed_sample_size = get_learned_features(obs_df, iter, working_model_G_two)
    bias_abc, psi_hat_abc_df = get_psi_hat_abc(learned_features_fixed_sample_size, experiment, obs_df, iter)
    plot_estimation_bias(psi_hat_abc_df, bias_abc)
    plot_qq(psi_hat_abc_df)

    # Practice Problem 8.3.4
    working_model_G_three = WorkingModelGThree()
    learned_features_fixed_sample_size = get_learned_features(obs_df, iter, working_model_G_three)
    bias_abc, psi_hat_abc_df = get_psi_hat_abc(learned_features_fixed_sample_size, experiment, obs_df, iter)
    plot_estimation_bias(psi_hat_abc_df, bias_abc)
    plot_qq(psi_hat_abc_df)

    # Practice Problem 8.3.5
    slope_CI_c, in_CI_c = evaluate_CI(psi_hat_abc_df_emp_c, alpha=0.05, est_type="c", psi_zero=evaluate_psi(experiment))
    p_value_c_emp_greater = perform_binom_test(in_CI_c, n=1000, p=0.95, alternative='greater')
    print(f"Empirical one-sided p-value for type 'c' (greater): {p_value_c_emp_greater}")
    p_value_c_emp_less = perform_binom_test(in_CI_c, n=1000, p=0.05, alternative='less')
    print(f"Empirical one-sided p-value for type 'c' (less): {p_value_c_emp_less}")
    p_value_c_emp_two_sided = perform_binom_test(in_CI_c, n=1000, p=0.95, alternative='two-sided')
    print(f"Empirical two-sided p-value for type 'c': {p_value_c_emp_two_sided}")


    # Practice Problem 8.3.6
    psi_zero = evaluate_psi(experiment)
    num_bootstrap = 10
    working_model_G_one = WorkingModelGOne()
    iter = 1000  # Number of iterations for sampling
    in_CI_bootstrap = []
    for i in range(num_bootstrap): 
        obs_b = experiment.sample_from(n=B)
        obs_b_df = pd.DataFrame(obs_b, columns=["W", "A", "Y"])

        learned_features_fixed_sample_size = get_learned_features(obs_b_df, iter, working_model_G_one)
        bias_abc, psi_hat_abc_df = get_psi_hat_abc(learned_features_fixed_sample_size, experiment, obs_df, iter)  
        psi_hat_abc_df_emp_c, emp_sig_n_c = get_estimation_bias_empirical(psi_hat_abc_df)
        slope_CI_c, in_CI_c = evaluate_CI(psi_hat_abc_df_emp_c, alpha=0.05, est_type="c", psi_zero=evaluate_psi(experiment))
        in_CI_bootstrap.append(in_CI_c.astype(int))  # Convert boolean to int for counting
    
    in_CI_bootstrap = list(itertools.chain.from_iterable(in_CI_bootstrap))
    p_value_c_emp_greater = perform_binom_test(in_CI_bootstrap, n=num_bootstrap*iter, p=0.95, alternative='greater')
    print(f"Empirical one-sided p-value for type 'c' (greater): {p_value_c_emp_greater}")
    p_value_c_emp_less = perform_binom_test(in_CI_bootstrap, n=num_bootstrap*iter, p=0.05, alternative='less')
    print(f"Empirical one-sided p-value for type 'c' (less): {p_value_c_emp_less}")
    p_value_c_emp_two_sided = perform_binom_test(in_CI_bootstrap, n=num_bootstrap*iter, p=0.95, alternative='two-sided')
    print(f"Empirical two-sided p-value for type 'c': {p_value_c_emp_two_sided}")

    # Practice Problem 8.3.7
    """As long as the sample size is big enough, that is bootstrap needs to be used."""

    # 8.4.1 Construction and computation
    QW = estimate_QW(dat=obs_df[:int(1e3)])

    kknn_algo = KknnAlgo()
    Qbar_hat_kknn = estimate_Qbar(dat=obs_df[:int(1e3)], algorithm=kknn_algo)
    working_model_Q_one = WorkingModelQOne()
    Qbar_hat_d = estimate_Qbar(dat=obs_df[:int(1e3)], algorithm=working_model_Q_one)

    kknn_g_comp = compute_gcomp(QW, Qbar_hat_kknn, algorithm=kknn_algo, use_predict=True)
    print(kknn_g_comp)

    d_g_comp = compute_gcomp(QW, Qbar_hat_d, algorithm=None, use_predict=True)
    print(d_g_comp)

    # 8.4.3 Empirical investigation, fixed sample size    
    updated_features = update_learned_features(learned_features_fixed_sample_size, algorithm_d=working_model_Q_one, algorithm_e=kknn_algo)
    psi_hat_de_df, bias_de = get_psi_hat_de(updated_features, psi_zero=evaluate_psi(experiment))
    print(bias_de)
    plot_estimation_bias_de(psi_hat_de_df, bias_de)

    ## psi_n^d
    # Filter the DataFrame for type == "d" and auto_renormalization == True
    filtered_sig_n = psi_hat_de_df[
        (psi_hat_de_df["type"] == "d") & (psi_hat_de_df["auto_renormalization"] == True)
    ]["sig_n"]
    
    # Compute summary statistics
    summary_stats = filtered_sig_n.describe()
    
    print(summary_stats)

    ## psi_n^e
    filtered_sig_n = psi_hat_de_df[
        (psi_hat_de_df["type"] == "e") & (psi_hat_de_df["auto_renormalization"] == True)
    ]["sig_n"]
    
    # Compute summary statistics
    summary_stats = filtered_sig_n.describe()
    
    print(summary_stats)

    # 8.4.4 Empirical investigation, varying sample size
    # Define sample sizes
    sample_size = [2500, 5000]
    block_size = sum(sample_size)
    learned_features_varying_sample_size = get_feature_sample_size(obs_df, sample_size, algorithm_d=working_model_Q_one, algorithm_e=kknn_algo)
    print(learned_features_varying_sample_size)
    ''' There is some discrepency in code here:
    This is how block variable defined:
        R: block = label(1:nrow(.), sample_size)
    which means that the observations will share the same block number based on sample size assigned

    However, in the calculation of root_n_bias, we catch a piece of code:
        R: group_by(block, type)
    if we group by block and type, then there is only "psi_n" to look at becasue we calcualte psi_n using all the data in one block

    Based on the following piece of code:
     R: mutate(block = "0",
          sample_size = B/iter) %>%  # because *fixed* sample size
    which group all blocks with the same size under the same block number.
    
    We believe that the code is not correct, instead of using the group_by(block, type), 
    we should use group_by(sample_size, type) to get the correct root_n_bias
    '''
    psi_hat_de_df_vs, bias_de_rootn_vs = get_root_n_bias(learned_features_varying_sample_size, psi_zero=evaluate_psi(experiment))
    print(bias_de_rootn_vs)  
    psi_hat_de_df_fs, bias_de_rootn_fs = get_root_n_bias(learned_features_fixed_sample_size, psi_zero=evaluate_psi(experiment))
    print(bias_de_rootn_fs)  


    # Plot root-n bias for varying sample size
    bias_de_rootn_combined = pd.concat([bias_de_rootn_vs, bias_de_rootn_fs], ignore_index=True)
    print(bias_de_rootn_combined)

    psi_hat_de_df_combined = pd.concat([psi_hat_de_df_vs, psi_hat_de_df_fs], ignore_index=True)
    print(psi_hat_de_df_combined)

    plot_root_n_bias(psi_hat_de_df_combined, psi_zero=evaluate_psi(experiment))

    # Pratice Problem 8.5.1
    '''Use Gaussian distribution to estimate the conditional mean Qbar_hat, because Qbar_hat represents a regression model'''
    working_model_Q_two = WorkingModelQTwo()
    Qbar_hat_d = estimate_Qbar(dat=obs_df[:int(1e3)], algorithm=working_model_Q_two)

    # Practice Problem 8.5.2
    QW = estimate_QW(dat=obs_df[:int(1e3)])
    d_g_comp = compute_gcomp(QW, Qbar_hat_d, algorithm=None, use_predict=True)
    print(d_g_comp)

    updated_features = update_learned_features(learned_features_fixed_sample_size, algorithm_d=working_model_Q_two, algorithm_e=kknn_algo)
    psi_hat_de_df, bias_de = get_psi_hat_de(updated_features, psi_zero=evaluate_psi(experiment))
    print(bias_de)
    plot_estimation_bias_de(psi_hat_de_df, bias_de)


    # Practice Problem 8.5.3
    # See Above

    # Practice Problem 8.5.4
    QW = estimate_QW_new(sample_size=1000)
    d_g_comp = compute_gcomp(QW, Qbar_hat_d, algorithm=None, use_predict=True)
    print(d_g_comp)





    


    


    

    




















