import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit, logit
from scipy.stats import norm

from python_section_1 import LAW, Experiment
from python_section_2 import AnotherExperiment
from python_section_3 import evaluate_psi, compute_eic

# 6.2 Delta-method
def compute_irrelevant_estimator(obs):
    """
    Compute an irrelevant estimator based on the given observations.

    Parameters:
        obs: pandas DataFrame containing columns "Y" and "A".

    Returns:
        result: pandas DataFrame containing psi_n and sig_n.
    """
    if isinstance(obs, np.ndarray):
        # Extract columns
        A = obs[:, 1]  # Column 1 corresponds to "A"
        Y = obs[:, 2]  # Column 2 corresponds to "Y"
    elif isinstance(obs, pd.DataFrame):
        # Extract columns
        A = obs["A"].values
        Y = obs["Y"].values
    else:
        raise TypeError("obs must be a pandas DataFrame or a NumPy array.")

    # Compute psi_n
    psi_n = np.mean(A * Y) / np.mean(A) - np.mean((1 - A) * Y) / np.mean(1 - A)

    # Compute covariance matrix
    Var_n = np.cov(np.column_stack((A * Y, A, (1 - A) * Y, (1 - A))), rowvar=False)

    # Compute phi_n
    phi_n = np.array([
        1 / np.mean(A),
        -np.mean(A * Y) / np.mean(A)**2,
        -1 / np.mean(1 - A),
        np.mean((1 - A) * Y) / np.mean(1 - A)**2
    ])

    # Compute variance
    var_n = float(phi_n.T @ Var_n @ phi_n)

    # Compute standard error
    sig_n = np.sqrt(var_n / len(obs))

    # Return results as a DataFrame
    result = pd.DataFrame({"psi_n": [psi_n], "sig_n": [sig_n]})
    return result

# 6.3.3 Empirical investigation
def compute_iptw(obs, Gbar, threshold=0.05):
    """
    Compute the Inverse Probability of Treatment Weighting (IPTW) estimator.

    Parameters:
        obs: pandas DataFrame containing columns "W", "A", and "Y".
        Gbar: Function to compute G(W).
        threshold: Numeric value to threshold probabilities (default is 0.05).

    Returns:
        result: pandas DataFrame containing psi_n and sig_n.
    """
    if isinstance(obs, np.ndarray):
        # Extract columns
        W = obs[:, 0]  # Column 0 corresponds to "W"
        A = obs[:, 1]  # Column 1 corresponds to "A"
        Y = obs[:, 2]  # Column 2 corresponds to "Y"
    elif isinstance(obs, pd.DataFrame):
        W = obs["W"].values
        A = obs["A"].values
        Y = obs["Y"].values
    else:
        raise TypeError("obs must be a pandas DataFrame or a NumPy array.")

    # Compute lGAW
    lGAW = A * Gbar(W) + (1 - A) * (1 - Gbar(W))
    lGAW = np.maximum(threshold, lGAW)  # Apply threshold

    # Compute psi_n
    psi_n = np.mean(Y * (2 * A - 1) / lGAW)

    # Compute standard error
    sig_n = np.std(Y * (2 * A - 1) / lGAW, ddof=1) / np.sqrt(len(obs))

    # Return results as a DataFrame
    result = pd.DataFrame({"psi_n": [psi_n], "sig_n": [sig_n]})
    return result


def compute_psi_hat_ab(obs, experiment, iter=1000):
    """
    Compute psi_hat_ab by grouping observations and applying estimators.

    Parameters:
        obs: pandas DataFrame containing columns "W", "A", and "Y".
        experiment: An instance of a LAW subclass with a Gbar method.
        iter: Number of iterations for grouping.

    Returns:
        psi_hat_ab: pandas DataFrame containing id, type, psi_n, and sig_n.
    """
    # Check if obs is a NumPy array, transform it into a DataFrame
    if isinstance(obs, np.ndarray):
        obs = pd.DataFrame(obs, columns=["W", "A", "Y"])
    
    # Add an ID column for grouping
    obs["id"] = (np.arange(len(obs)) % iter)

    # Group observations by ID
    grouped = obs.groupby("id")

    # Apply estimators to each group
    results = []
    for id, group in grouped:
        est_a = compute_irrelevant_estimator(group)
        est_b = compute_iptw(group, experiment.Gbar)
        results.append({"id": id, "type": "a", "psi_n": est_a["psi_n"].iloc[0], "sig_n": est_a["sig_n"].iloc[0]})
        results.append({"id": id, "type": "b", "psi_n": est_b["psi_n"].iloc[0], "sig_n": est_b["sig_n"].iloc[0]})

    # Convert results to a DataFrame
    psi_hat_ab = pd.DataFrame(results)

    return psi_hat_ab

def compute_clt(psi_hat_ab, psi_zero):
    """
    Compute the clt column for psi_hat_ab.

    Parameters:
        psi_hat_ab: pandas DataFrame containing columns "id", "psi_n", and "sig_n".
        psi_zero: True value of Psi.

    Returns:
        psi_hat_ab: Updated DataFrame with the clt column.
    """
    # Group by id and compute clt
    psi_hat_ab["clt"] = (psi_hat_ab["psi_n"] - psi_zero) / psi_hat_ab["sig_n"]
    return psi_hat_ab

def compute_bias_ab(psi_hat_ab):
    """
    Compute bias_ab by grouping psi_hat_ab by type and summarizing the bias.

    Parameters:
        psi_hat_ab: pandas DataFrame containing columns "type" and "clt".

    Returns:
        bias_ab: pandas DataFrame containing type and bias.
    """
    # Group by type and compute bias
    bias_ab = psi_hat_ab.groupby("type").agg(bias=("clt", "mean")).reset_index()
    return bias_ab


def plot_bias_ab(psi_hat_ab, bias_ab):
    """
    Plot bias_ab using density plots and vertical bias lines.

    Parameters:
        psi_hat_ab: pandas DataFrame containing columns "clt" and "type".
        bias_ab: pandas DataFrame containing columns "type" and "bias".

    Returns:
        None. Displays the plot.
    """
    # Create standard normal density data
    x = np.linspace(-3, 3, int(1e3))
    y = norm.pdf(x)

    # Create the plot
    plt.figure(figsize=(10, 6))

    # Plot standard normal density
    plt.plot(x, y, linestyle="--", alpha=0.5, label="Standard Normal Density")

    # Plot density of clt values for each type
    for type_, group in psi_hat_ab.groupby("type"):
        sns.kdeplot(group["clt"], fill=True, alpha=0.1, label=f"Type {type_}")

    # Add vertical lines for bias
    for _, row in bias_ab.iterrows():
        plt.axvline(x=row["bias"], color="red", linestyle="--", linewidth=1.5, alpha=0.5, label=f"Bias (Type {row['type']})")

    # Add labels and legend
    plt.xlabel(r"$\sqrt{n/v_n^{\{a, b\}}} (\psi_n^{\{a, b\}} - \psi_0)$")
    plt.ylabel("")
    plt.legend()
    plt.grid(True)
    plt.show()


# Example usage
if __name__ == "__main__":
    # Example usage

    # 6.2 Delta-method
    experiment = Experiment()
    B=int(1e6)
    obs = experiment.sample_from(n=B)
    results = compute_irrelevant_estimator(obs[:int(1e3)])
    print(results)

    # 6.3.3 Empirical investigation
    result = compute_iptw(obs, experiment.Gbar)
    print(result)

    # Compute psi_hat_ab
    psi_hat_ab = compute_psi_hat_ab(obs, experiment)
    print(psi_hat_ab)

    # Compute clt
    # True value of Psi
    psi_zero = evaluate_psi(experiment)

    # Compute clt column
    psi_hat_ab = compute_clt(psi_hat_ab, psi_zero)
    print(psi_hat_ab)

    # Compute bias_ab
    bias_ab = compute_bias_ab(psi_hat_ab)
    print(bias_ab)

    # Plot bias_ab
    plot_bias_ab(psi_hat_ab, bias_ab)
