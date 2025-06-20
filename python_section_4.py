import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit, logit

from python_section_1 import LAW, Experiment
from python_section_2 import AnotherExperiment
from python_section_3 import evaluate_psi, compute_eic


# 4.1.4 Expressing the remainder term as a function of the relevant features
def compute_integrand(w, Qbar_A, Qbar_B, Gbar_A, Gbar_B, QW_A, Params_A={}, Params_B={}):
    """
    Compute the integrand for the remainder term.

    Parameters:
        w: Array of W values.
        Qbar_A: Function to compute Q(A, W) for experiment1.
        Qbar_B: Function to compute Q(A, W) for experiment2.
        Gbar_A: Function to compute G(W) for experiment1.
        Gbar_B: Function to compute G(W) for experiment2.
        QW_A: Function to compute Q(W) for experiment1.
        Params_A: Dictionary of keyword arguments for Qbar_A.
        Params_B: Dictionary of keyword arguments for Qbar_B.

    Returns:
        fw: Computed integrand values.
    """
    fw = (Qbar_A(np.column_stack((np.ones(len(w)), w)), **Params_A) - 
          Qbar_B(np.column_stack((np.ones(len(w)), w)), **Params_B)) / Gbar_B(w)
    fw += (Qbar_A(np.column_stack((np.zeros(len(w)), w)), **Params_A) - 
           Qbar_B(np.column_stack((np.zeros(len(w)), w)), **Params_B)) / (1 - Gbar_B(w))
    fw *= (Gbar_A(w) - Gbar_B(w)) * QW_A(w)
    return fw


def compute_remainder(experiment1, experiment2, params=None):
    """
    Compute the remainder term for two experiments.

    Parameters:
        experiment1: An instance of a LAW subclass (e.g., Experiment).
        experiment2: Another instance of a LAW subclass.
        params: Optional. A list of two dictionaries, each providing keyword arguments for the corresponding LAW object.

    Returns:
        remainder: The computed remainder term.
    """
    # Validate params
    if params is not None:
        if len(params) != 2:
            raise ValueError(
                "If not 'None', argument 'params' must be a list consisting of two dictionaries, "
                "each providing parameters to the corresponding LAW object."
            )
        params_A, params_B = params
    else:
        params_A, params_B = {}, {}

    Qbar_A = experiment1.Qbar
    QW_A = experiment1.QW
    Gbar_A = experiment1.Gbar
    Qbar_B = experiment2.Qbar
    Gbar_B = experiment2.Gbar

    # Compute remainder term
    if callable(QW_A):  # If QW_A is a function
        w_values = np.linspace(0, 1, 1000)  # Generate values for integration
        integrand_values = compute_integrand(w_values, Qbar_A, Qbar_B, Gbar_A, Gbar_B, QW_A, params_A, params_B)
        remainder = np.trapz(integrand_values, w_values)  # Numerical integration
    else:  # If QW_A is a discrete law
        if not all(col in QW_A.columns for col in ["value", "weight"]):
            raise ValueError("Argument 'QW' is neither a function nor a valid discrete law.")

        W = QW_A["value"].values
        weights = QW_A["weight"].values
        fw = compute_integrand(W, Qbar_A, Qbar_B, Gbar_A, Gbar_B, lambda w: np.ones(len(w)), params_A, params_B)
        remainder = np.average(fw, weights=weights)  # Weighted average

    return remainder

# 4.2 The remainder term: Practice Problem 1
def compute_remainder_vs_h(experiment1, experiment2, h_values):
    """
    Compute the remainder term for different values of h.

    Parameters:
        experiment1: An instance of a LAW subclass (e.g., Experiment).
        experiment2: Another instance of a LAW subclass.
        h_values: Array of h values to sample.

    Returns:
        results: A dictionary with h values and corresponding remainder values.
    """
    results = {"h": [], "remainder": []}

    for h in h_values:
        params_A = {}  # Parameters for experiment1
        params_B = {"h": h}  # Parameters for experiment2

        remainder = compute_remainder(experiment1, experiment2, params=[params_A, params_B])
        results["h"].append(h)
        results["remainder"].append(remainder)

    return results


def plot_remainder_vs_h(results):
    """
    Plot the remainder values against h.

    Parameters:
        results: A dictionary with h values and corresponding remainder values.

    Returns:
        None. Displays the plot.
    """
    h_values = results["h"]
    remainder_values = results["remainder"]

    plt.figure(figsize=(10, 6))
    plt.plot(h_values, remainder_values, marker="o", linestyle="-", color="b", label="Remainder")
    plt.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    plt.xlabel("h")
    plt.ylabel("Remainder")
    plt.title("Remainder Term vs h")
    plt.legend()
    plt.grid(True)
    plt.show()


# 4.2 The remainder term: Practice Problem 2 
'''
CAUTION: There is a typo in equation 4.2 in some printed versions.

Correctly derived from equation 4.1:
    Ψ(P₀) = Ψ(P) + (P₀ − P) · D*(P) − Rem_{P₀}(P)

Rearranged:
    Rem_{P₀}(P) = Ψ(P) + (P₀ − P) · D*(P) − Ψ(P₀)

Expanded as:
    = Ψ(P) + E_{P₀}[D*(P)] − E_{P}[D*(P)] − Ψ(P₀)
    = ψ_p + mean(eic_p₀) − mean(eic_p) − ψ_p0
    = (ψ_p − ψ_p0) + (mean(eic_p₀) − mean(eic_p))

But:
    E_P[D*(P)] = 0 by definition of the canonical gradient
So:
    Rem_{P₀}(P) ≈ ψ_p + mean(eic_p₀) − ψ_p0
'''

def compute_remainder_vs_h_numerical(experiment1, experiment2, h_values, B=int(1e6)):
    """
    Numerically estimate the second-order remainder term Rem_{P₀}(P)
    for a range of fluctuation levels h, using Monte Carlo approximation.

    Args:
        experiment1: A baseline experiment representing P₀ (e.g., the true DGP).
        experiment2: A fluctuated experiment representing P (depends on h).
        h_values: A list of h values controlling the strength of fluctuation in P.
        B: Number of Monte Carlo samples to use (should be large).

    Returns:
        results: A dictionary mapping each h to its estimated remainder.
    """
    results = {"h": [], "remainder": []}

    for h in h_values:
        # Define parameters for baseline (P₀) and fluctuated model (P = Πₕ)
        params_A = {}            # P₀: fixed
        params_B = {"h": h}      # P: fluctuated by h

        # Compute Ψ(P₀) and Ψ(P)
        psi_p0 = evaluate_psi(experiment1, **params_A)   # True value under P₀
        psi_p  = evaluate_psi(experiment2, **params_B)   # Plug-in estimate under Πₕ

        # Sample from P₀ (observations to integrate D*(P) under P₀)
        obs_p0 = experiment1.sample_from(n=B, **params_A)

        # Compute D*(P) evaluated at obs ~ P₀ (correct way to approximate E_{P₀}[D*(P)])
        eic_p0 = compute_eic(experiment2, obs_p0, psi_p, **params_B)

        # Numerical estimate of Rem_{P₀}(P) based on equation (4.2):
        remainder = psi_p + np.mean(eic_p0) - psi_p0

        # NOTE: np.mean(eic_p) = 0, by definition of the canonical gradient under P
        # So no need to compute it, unless verifying theory.

        # Store results
        results["h"].append(h)
        results["remainder"].append(remainder)

    return results


# Example usage
if __name__ == "__main__":

    # 4.1.4 part I
    experiment1 = Experiment()
    experiment2 = Experiment()

    params_A = {}  # Example parameters for experiment1
    params_B = {}  # Example parameters for experiment2

    remainder = compute_remainder(experiment1, experiment2, params=[params_A, params_B])
    print(f"Remainder term: {remainder}")

    # 4.1.4 part II
    experiment1 = Experiment()
    experiment2 = AnotherExperiment()

    params_A = {}  # Example parameters for experiment1
    params_B = {"h": 0}  # Example parameters for experiment2

    remainder = compute_remainder(experiment1, experiment2, params=[params_A, params_B])
    print(f"Remainder term with different experiments: {remainder}")

    # 4.2 Practice Problem 1
    # Sample h values from [-1, 1]
    h_values = np.linspace(-1, 1, 100)

    # Compute remainder values for different h
    results = compute_remainder_vs_h(experiment1, experiment2, h_values)

    # Plot the remainder values against h
    plot_remainder_vs_h(results)

    # 4.2 Practice Problem 2
    results_numerical = compute_remainder_vs_h_numerical(experiment1, experiment2, 
                                                         h_values, B=int(1e6))
    # Plot the numerical results
    plot_remainder_vs_h(results_numerical)

