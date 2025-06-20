import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit, logit

from python_section_1 import LAW, Experiment
from python_section_2 import AnotherExperiment


# 3.1.2 Numerical illustration
# Evaluate psi_Pi_h for each value of `approx`
def evaluate_psi(law, **kwargs):
    """
    Evaluate Psi for the given law.

    Parameters:
        law: An instance of a LAW subclass (e.g., AnotherExperiment or YetAnotherExperiment).
        kwargs: Additional arguments passed to relevant feature functions.

    Returns:
        psi: The value of Psi at the given law.
    """
    # Reveal relevant features
    some_relevant_features = law.reveal()

    # Check if the law is sufficiently characterized
    if len(set(some_relevant_features.keys()).intersection({"Qbar", "QW"})) != 2:
        raise ValueError(f"Is law '{law.name}' sufficiently characterized?")

    # Access Qbar and QW directly from the law instance
    Qbar = law.Qbar
    QW = law.QW

    # Compute Psi
    if callable(QW):  # If QW is a function
        w_values = np.linspace(0, 1, 1000)  # Generate values for integration
        q1w = Qbar(np.column_stack((np.ones(len(w_values)), w_values)), **kwargs)
        q0w = Qbar(np.column_stack((np.zeros(len(w_values)), w_values)), **kwargs)
        integrand = (q1w - q0w) * QW(w_values)
        psi = np.trapz(integrand, w_values)  # Numerical integration
    else:  # If QW is a discrete law
        if not all(col in QW.columns for col in ["value", "weight"]):
            raise ValueError("Argument 'QW' is neither a function nor a valid discrete law.")

        W = QW["value"].values
        weights = QW["weight"].values
        q1w = Qbar(np.column_stack((np.ones(len(W)), W)), **kwargs)
        q0w = Qbar(np.column_stack((np.zeros(len(W)), W)), **kwargs)
        psi_values = q1w - q0w
        psi = np.average(psi_values, weights=weights)  # Weighted average

    return psi

def gen_data(another_experiment, approx=np.linspace(-1, 1, int(1e2))):
    # Compute psi_Pi_h for all values of `approx`
    psi_Pi_h = np.array([evaluate_psi(another_experiment, h=t) for t in approx])

    # Compute the slope approximation
    psi_Pi_zero = evaluate_psi(another_experiment, h=0)
    slope_approx = (psi_Pi_h - psi_Pi_zero) / approx
    slope_approx = slope_approx[np.where(approx > 0)[0][0]]

    # Create a DataFrame for plotting
    data = pd.DataFrame({"h": approx, "psi_Pi_h": psi_Pi_h})

    return data, psi_Pi_zero, slope_approx

def plot_data(data, psi_Pi_zero, slope_approx):
    # Plot the results
    plt.figure(figsize=(10, 6))

    # Scatter plot for psi_Pi_h
    plt.scatter(data["h"], data["psi_Pi_h"], color="#CC6666", label=r"$\Psi(\Pi_h)$")

    # Add a segment for the slope approximation
    plt.plot([-1, 1], [psi_Pi_zero - slope_approx, psi_Pi_zero + slope_approx],
            color="#9999CC", linestyle="--", label="Slope Approximation")

    # Add vertical and horizontal reference lines
    plt.axvline(x=0, color="#66CC99", linestyle="--", label="h = 0")
    plt.axhline(y=psi_Pi_zero, color="#66CC99", linestyle="--", label=r"$\Psi(\Pi_0)$")

    # Add labels and title
    plt.xlabel("h")
    plt.ylabel(r"$\Psi(\Pi_h)$")
    plt.title("Numerical Illustration of Psi(Pi_h)")
    plt.legend()
    plt.show()
    plt.close('all')

# 3.2 Yet another experiment 
# Practice Problem 1
def gen_features(another_experiment, w=np.linspace(0, 1, int(1e3)), h=0.):
    features = pd.DataFrame({"w": w})
    features["Q1w"] = another_experiment.Qbar(np.column_stack((np.ones(len(w)), w)), h)
    features["Q0w"] = another_experiment.Qbar(np.column_stack((np.zeros(len(w)), w)), h)
    features["blip_Qw"] = features["Q1w"] - features["Q0w"]

    # Manipulate the data
    features_long = (
        features.rename(columns={"Q1w": "Q(1,.)", "Q0w": "Q(0,.)", "blip_Qw": "Q(1,.) - Q(0,.)"})
                .melt(id_vars=["w"], var_name="f", value_name="value")
    )

    return features_long

def plot_curves(another_experiment, h_list=[-0.5, 0., 0.5]):
    for h in h_list:
        features_long = gen_features(another_experiment, h=h)
        # Plot the data
        plt.figure(figsize=(10, 6))
        for label, df in features_long.groupby("f"):
            plt.plot(df["w"], df["value"], label=label, linewidth=1)

        plt.xlabel("w")
        plt.ylabel("f(w)")
        plt.title(r"Visualizing $\bar{Q}_0$"+ f" for h={h}")
        plt.ylim(None, 1)
        plt.legend(title="Features")
        plt.show()
    plt.close('all')

# Practice Problem 2
class YetAnotherExperiment(AnotherExperiment):
    def __init__(self, name="YetAnotherExperiment"):
        super().__init__(name)

    def Qbar(self, AW, h):
        """
        Altered Qbar function for YetAnotherExperiment.
        """
        A = AW[:, 0]
        W = AW[:, 1]
        # Compute the altered Qbar
        return expit(
            logit(A * W + (1 - A) * W**2) +
            h * (2 * A - 1) / np.where(
                A == 1,
                np.sin((1 + W) * np.pi / 6),
                1 - np.sin((1 + W) * np.pi / 6)
            ) * (W - A * W + (1 - A) * W**2)
        )

# 3.3 More on fluctuations and smoothness
def compute_eic(experiment, obs, psi, **kwargs):
    """
    Compute the efficient influence curve (EIC) for given observations.

    Parameters:
        experiemnt: An instance of the Experiment class.
        obs: DataFrame containing columns "W", "A", and "Y".
        psi: Value of Psi at the law.

    Returns:
        eic_values: Computed EIC values.
    """
    W = obs[:, 0]  # Column 0 corresponds to "W"
    A = obs[:, 1]  # Column 1 corresponds to "A"
    Y = obs[:, 2]  # Column 2 corresponds to "Y"

    # Compute Qbar values
    QAW = experiment.Qbar(np.column_stack((A, W)), **kwargs)
    QoneW = experiment.Qbar(np.column_stack((np.ones(len(W)), W)), **kwargs)
    QzeroW = experiment.Qbar(np.column_stack((np.zeros(len(W)), W)), **kwargs)

    # Compute Gbar values
    GW = experiment.Gbar(W)

    # Compute lGAW
    lGAW = A * GW + (1 - A) * (1 - GW)

    # Compute EIC
    eic_values = (QoneW - QzeroW - psi) + (2 * A - 1) / lGAW * (Y - QAW)
    return eic_values

# visulization of the EIC
def visualize_eic(experiment, psi):
    # Step 1: Cartesian product like `crossing(w, a, y)`
    w_values = np.linspace(0, 1, 200)
    y_values = np.linspace(0, 1, 200)
    a_values = [0, 1]

    grid = pd.DataFrame([
        (w, a, y) for a in a_values for w in w_values for y in y_values
    ], columns=["W", "A", "Y"])

    # Step 2: Compute EIC with the correct input order: W, A, Y
    obs = grid[["W", "A", "Y"]].values
    grid["eic"] = compute_eic(experiment, obs, psi)

    # Step 3: Plot with matplotlib and contour, faceted by a
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for i, a in enumerate(a_values):
        subset = grid[grid["A"] == a]
        W = subset["W"].values.reshape(200, 200)
        Y = subset["Y"].values.reshape(200, 200)
        EIC = subset["eic"].values.reshape(200, 200)

        ax = axes[i]
        im = ax.imshow(
            EIC, extent=(0, 1, 0, 1), origin="lower", cmap="coolwarm", aspect='auto', interpolation="bilinear"
        )
        cs = ax.contour(W, Y, EIC, levels=10, colors="white", linewidths=0.5)

        ax.set_title(f"a = {a}", fontsize=14)
        ax.set_xlabel("W")
        if i == 0:
            ax.set_ylabel("Y")
        else:
            ax.set_yticklabels([])

        ax.set_xticks(np.round(np.linspace(0, 1, 6), 2))
        ax.set_yticks(np.round(np.linspace(0, 1, 6), 2))
        ax.tick_params(labelsize=10)

    # fig.colorbar(im, ax=axes.ravel().tolist(), label=r"$D^*(P_0)(w,a,y)$", location='right')
    # Add colorbar manually on the right
    cbar_ax = fig.add_axes([0.99, 0.15, 0.015, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label(r"$D^*(P_0)(w,a,y)$")
    fig.suptitle(r"Efficient Influence Curve $D^*(P_0)(w,a,y)$", fontsize=16)
    plt.tight_layout()
    plt.show()

# 3.4.2 Numerical validation
def sigma0(obs, law, **kwargs):
    """
    Compute the sigma0 values for the given observations and law.

    Parameters:
        obs: NumPy array or pandas DataFrame containing columns "Y", "A", and "W".
        law: An instance of a LAW subclass (e.g., AnotherExperiment).

    Returns:
        sigma0_values: Computed sigma0 values.
    """
    # Access Qbar and qY functions from the law
    Qbar = law.Qbar
    qY_params = {"shape1": 4}  # Assuming shape1 is fixed; replace with dynamic retrieval if needed

    # Compute QAW
    QAW = Qbar(np.column_stack((obs[:, 1], obs[:, 0])), **kwargs)  # A and W columns

    # Retrieve shape1 parameter
    shape1 = qY_params["shape1"]

    # Compute betaAW
    betaAW = shape1 * (1 - QAW) / QAW

    # Compute sigma0 values
    sigma0_values = np.log(1 - obs[:, 2])  # Column 0 corresponds to "Y"
    for i in range(1, shape1 + 1):
        sigma0_values += 1 / (i - 1 + betaAW)

    sigma0_values = -sigma0_values * shape1 * (1 - QAW) / QAW * 10 * np.sqrt(obs[:, 0]) * obs[:, 1]  # W and A columns

    return sigma0_values

# 3.6 Cramér-Rao bounds
# What does the following chunk do?
def cramer_rao_bound(experiment, B, **kwargs):
    """
    Compute the Cramér-Rao bound for the given experiment.

    Parameters:
        experiment: An instance of a LAW subclass (e.g., AnotherExperiment).
        B: Number of samples to generate.

    Returns:
        cramer_rao_hat: Estimated Cramér-Rao bound.
    """
    # Generate observations
    obs = experiment.sample_from(B, **kwargs)

    # Compute the efficient influence curve (EIC) for the observations
    psi = evaluate_psi(experiment, **kwargs)
    eic_values = compute_eic(experiment, obs, psi, **kwargs)

    # Compute the variance of the EIC
    cramer_rao_hat = np.var(eic_values)

    return cramer_rao_hat


# Example usage
if __name__ == "__main__":
    # 3.1.2 Numerical illustration
    another_experiment = AnotherExperiment()
    data, psi_Pi_zero, slope_approx = gen_data(another_experiment)
    plot_data(data, psi_Pi_zero, slope_approx)

    # 3.2 Yet another experiment 
    # Practice Problem 1
    plot_curves(another_experiment)

    # Practice Problem 2
    yet_another_experiment = YetAnotherExperiment()
    plot_curves(yet_another_experiment, h_list=[-0.5, 0., 0.5])

    # Practice Problem 3
    data, psi_Pi_zero, slope_approx = gen_data(yet_another_experiment)
    plot_data(data, psi_Pi_zero, slope_approx)

    # 3.3 More on fluctuations and smoothness
    # evalute_eic PS. you need to evalute_psi first, before you can get eic
    experiment = Experiment()
    psi = evaluate_psi(experiment)
    obs = experiment.sample_from(n=5)
    eic_values = compute_eic(experiment, obs, psi)
    print(eic_values)
    
    # Visualize the EIC
    visualize_eic(experiment, psi)

    # 3.4.2 Numerical validation
    B=int(1e6)
    obs_another_experiment = another_experiment.sample_from(n=B, h=0)
    psi_another_experiment = evaluate_psi(another_experiment, h=0)
    eic_another_experiment = compute_eic(another_experiment, obs_another_experiment, psi_another_experiment, h=0)
    vars = eic_another_experiment * sigma0(obs_another_experiment, another_experiment, h=0)
    sd_hat = np.std(vars)
    slope_hat = np.mean(vars)
    print(f"Slope Estimate: {slope_hat}")
    alpha = 0.05
    slope_CI = slope_hat + np.array([-1, 1]) * np.sqrt(alpha) * sd_hat / np.sqrt(B)
    print(f"Slope Confidence Interval: {slope_CI}")

    # 3.6 Cramér-Rao bounds
    # Practice Problem 1. What does the following chunk do?
    cramer_rao_hat = cramer_rao_bound(experiment, B)
    print(f"Cramér-Rao bound: {cramer_rao_hat}")

    # Practice Problem 2. Same question about this one.
    cramer_rao_hat_another = cramer_rao_bound(another_experiment, B, h=0)
    print(f"Cramér-Rao bound for AnotherExperiment: {cramer_rao_hat_another}")

    # Practice Problem 3.
    '''
    No.The Cramér-Rao bound represents the lowest possible asymptotic variance of 
    any regular, asymptotically linear estimator. So, no regular estimator can beat it.
    '''

    # Practice Problem 4.
    '''We define the ratio between Cramér-Rao bounds of P_0 and \Pi_0 as below
    to evaluate the difficulty in estimate Psi(\Pi_0) compared to Psi(P_0). 
    A lower variance means the estimation is easier, while a higher variance means it is harder.
    Answer: \Pi_0 is easier to estimate than P_0 if the ratio is less than 1.  
    '''
    ratio = cramer_rao_hat / cramer_rao_hat_another
    print(f"Ratio of Cramér-Rao bounds: {ratio}")