import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import expit, logit
from scipy.stats import norm

from python_section_1 import LAW, Experiment
from python_section_2 import AnotherExperiment
from python_section_3 import evaluate_psi, compute_eic
from python_algorithms import WorkingModelGOne, WorkingModelGTwo, WorkingModelGThree, WorkingModelQOne, KknnAlgo, BoostingTreeAlgo, BoostingLMAlgo

# 7.3 QW
def estimate_QW(dat):
    """
    Estimate QW by transforming the input data.

    Parameters:
        dat: pandas DataFrame containing a column "W".

    Returns:
        result: pandas DataFrame with columns "value" and "weight".
    """
    # Convert to DataFrame if necessary
    if isinstance(dat, np.ndarray):
        dat = pd.DataFrame(dat, columns=["W", "A", "Y"])

    # Select "W" column and compute weights
    result = dat[["W"]].rename(columns={"W": "value"})
    result["weight"] = 1 / len(dat)

    return result

class EmpiricalExperiment(LAW):
    def __init__(self, dat):
        """
        Initialize the empirical experiment with data.

        Parameters:
            dat: pandas DataFrame containing a column "W".
        """
        super().__init__(name="EmpiricalExperiment")
        self.dat = dat

    def QW(self, *args, **kwargs):
        """
        Estimate Q(W) using the provided data.
        """
        return estimate_QW(self.dat)

    def sample_from(self, n):
        """
        Sample data from the empirical experiment.

        Parameters:
            n: Number of samples to generate.

        Returns:
            sampled_data: pandas DataFrame containing sampled W values.
        """
        QW = self.QW()
        W = np.random.choice(
            QW["value"], size=n, p=QW["weight"]
        )
        sampled_data = pd.DataFrame({"W": W, "A": np.nan, "Y": np.nan})
        return sampled_data



def plot_histogram_with_true_density(W, experiment):
    """
    Plot histogram of sampled W values with the true density function overlay.

    Parameters:
        W: pandas DataFrame containing sampled W values.
        experiment: An instance of the Experiment class.

    Returns:
        None. Displays the plot.
    """
    plt.figure(figsize=(10, 6))

    # Plot histogram
    sns.histplot(W["W"], bins=40, stat="density", kde=False, label="Sampled W")

    # Overlay true density function
    x = np.linspace(W["W"].min(), W["W"].max(), 1000)
    true_density = experiment.QW(x)
    plt.plot(x, true_density, color="red", label="True Density (QW)")

    # Add labels and legend
    plt.xlabel("W")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


# 7.4 Gbar
def estimate_Gbar(dat, algorithm, formula=None, **kwargs):
    """
    Estimate Gbar using the provided algorithm.

    Parameters:
        dat: pandas DataFrame or NumPy array containing the data.
        algorithm: A tuple or list where:
            - algorithm[0] is the model fitting function.
            - algorithm[1] is the formula or configuration for the model.
            - algorithm['ML'] (optional) indicates if the algorithm is machine learning-based.
        **kwargs: Additional arguments to pass to the algorithm.

    Returns:
        fit: The fitted model with an added attribute 'type_of_preds'.
    """
    # Check if the algorithm is machine learning-based
    if algorithm.ML == False:
        # Use the formula-based fitting function
        model = algorithm.fit(dat=dat, formula=formula, **kwargs)
    else:
        # Use the machine learning-based fitting function
        model = algorithm.fit(dat=dat, **kwargs)

    # TODO: Add an attribute to the fitted model
    return model # return the fitted model directly

def compute_Gbar_hatW(W, Gbar_hat, threshold=0.05):
    """
    Compute Gbar_hat(W) with thresholding.

    Parameters:
        W: NumPy array or pandas Series of W values.
        Gbar_hat: Fitted model object with a predict method.
        threshold: Numeric value for thresholding (default is 0.05).

    Returns:
        pred: Thresholded predictions as a NumPy array.
    """
    # Ensure threshold is within the valid range
    if not (0 <= threshold <= 0.5):
        raise ValueError("Threshold must be between 0 and 0.5.")

    # Create a DataFrame for prediction
    dat = pd.DataFrame({"W": W})

    # Predict using the fitted model
    GW = Gbar_hat.predict(dat)

    # Handle matrix-like predictions
    if isinstance(GW, (np.ndarray, pd.DataFrame)):
        if GW.shape[1] == 2:
            GW = GW[:, 1]  # Use the second column
        else:
            raise ValueError("Object 'GW' is neither a vector nor a two-column matrix.")

    # Ensure GW is a vector
    if not isinstance(GW, (np.ndarray, pd.Series)):
        raise ValueError("Object 'GW' is neither a vector nor a two-column matrix.")

    # Apply thresholding
    pred = np.minimum(1 - threshold, np.maximum(GW, threshold))

    return pred

def visualize_Gbar(experiment, Gbar_hat):
    # Generate data
    w = np.linspace(0, 1, int(1e3))  # Equivalent to seq(0, 1, length.out = 1e3)
    truth = experiment.Gbar(w)  # True values
    estimated = compute_Gbar_hatW(w, Gbar_hat)  # Estimated values

    # Create a DataFrame
    df = pd.DataFrame({
        "w": w,
        "truth": truth,
        "estimated": estimated
    })

    # Reshape the DataFrame to long format
    df_long = df.melt(id_vars="w", var_name="f", value_name="value")

    # Plot using seaborn
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_long, x="w", y="value", hue="f", linewidth=1)
    plt.ylim(None, 1)
    plt.xlabel("w")
    plt.ylabel("f(w)")
    plt.title(r"Visualizing $\bar{G}_0$ and $\hat{G}_n$")
    plt.legend(title="Function")
    plt.show()

# 7.5 Qbar 
def print_summary(model): #TODO
    """
    Print a summary of the fitted model.

    Parameters:
        model: The fitted model object.

    Returns:
        None. Prints the summary to the console.
    """
    if hasattr(model, "summary"):
        print(model.summary())
    else:
        params = model.get_params()
        for key, value in params.items():
            print(f"{key}: {value}")

def estimate_Qbar(dat, algorithm, formula=None, verbose=False, **kwargs):
    """
    Estimate Qbar using the provided algorithm.

    Parameters:
        dat: pandas DataFrame containing the data.
        algorithm: An object representing the algorithm with attributes 'ML' and 'stratify'.
        **kwargs: Additional arguments to pass to the algorithm.

    Returns:
        A pandas DataFrame with the fitted models for 'both', 'one', and 'zero'.
    """
    if not isinstance(dat, pd.DataFrame):
        dat = pd.DataFrame(dat,  columns=["W", "A", "Y"])

    if not getattr(algorithm, "ML", False):  # Non-ML algorithm
        if not getattr(algorithm, "stratify", False):  # No stratification
            fit_both = algorithm.fit(dat=dat, formula=formula, **kwargs)
            fit_one = None
            fit_zero = None
        else:  # Stratification required
            idx_one = dat["A"] == 1
            if idx_one.sum() in [0, len(dat)]:
                raise ValueError("Impossible to stratify.")
            fit_both = None
            fit_one = algorithm.fit(data=dat[idx_one], formula=formula, **kwargs)
            fit_zero = algorithm.fit(data=dat[~idx_one], formula=formula, **kwargs)
    else:  # ML algorithm
        if not getattr(algorithm, "stratify", False):  # No stratification
            fit_both = algorithm.fit(dat=dat, **kwargs)
            fit_one = None
            fit_zero = None
        else:  # Stratification required
            idx_one = dat["A"] == 1
            if idx_one.sum() in [0, len(dat)]:
                raise ValueError("Impossible to stratify.")
            fit_both = None
            fit_one = algorithm.fit(dat=dat[idx_one], **kwargs)
            fit_zero = algorithm.fit(dat=dat[~idx_one], **kwargs)
    # If verbose is True, print summaries of the fitted models
    if verbose:
        if fit_both is not None:
            print("Model summary for both treatments:")
            print_summary(fit_both)
        if fit_one is not None:
            print("Model summary for A=1 treatment:")
            print_summary(fit_one)
        if fit_zero is not None:
            print("Model summary for A=0 treatment:")
            print_summary(fit_zero)

    # Create a DataFrame to store the results
    fit = pd.DataFrame({
        "a": ["both", "one", "zero"],
        "fit": [fit_both, fit_one, fit_zero]
    })

    # Add attributes to the DataFrame
    fit.attrs["type_of_preds"] = getattr(algorithm, "type_of_preds", None)
    fit.attrs["stratify"] = getattr(algorithm, "stratify", False)

    return fit


def compute_Qbar_hatAW(A, W, Qbar_hat, algorithm, blip=False):
    """
    Compute predictions based on Qbar_hat with optional blip computation.

    Parameters:
        A: Array-like, treatment variable.
        W: Array-like, covariates.
        Qbar_hat: pandas DataFrame containing fitted models and attributes.
        blip: Boolean, whether to compute the blip (difference between A=1 and A=0 predictions).

    Returns:
        pred: Array of predictions.
    """
    stratify = Qbar_hat.attrs.get("stratify", False)  # Check if stratification is required
    type_of_preds = Qbar_hat.attrs.get("type_of_preds", None)  # Type of predictions

    if not blip:
        # Create a DataFrame for prediction
        dat = pd.DataFrame({"W": W, "A": A, "Y": np.nan})

        if not stratify:
            # Use the "both" model for prediction
            fit = Qbar_hat.loc[Qbar_hat["a"] == "both", "fit"].iloc[0]
            pred = algorithm.predict(dat=dat, model=fit) 
        else:
            # Use stratified models for prediction
            fit_one = Qbar_hat.loc[Qbar_hat["a"] == "one", "fit"].iloc[0]
            fit_zero = Qbar_hat.loc[Qbar_hat["a"] == "zero", "fit"].iloc[0]
            pred = np.zeros(len(dat))
            idx_one = dat["A"] == 1

            if idx_one.sum() > 0:
                pred[idx_one] = algorithm.predict(dat=dat[idx_one], model=fit_one)
            if (~idx_one).sum() > 0:
                pred[~idx_one] = algorithm.predict(dat=dat[~idx_one], model=fit_zero)
    else:
        if not stratify:
            # Use the "both" model for blip computation
            fit = Qbar_hat.loc[Qbar_hat["a"] == "both", "fit"].iloc[0]
            pred = algorithm.predict(pd.DataFrame({"W": W, "A": 1, "Y": np.nan}), model=fit) - \
                    algorithm.predict(pd.DataFrame({"W": W, "A": 0, "Y": np.nan}), model=fit)
        else:
            # Use stratified models for blip computation
            fit_one = Qbar_hat.loc[Qbar_hat["a"] == "one", "fit"].iloc[0]
            fit_zero = Qbar_hat.loc[Qbar_hat["a"] == "zero", "fit"].iloc[0]
            pred = algorithm.predict(pd.DataFrame({"W": W, "A": 1, "Y": np.nan}), model=fit_one) - \
                    algorithm.predict(pd.DataFrame({"W": W, "A": 0, "Y": np.nan}), model=fit_zero)
    return pred


def post_process_tree_data(fig, pred_1, pred_2):
    """
    Process the input DataFrame `fig`, compute predictions using Qbar_hat_trees,
    and plot the results.

    Parameters:
        fig (pd.DataFrame): Input DataFrame containing the `w` column.
        Qbar_hat_trees (pd.DataFrame): DataFrame containing fitted models for "one" and "zero".
        algorithm (object): Algorithm object with a `predict` method.
        verbose (bool): Whether to print model summaries.

    Returns:
        pd.DataFrame: The processed DataFrame with predictions added.
    """
    # Add predictions for trees_1 and trees_0
    fig["trees_1"] = pred_1
    fig["trees_0"] = pred_2

    # Reshape the DataFrame to long format
    fig_long = fig.melt(id_vars="w", var_name="f", value_name="value")

    # Extract "f" and "a" from the column names (e.g., "truth_1" -> "truth", "1")
    fig_long[["f", "a"]] = fig_long["f"].str.extract(r"([^_]+)_([01]+)")
    fig_long["a"] = "a=" + fig_long["a"]  # Add "a=" prefix to the "a" column

    return fig_long

def plot_fig(fig_long):
    """
    Plot the processed DataFrame `fig_long` with two subplots:
    one for a=1 and the other for a=0.

    Parameters:
        fig_long (pd.DataFrame): Long-format DataFrame with columns `w`, `value`, `f`, and `a`.

    Returns:
        None
    """

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    # Iterate over the unique values of "a" (e.g., "a=1" and "a=0")
    for i, a_value in enumerate(fig_long["a"].unique()):
        ax = axes[i]
        subset = fig_long[fig_long["a"] == a_value]  # Filter data for the current "a"
        sns.lineplot(data=subset, x="w", y="value", hue="f", ax=ax, linewidth=1)
        ax.set_title(f"Visualizing for {a_value}")
        ax.set_xlabel("w")
        ax.set_ylabel("f(w)" if i == 0 else "")  # Only set y-label for the first subplot
        ax.set_ylim(0, 1)
        ax.legend(title="Function")

    # Adjust layout
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    # Example usage of the functions defined in this script
    # This section can be used for testing or demonstration purposes

    # initialize an experiment
    experiment = Experiment()
    B=int(1e6)
    obs = experiment.sample_from(n=B)
    
    # 7.3 QW
    QW = estimate_QW(obs[:int(1e3)])
    print(QW)

    # Create empirical experiment
    empirical_experiment = EmpiricalExperiment(obs[:int(1e3)])
    W = empirical_experiment.sample_from(1000)
    plot_histogram_with_true_density(W, experiment) 

    len_intersect = len(set(W["W"]).intersection(set(obs[:1000, 0]))) # column 0 is W
    print(len_intersect)

    # 7.4 Gbar
    working_model_G_one = WorkingModelGOne()
    Gbar_hat = estimate_Gbar(dat=obs[:int(1e3)], algorithm=working_model_G_one)
    # pred = compute_Gbar_hatW(np.linspace(0, 1, int(1e3)), Gbar_hat)
    visualize_Gbar(experiment, Gbar_hat)

    # 7.6.2 Qbar, kNN algorithm
    kknn_algo = KknnAlgo()
    Qbar_hat_kknn = estimate_Qbar(dat=obs[:int(1e3)], algorithm=kknn_algo)

    # Generate data
    w = np.linspace(0, 1, int(1e3))  # Equivalent to seq(0, 1, length.out = 1e3)
    
    # Create a DataFrame
    fig = pd.DataFrame({
        "w": w,
        "truth_1": experiment.Qbar(np.column_stack((np.ones_like(w), w))),  # True values for A=1
        "truth_0": experiment.Qbar(np.column_stack((np.zeros_like(w), w))),  # True values for A=0
        "kNN_1": compute_Qbar_hatAW(A=np.ones_like(w), W=w, Qbar_hat=Qbar_hat_kknn, algorithm=kknn_algo),  # Estimated values for A=1
        "kNN_0": compute_Qbar_hatAW(A=np.zeros_like(w), W=w, Qbar_hat=Qbar_hat_kknn, algorithm=kknn_algo)   # Estimated values for A=0
    })
    
    # Display the DataFrame
    print(fig)


    # 7.6.3 Qbar, boosted trees algorithm
    boosting_tree_algo = BoostingTreeAlgo()
    boosting_tree_algo.control["method"] = "None"
    Qbar_hat_trees = estimate_Qbar(dat=obs[:int(1e3)], algorithm=boosting_tree_algo, verbose=True)
    pred_1 = compute_Qbar_hatAW(A=np.ones_like(w), W=w, Qbar_hat=Qbar_hat_trees, algorithm=boosting_tree_algo)
    pred_2 = compute_Qbar_hatAW(A=np.zeros_like(w), W=w, Qbar_hat=Qbar_hat_trees, algorithm=boosting_tree_algo)     
    fig_long = post_process_tree_data(fig, pred_1, pred_2)
    # Plot the results
    plot_fig(fig_long)




