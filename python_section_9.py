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


# 9.3 Empirical investigation
def eic(obs, Gbar_hat, Qbar_hat, psi, 
        Gbar_use_predict=False, Qbar_use_predict=False,
        Gbar_algorithm=None, Qbar_algorithm=None):
    """
    Compute the efficient influence curve (EIC) for the given observations.

    Args:
        obs (pd.DataFrame): Input data containing columns 'W', 'A', and 'Y'.
        Gbar (callable): Function to estimate the marginal law of 'A' given 'W'.
        Qbar (callable): Function to estimate the conditional expectation of 'Y' given 'A' and 'W'.
        psi (float): Initial estimate of psi.

    Returns:
        np.ndarray: The efficient influence curve values.
    """
    # Check if required columns are present
    required_columns = {"W", "A", "Y"}
    if not required_columns.issubset(obs.columns):
        raise ValueError(f"Argument 'obs' must contain columns {required_columns}.")

    # Compute Qbar values
    if Qbar_use_predict:# If using a prediction model, apply it to compute Qbar_hat     
        stratify = Qbar_hat.attrs.get("stratify", False)  # Check if stratification is required
        if not stratify:
            fit = Qbar_hat.loc[Qbar_hat["a"] == "both", "fit"].iloc[0]
            if Qbar_algorithm is not None:
                QAW = Qbar_algorithm.predict(pd.DataFrame({"W": obs["W"], "A": obs["A"], "Y": np.nan}), model=fit)
                QoneW = Qbar_algorithm.predict(pd.DataFrame({"W": obs["W"], "A": 1, "Y": np.nan}), model=fit)
                QzeroW = Qbar_algorithm.predict(pd.DataFrame({"W": obs["W"], "A": 0, "Y": np.nan}), model=fit)
            else:
                QAW = fit.predict(pd.DataFrame({"W": obs["W"], "A": obs["A"], "Y": np.nan}))
                QoneW = fit.predict(pd.DataFrame({"W": obs["W"], "A": 1, "Y": np.nan}))
                QzeroW = fit.predict(pd.DataFrame({"W": obs["W"], "A": 0, "Y": np.nan}))
        else:
            fit_one = Qbar_hat.loc[Qbar_hat["a"] == "one", "fit"].iloc[0]
            fit_zero = Qbar_hat.loc[Qbar_hat["a"] == "zero", "fit"].iloc[0]
            if Qbar_algorithm is not None: # if a ML model is used
                QAW = Qbar_algorithm.predict(pd.DataFrame({"W": obs["W"], "A": obs["A"], "Y": np.nan}), model=fit_one)
                QoneW = Qbar_algorithm.predict(pd.DataFrame({"W": obs["W"], "A": 1, "Y": np.nan}), model=fit_one)
                QzeroW = Qbar_algorithm.predict(pd.DataFrame({"W": obs["W"], "A": 0, "Y": np.nan}), model=fit_zero)
            else:
                # if a non-ML model is used
                QAW = fit_one.predict(pd.DataFrame({"W": obs["W"], "A": obs["A"], "Y": np.nan}))
                QoneW = fit_one.predict(pd.DataFrame({"W": obs["W"], "A": 1, "Y": np.nan}))
                QzeroW = fit_zero.predict(pd.DataFrame({"W": obs["W"], "A": 0, "Y": np.nan}))
    else: # if the true function is used, e.g. experiment.Qbar()
        QAW = Qbar_hat(obs[["A", "W"]])
        QoneW = Qbar_hat(pd.DataFrame({"A": 1, "W": obs["W"]}))
        QzeroW = Qbar_hat(pd.DataFrame({"A": 0, "W": obs["W"]}))


    # Compute Gbar values
    if Gbar_use_predict: # If using a prediction model, apply it to compute Gbar_hat
        GW = Gbar_hat.predict(pd.DataFrame({"W": obs["W"]}))
    else: # if the true function is used, e.g. experiment.Gbar()
        GW = Gbar_hat(obs[["W"]])
    lGAW = obs["A"] * GW + (1 - obs["A"]) * (1 - GW)

    # Compute the efficient influence curve
    out = (QoneW - QzeroW - psi) + (2 * obs["A"] - 1) / lGAW * (obs["Y"] - QAW)
    return out.values


def apply_one_step_correction(dat, Gbar, Qbar, psi, 
                              Gbar_use_predict=False, Qbar_use_predict=False,
                              Gbar_algorithm=None, Qbar_algorithm=None):
    """
    Apply one-step correction to estimate psi_n and sig_n based on the efficient influence curve (EIC).

    Args:
        dat (pd.DataFrame): Input data containing columns 'W', 'A', and 'Y'.
        Gbar (callable): Function to estimate the marginal law of 'A' given 'W'.
        Qbar (callable): Function to estimate the conditional expectation of 'Y' given 'A' and 'W'.
        psi (float): Initial estimate of psi.

    Returns:
        pd.DataFrame: A DataFrame containing psi_n, sig_n, and crit_n.
    """
    # Compute EIC for the dataset
    eic_dat = eic(dat, Gbar, Qbar, psi, Gbar_use_predict=Gbar_use_predict, Qbar_use_predict=Qbar_use_predict,
                                        Gbar_algorithm=Gbar_algorithm, Qbar_algorithm=Qbar_algorithm)

    # Compute psi_n, sig_n, and crit_n
    psi_n = psi + np.mean(eic_dat)
    sig_n = np.std(eic_dat) / np.sqrt(len(dat))
    crit_n = np.mean(eic_dat)

    # Return results as a DataFrame
    return pd.DataFrame({"psi_n": [psi_n], "sig_n": [sig_n], "crit_n": [crit_n]})

# learned_features_fixed_sample_size reinvented because  Gbar_hat is added 

def update_learned_features_one_step(learned_features, 
                                     algorithm_d=None, algorithm_e=None):
    for key, entry in learned_features.items():
        entry["Qbar_hat_d"] = estimate_Qbar(entry["obs"], algorithm=algorithm_d)
        entry["Qbar_hat_e"] = estimate_Qbar(entry["obs"], algorithm=algorithm_e)
        entry["os_est_d"] = apply_one_step_correction(entry['obs'], 
                                                        entry['Gbar_hat'], 
                                                        entry['Qbar_hat_d'], 
                                                        entry['est_d']['psi_n'][0],
                                                        Gbar_use_predict=True, 
                                                        Qbar_use_predict=True,
                                                        Gbar_algorithm=None, 
                                                        Qbar_algorithm=None) # This is a non-ML model
        entry["os_est_e"] = apply_one_step_correction(entry['obs'],
                                                        entry['Gbar_hat'], 
                                                        entry['Qbar_hat_e'], 
                                                        entry['est_e']['psi_n'][0],
                                                        Gbar_use_predict=True, 
                                                        Qbar_use_predict=True,
                                                        Gbar_algorithm=None, 
                                                        Qbar_algorithm=algorithm_e) # this is a ML model
    return learned_features

def get_psi_hat_de_one_step(updated_features, psi_zero):
    # Assuming learned_features_fixed_sample_size is a pandas DataFrame
    psi_hat_de = []
    for key, entry in updated_features.items():
        os_est_d = entry["os_est_d"]
        psi_hat_de.append({
            "id": key,
            "psi_n": os_est_d["psi_n"].values[0],  # Assuming est_d is a DataFrame
            "sig_n": os_est_d["sig_n"].values[0],  # Assuming est_d is a DataFrame
            "type": "d_one_step"
        })

        os_est_d = entry["os_est_d"]
        psi_hat_de.append({
            "id": key,
            "psi_n": os_est_d["psi_n"].values[0],  # Assuming est_e is a DataFrame
            "sig_n": os_est_d["sig_n"].values[0],  # Assuming est_e is a DataFrame
            "type": "e_one_step"
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


def plot_estimation_bias_de_os(psi_hat_de_df, bias_de):    
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
            ax[id_norm].set_xlim(-4, 4)
            ax[id_norm].set_ylim(0, 0.5)
            ax[id_norm].legend(title="Type", loc='upper right')
        
    fig.suptitle(r"$\sqrt{n/v_n^{d, e, os}} * (\psi_n^{d, e, os} - \psi_0)$", fontsize=12)
    plt.show()

# Practice Problem 9.4.1
# def estimate_Gbar_oracle(obs_df, Gbar_truth, s):
#     G_0 = Gbar_truth(obs_df['W'])
#     Z = np.random.normal(size=G_0.shape)
#     return expit(logit(G_0) + s * Z)

class OracleGbarModel(WorkingModelBase):
    def __init__(self, G0_fn, s=0.0, seed=42):
        """
        Oracle estimator for Gbar using a known G0(w) function and noise level s.

        Parameters:
            G0_fn: A function G0(w) returning the true P(A=1 | W=w) as a numpy array.
            s: Scalar for noise level (i.e., how much standard normal noise to add to logit).
            seed: Optional random seed for reproducibility.
        """
        super().__init__()
        self.G0_fn = G0_fn
        self.s = s
        self.seed = seed
        self.fitted_values = None  # Save values after "fit"

    def fit(self, dat, formula=None, **kwargs):
        """
        Generates perturbed probabilities from known G0(w).

        Parameters:
            dat: pandas DataFrame with at least column 'W'

        Returns:
            self (for chaining)
        """
        if isinstance(dat, np.ndarray):
            dat = pd.DataFrame(dat, columns=["W", "A", "Y"])
        
        if self.seed is not None:
            np.random.seed(self.seed)

        # Compute G0(w)
        w = dat["W"].values
        G0_vals = self.G0_fn(w)

        # Apply oracle perturbation: expit(logit(G0(w)) + sZ)
        Z = np.random.normal(size=len(w))
        perturbed_logits = logit(G0_vals) + self.s * Z
        self.fitted_values = expit(perturbed_logits)

        return self

    def predict(self, W, **kwargs):
        """
        Predict using the oracle model.

        Parameters:
            W: numpy.array of column 'W'.

        Returns:
            Numpy array of predicted probabilities.
        """
        if isinstance(W, pd.DataFrame):
            w = W["W"].values
        else:
            w = W
        if self.seed is not None:
            np.random.seed(self.seed)

        G0_vals = self.G0_fn(w)
        Z = np.random.normal(size=len(w))
        perturbed_logits = logit(G0_vals) + self.s * Z
        return expit(perturbed_logits)


# Practice Problem 9.4.2
class OracleQbarModel(WorkingModelBase):
    def __init__(self, Qbar_truth_fn, s=0.0, seed=42):
        """
        Oracle estimator for Qbar using a known Q0(a, w) and noise level s.

        Parameters:
            Qbar_truth_fn: Function that takes np.array([[A, W], ...]) and returns true Q0(A, W)
            s: Scalar controlling noise level (standard deviation in logit space)
            seed: Optional random seed for reproducibility
        """
        super().__init__()
        self.Qbar_truth_fn = Qbar_truth_fn
        self.s = s
        self.seed = seed
        self.fitted_values = None

    def fit(self, dat, formula=None, **kwargs):
        """
        Generate perturbed Qbar values based on known Q0(A, W)

        Parameters:
            dat: DataFrame with columns ['A', 'W']

        Returns:
            self
        """
        if isinstance(dat, np.ndarray):
            dat = pd.DataFrame(dat, columns=["W", "A", "Y"])

        if self.seed is not None:
            np.random.seed(self.seed)

        a_w = dat[['A', 'W']].to_numpy()
        Q0_vals = self.Qbar_truth_fn(a_w)

        Z = np.random.normal(size=Q0_vals.shape)
        self.fitted_values = expit(logit(Q0_vals) + self.s * Z)

        return self

    def predict(self, newdata=None, **kwargs):
        """
        Predict perturbed Qbar values for new (A, W) input

        Parameters:
            newdata: DataFrame with columns ['A', 'W']

        Returns:
            Numpy array of predicted Qbar values
        """
        if newdata is None:
            return self.fitted_values

        if isinstance(newdata, np.ndarray):
            newdata = pd.DataFrame(newdata, columns=["W", "A", "Y"])

        if self.seed is not None:
            np.random.seed(self.seed)

        a_w = newdata[['A', 'W']].to_numpy()
        Q0_vals = self.Qbar_truth_fn(a_w)
        Z = np.random.normal(size=Q0_vals.shape)
        return expit(logit(Q0_vals) + self.s * Z)


# def estimate_Qbar_oracle(obs_df, Qbar_truth, s):
#     Q_0 = Qbar_truth(obs_df[['A', 'W']].to_numpy())
#     Z = np.random.normal(size=Q_0.shape)
#     return expit(logit(Q_0) + s * Z)

# Practice Problem 9.4.3

if __name__ == "__main__":
    # initialize an experiment
    experiment = Experiment()
    B=int(1e6)
    obs = experiment.sample_from(n=B)
    obs_df = pd.DataFrame(obs, columns=["W", "A", "Y"])

    # 9.3 Empirical investigation
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


    iter = 1000  # Number of iterations
    # working_model_G_one = WorkingModelGOne()
    learned_features_fixed_sample_size = get_learned_features(obs_df, iter, working_model_G_one)
    working_model_Q_one = WorkingModelQOne()
    updated_features = update_learned_features(learned_features_fixed_sample_size, algorithm_d=working_model_Q_one, algorithm_e=kknn_algo)
    # apply one step correction
    updated_features_one_step = update_learned_features_one_step(updated_features, 
                                                                 algorithm_d=working_model_Q_one, 
                                                                 algorithm_e=kknn_algo)
    psi_hat_de_os_df, bias_de_os = get_psi_hat_de_one_step(updated_features_one_step, psi_zero=evaluate_psi(experiment))
    print(bias_de_os)

    plot_estimation_bias_de_os(psi_hat_de_os_df, bias_de_os)

    # compare bias_de with bias_de_one_step
    psi_hat_de_df, bias_de = get_psi_hat_de(updated_features, psi_zero=evaluate_psi(experiment))
    print(bias_de)
    combined_bias = pd.concat([bias_de, bias_de_os], ignore_index=True)
    filtered_bias = combined_bias[~combined_bias["auto_renormalization"]]
    sorted_bias = filtered_bias.sort_values(by="type")
    print(sorted_bias)

    combined_psi_hat = pd.concat([psi_hat_de_df, psi_hat_de_os_df], ignore_index=True)
    filtered_psi_hat = combined_psi_hat[combined_psi_hat["auto_renormalization"]]
    summary_stats = (
        filtered_psi_hat.groupby("type")
        .apply(lambda group: pd.Series({
            "sd": group["sig_n"].mean(),
            "se": group["psi_n"].std(),
            "mse": ((group["psi_n"] - evaluate_psi(experiment)) ** 2).mean() * len(group)
        }))
        .reset_index()
        .sort_values(by="type")
    )
    print(summary_stats)

    # Practice Problem 9.4.1
    # if s=0
    gbar_hat_algo = OracleGbarModel(G0_fn=experiment.Gbar, s=0)
    Gbar_hat_s0 = gbar_hat_algo.fit(obs_df)
    Gbar_hat_s0.predict(obs_df['W'])

    # Gbar_hat_s0 = estimate_Gbar_oracle(obs_df, experiment.Gbar, s=0) 
    # print(Gbar_hat_s0)
    '''
    when s=0:
    You recover the true conditional probability exactly — no randomness.

    What happens as s → 0?
    You converge in probability to \bar{G}_0(w), because the added noise vanishes. 
    But if s > 0, even very small, you are adding a little noise in log-odds space, 
    and the result is a slightly randomized version of the true \bar{G}_0(w).

    Why is it called an “oracle algorithm”?
    Because it uses knowledge of the true conditional probability \bar{G}_0(w), 
    which you would not have in real data. That is, the procedure is not estimating \bar{G}_0(w) from data,
    but it is simulating what an estimator might look like if it had access to the true function, 
    and how it behaves when noise is injected.
    '''

    # Practice Problem 9.4.2
    qbar_hat_algo = OracleQbarModel(Qbar_truth_fn=experiment.Qbar, s=0)
    Qbar_hat_s0 = qbar_hat_algo.fit(obs_df)
    Qbar_hat_s0.predict(obs_df)

    # Qbar_hat_s0 = estimate_Qbar_oracle(obs_df, experiment.Qbar, s=0)
    # print(Qbar_hat_s0)

    # Practice Problem 9.4.3
    s=5 # Use s>=10 cause model to fail to fit at all
    iter = 1000  # Number of iterations
    gbar_hat_algo = OracleGbarModel(G0_fn=experiment.Gbar, s=s)
    learned_features_fixed_sample_size = get_learned_features(obs_df, iter, gbar_hat_algo)

    qbar_hat_algo = OracleQbarModel(Qbar_truth_fn=experiment.Qbar, s=s)
    updated_features = update_learned_features(learned_features_fixed_sample_size, 
                                               algorithm_d=qbar_hat_algo, 
                                               algorithm_e=kknn_algo)

    # apply one step correction
    updated_features_one_step = update_learned_features_one_step(updated_features, 
                                                                 algorithm_d=qbar_hat_algo, 
                                                                 algorithm_e=kknn_algo)
    

    psi_hat_de_os_df, bias_de_os = get_psi_hat_de_one_step(updated_features_one_step, psi_zero=evaluate_psi(experiment))
    print(bias_de_os)

    plot_estimation_bias_de_os(psi_hat_de_os_df, bias_de_os)


            
            



    
