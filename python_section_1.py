# Section 1 A ride @ https://achambaz.github.io/tlride/1-a-ride.html
# for experiments
import numpy as np
from scipy.special import expit
from scipy.stats import beta, bernoulli, uniform
# for visualization
import pandas as pd
import matplotlib.pyplot as plt

class LAW:
    def __init__(self, name="LAW"):
        self.name = name

    def QW(self, *args, **kwargs):
        """
        Function to estimate Q(W) — distribution of baseline covariates.
        Override in child class for specific behavior.
        """
        raise NotImplementedError("QW() must be implemented in the subclass")

    def Gbar(self, *args, **kwargs):
        """
        Function to estimate G — treatment assignment mechanism.
        Override in child class for specific behavior.
        """
        raise NotImplementedError("Gbar() must be implemented in the subclass")

    def Qbar(self, *args, **kwargs):
        """
        Function to estimate Q — outcome regression model.
        Override in child class for specific behavior.
        """
        raise NotImplementedError("Qbar() must be implemented in the subclass")

    def qY(self, *args, **kwargs):
        """
        Function to evaluate or sample the potential outcome Y.
        Override in child class for specific behavior.
        """
        raise NotImplementedError("qY() must be implemented in the subclass")

    def sample_from(self, *args, **kwargs):
        """
        Function to sample from the data-generating distribution.
        Override in child class for specific behavior.
        """
        raise NotImplementedError("sample_from() must be implemented in the subclass")
    
    def reveal(self):
        # Return a dictionary of function names and their references
        return {func: getattr(self, func) for func in ["QW", "Gbar", "Qbar", "qY", "sample_from"]}
    
class Experiment(LAW):
    def __init__(self):
        super().__init__(name="experiment")

    def QW(self, W, 
           mixture_weights=[1/10, 9/10, 0], 
           mins=[0, 11/30, 0], 
           maxs=[1, 14/30, 1]):
        ''' QW feature describes the marginal law of W '''
        out = np.array([mixture_weights[ii] * uniform.pdf(x=W, loc=mins[ii], scale=maxs[ii]- mins[ii])
                        for ii in range(len(mixture_weights))])
        return np.sum(out, axis=0)


    def Gbar(self, W):
        ''' Gbar feature describes the conditional probability of action A=1 given W'''
        return expit(1 + 2 * W - 4 * np.sqrt(np.abs(W - 5/12)))

    def Qbar(self, AW):
        """
        Compute the conditional mean of Y given A and W.
        AW is assumed to be a NumPy array where:
            Column 0 corresponds to A.
            Column 1 corresponds to W.
        """
        A = AW[:, 0]  # Extract column 0 (A)
        W = AW[:, 1]  # Extract column 1 (W)
        
        # Compute Qbar based on the formula
        Qbar_A1 = (np.cos((-1/2 + W) * np.pi) * 2/5 + 1/5 +
                ((1/3 <= W) & (W <= 1/2)) / 5 +
                (W >= 3/4) * (W - 3/4) * 2)
        Qbar_A0 = (np.sin(4 * W**2 * np.pi) / 4 + 1/2)
        
        return A * Qbar_A1 + (1 - A) * Qbar_A0

    def qY(self, obs, shape10=2, shape11=3):
        """
        The obs array is assumed to be a NumPy array where:
            Column 0 corresponds to W.
            Column 1 corresponds to A.
            Column 2 corresponds to Y.
        The Qbar function is expected to take AW as input and return the conditional probability values.
        The beta.pdf function computes the Beta distribution's PDF.
        """
        W = obs[:, 0]  # Column 0 corresponds to "W"
        A = obs[:, 1]  # Column 1 corresponds to "A"
        AW = np.column_stack((A, W))  # Correctly construct AW with A as column 0 and W as column 1    QAW = Qbar(AW)
        QAW = self.Qbar(AW)
        shape1 = np.where(A == 0, shape10, shape11)
        Y = obs[:, 2]  # Column 2 corresponds to "Y"
        return beta.pdf(Y, a=shape1, b=shape1 * (1 - QAW) / QAW)
    
    def sample_from_mixture_of_uniforms(self, n, 
                                        mixture_weights=[1/10, 9/10, 0], 
                                        mins=[0, 11/30, 0], 
                                        maxs=[1, 14/30, 1]):
        """
        Helper function to sample from a mixture of uniform distributions.
        """
        components = np.random.choice(len(mixture_weights), size=n, p=mixture_weights)
        W = np.array([np.random.uniform(mins[i], maxs[i]) for i in components])
        return W

    def sample_from(self, n, ideal=False):
        """
        Sample observations based on the given parameters.
        """
        # Validate inputs
        if not isinstance(n, int):
            raise TypeError(f"Expected integer, got {type(n).__name__}")
        lower, upper = 1, np.inf
        if not (lower <= n < upper):
            raise ValueError(f"Integer value {n} must be in the range [{lower}, {upper})")
        if not isinstance(ideal, bool):
            raise ValueError("ideal must be a boolean.")

        # Sample W from the mixture of uniforms
        W = self.sample_from_mixture_of_uniforms(n)

        # Compute counterfactual rewards
        zeroW = np.column_stack((np.zeros(n), W))
        oneW = np.column_stack((np.ones(n), W))
        Qbar_zeroW = self.Qbar(zeroW)
        Qbar_oneW = self.Qbar(oneW)
        Yzero = beta.rvs(a=2, b=2 * (1 - Qbar_zeroW) / Qbar_zeroW, size=n)
        Yone = beta.rvs(a=3, b=3 * (1 - Qbar_oneW) / Qbar_oneW, size=n)

        # Sample actions
        A = bernoulli.rvs(p=self.Gbar(W), size=n)

        # Compute actual rewards
        Y = A * Yone + (1 - A) * Yzero

        # Create observations
        if ideal:
            obs = np.column_stack((W, Yzero, Yone, A, Y))
        else:
            obs = np.column_stack((W, A, Y))

        return obs


if __name__ == "__main__":
    # Instantiate the Experiment class
    relevant_features = Experiment()

    # TEST: Generate 5 observations
    five_obs = relevant_features.sample_from(n=5)
    print(five_obs)

    # Visulization
    # Generate a sequence of w values
    w = np.linspace(0, 1, int(1e3))

    # Compute features
    features = pd.DataFrame({"w": w})
    features["Qw"] = relevant_features.QW(features["w"])
    features["Gw"] = relevant_features.Gbar(features["w"])
    features["Q1w"] = relevant_features.Qbar(np.column_stack((np.ones(len(w)), w)))
    features["Q0w"] = relevant_features.Qbar(np.column_stack((np.zeros(len(w)), w)))
    features["blip_Qw"] = features["Q1w"] - features["Q0w"]

    # Manipulate the data
    features_long = (
        features.drop(columns=["Qw", "Gw"])
        .rename(columns={"Q1w": "Q(1,.)", "Q0w": "Q(0,.)", "blip_Qw": "Q(1,.) - Q(0,.)"})
        .melt(id_vars=["w"], var_name="f", value_name="value")
    )

    # Plot the data
    plt.figure(figsize=(10, 6))
    for label, df in features_long.groupby("f"):
        plt.plot(df["w"], df["value"], label=label, linewidth=1)

    plt.xlabel("w")
    plt.ylabel("f(w)")
    plt.title(r"Visualizing $\bar{Q}_0$")
    plt.ylim(None, 1)
    plt.legend(title="Features")
    plt.show()
