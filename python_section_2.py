# Section 2 A ride @ https://achambaz.github.io/tlride/1-a-ride.html
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform, beta, bernoulli
from scipy.special import expit, logit
from python_section_1 import LAW, Experiment



# 2.2 Practice problem 2
# (a,c) \in {0,1} \times {1/4, 1/2, 3/4}
def compute_gamma_true(experiment, a, c, y_grid, B=int(1e6)):
    # Step 1: sample W
    W = experiment.sample_from_mixture_of_uniforms(B)
    
    # Step 2: compute Qbar
    A = np.full_like(W, fill_value=a)
    AW = np.column_stack([A, W])
    Qaw = experiment.Qbar(AW)
    
    # Step 3: determine beta params
    theta = 2 if a == 0 else 3
    alpha = theta
    beta_param = theta * (1 - Qaw) / Qaw
    
    # Step 4: compute expected CDF for each y in grid
    cdf_vals = np.array([
        beta.cdf(y, a=alpha, b=beta_param).mean() for y in y_grid
    ])
    
    # Step 5: find smallest y where mean CDF ≥ c
    satisfying = y_grid[cdf_vals >= c]
    return satisfying[0] if len(satisfying) > 0 else 1.0

# Vectorized version of compute_gamma_true
def compute_gamma_vectorized(experiment, a, c, y_grid, B=int(1e6)):
    # Step 1: sample W
    W = experiment.sample_from_mixture_of_uniforms(B)
    
    # Step 2: compute Qbar
    A = np.full_like(W, fill_value=a)
    AW = np.column_stack([A, W])
    Qaw = experiment.Qbar(AW)
    
    # Step 3: determine beta params
    theta = 2 if a == 0 else 3
    alpha = theta
    beta_params = theta * (1 - Qaw) / Qaw

    # Vectorized computation of the conditional CDFs:
    # Shape: [len(W), len(y_grid)]
    cdf_matrix = beta.cdf(y_grid[np.newaxis, :], a=alpha, b=beta_params[:, np.newaxis])

    # Take expectation over W: mean over rows
    avg_cdf = np.mean(cdf_matrix, axis=0)  # Shape: [len(y_grid)]

    # Find the smallest y where the expected CDF >= c
    satisfying = y_grid[avg_cdf >= c]
    return satisfying[0] if len(satisfying) > 0 else 1.0


def calculate_deltas(experiment, c=[0.25, 0.5, 0.75], y_grid=np.linspace(1e-5, 1 - 1e-5, 500)):
    """
    Calculate the difference between two gamma values.
    """
    deltas = {}
    for c in [0.25, 0.5, 0.75]:
        gamma0 = compute_gamma_vectorized(experiment, a=0, c=c, y_grid=y_grid)
        gamma1 = compute_gamma_vectorized(experiment, a=1, c=c, y_grid=y_grid)
        delta = gamma1 - gamma0
        deltas[c] = delta
        print(f"delta_0,{c} = gamma_0,1,{c} - gamma_0,0,{c} ≈ {gamma1:.4f} - {gamma0:.4f} = {delta:.4f}")
    return deltas




# 2.2 Practice problem 3: same question with practice 2, but approximate use sampling
def estimate_gamma_from_sample(ideal_obs, a, c, y_grid):
    # Extract Y^a
    Y_a = ideal_obs[:, 1] if a == 0 else ideal_obs[:, 2]  # Yzero or Yone

    # Estimate the empirical CDF over the grid
    cdf_vals = np.array([(Y_a <= y).mean() for y in y_grid])

    # Find the smallest y such that CDF >= c
    satisfying = y_grid[cdf_vals >= c]
    return satisfying[0] if len(satisfying) > 0 else 1.0  # fallback

# vectorized
def estimate_gamma_from_sample_vec(ideal_obs, a, c, y_grid):
    Y_a = ideal_obs[:, 1] if a == 0 else ideal_obs[:, 2]
    # Broadcast comparison across grid
    cdf_vals = np.mean(Y_a[:, np.newaxis] <= y_grid[np.newaxis, :], axis=0)
    satisfying = y_grid[cdf_vals >= c]
    return satisfying[0] if len(satisfying) > 0 else 1.0

# calcualted deltas
def calculate_deltas_sampling(ideal_obs, c=[0.25, 0.5, 0.75], y_grid=np.linspace(1e-5, 1 - 1e-5, 500)):
    """
    Calculate the difference between two gamma values.
    """
    deltas = {}
    for c in [0.25, 0.5, 0.75]:
        gamma0 = estimate_gamma_from_sample_vec(ideal_obs, a=0, c=c, y_grid=y_grid)
        gamma1 = estimate_gamma_from_sample_vec(ideal_obs, a=1, c=c, y_grid=y_grid)
        delta = gamma1 - gamma0
        deltas[c] = delta
        print(f"delta_0,{c} = gamma_0,1,{c} - gamma_0,0,{c} ≈ {gamma1:.4f} - {gamma0:.4f} = {delta:.4f}")
    return deltas


# 2.3.3 another experiment
class AnotherExperiment(LAW):
    def __init__(self, name="AnotherExperiment"):
        super().__init__(name)

    def QW(self, x, min_val=1/10, max_val=9/10):
        """
        Function to estimate Q(W) — distribution of baseline covariates.
        """
        return uniform.pdf(x, loc=min_val, scale=max_val - min_val)

    def Gbar(self, W):
        """
        Function to estimate G — treatment assignment mechanism.
        """
        return np.sin((1 + W) * np.pi / 6)

    def Qbar(self, AW, h):
        """
        Function to estimate Q — outcome regression model.
        """
        A = AW[:, 0]
        W = AW[:, 1]
        return expit(logit(A * W + (1 - A) * W**2) + h * 10 * np.sqrt(W) * A)

    def qY(self, obs, Qbar, shape1=4):
        """
        Function to evaluate or sample the potential outcome Y.
        """
        W = obs[:, 0]  # Column 0 corresponds to "W"
        A = obs[:, 1]  # Column 1 corresponds to "A"
        AW = np.column_stack((A, W))  # Correctly construct AW with A as column 0 and W as column 1    QAW = Qbar(AW)
        QAW = Qbar(AW)
        Y = obs[:, 2]  # Column 2 corresponds to "Y"
        return beta.pdf(Y, a=shape1, b=shape1 * (1 - QAW) / QAW)

    def sample_from(self, n, h, ideal=False):
        """
        Function to sample from the data-generating distribution.
        """
        # Sample W from a uniform distribution
        min_val, max_val = 1/10, 9/10
        W = np.random.uniform(low=min_val, high=max_val, size=n)

        # Compute counterfactual rewards
        if ideal:
            zeroW = np.column_stack((np.zeros(n), W))
            oneW = np.column_stack((np.ones(n), W))
            Qbar_zeroW = self.Qbar(zeroW, h=h)
            Qbar_oneW = self.Qbar(oneW, h=h)
            Yzero = beta.rvs(a=4, b=4 * (1 - Qbar_zeroW) / Qbar_zeroW, size=n)
            Yone = beta.rvs(a=4, b=4 * (1 - Qbar_oneW) / Qbar_oneW, size=n)

        # Sample actions A using the Gbar function
        A = bernoulli.rvs(p=self.Gbar(W), size=n)

        # Compute Q(A, W)
        AW = np.column_stack((A, W))
        QAW = self.Qbar(AW, h=h)

        # Sample rewards Y using the Beta distribution
        shape1 = 4
        Y = beta.rvs(a=shape1, b=shape1 * (1 - QAW) / QAW, size=n)

        # Combine W, A, and Y into observations
        if ideal:
            obs = np.column_stack((W, Yzero, Yone, A, Y))
        else:
            obs = np.column_stack((W, A, Y))
        return obs



# 2.4 Practice Problem 1
# Vectorized version of compute_gamma_true
def compute_gamma_vectorized_another(experiment, a, c, y_grid, B=int(1e6), h=0):
    # Step 1: sample W
    min_val, max_val = 1/10, 9/10
    W = np.random.uniform(low=min_val, high=max_val, size=B)
    
    # Step 2: compute Qbar
    A = np.full_like(W, fill_value=a)
    AW = np.column_stack([A, W])
    Qaw = experiment.Qbar(AW, h)
    
    # Step 3: determine beta params
    theta = 4
    alpha = theta
    beta_params = theta * (1 - Qaw) / Qaw

    # Vectorized computation of the conditional CDFs:
    # Shape: [len(W), len(y_grid)]
    cdf_matrix = beta.cdf(y_grid[np.newaxis, :], a=alpha, b=beta_params[:, np.newaxis])

    # Take expectation over W: mean over rows
    avg_cdf = np.mean(cdf_matrix, axis=0)  # Shape: [len(y_grid)]

    # Find the smallest y where the expected CDF >= c
    satisfying = y_grid[avg_cdf >= c]
    return satisfying[0] if len(satisfying) > 0 else 1.0

def calculate_deltas_new(another_experiment, c=[0.25, 0.5, 0.75], y_grid=np.linspace(1e-5, 1 - 1e-5, 500)):
    """
    Calculate the difference between two gamma values.
    """
    deltas = {}
    for c in [0.25, 0.5, 0.75]:
        gamma0 = compute_gamma_vectorized_another(another_experiment, 
                                              a=0, c=c, y_grid=y_grid, B=int(1e6), h=0)
        gamma1 = compute_gamma_vectorized_another(another_experiment, 
                                              a=1, c=c, y_grid=y_grid, B=int(1e6), h=0)
        delta = gamma1 - gamma0
        deltas[c] = delta
        print(f"delta_0,{c} = gamma_0,1,{c} - gamma_0,0,{c} ≈ {gamma1:.4f} - {gamma0:.4f} = {delta:.4f}")
    return deltas


# 2.4 Practice Problem 2: Estimate gamma using sampling
# vectorized
def estimate_gamma_from_sample_vec(ideal_obs, a, c, y_grid):
    Y_a = ideal_obs[:, 1] if a == 0 else ideal_obs[:, 2]
    # Broadcast comparison across grid
    cdf_vals = np.mean(Y_a[:, np.newaxis] <= y_grid[np.newaxis, :], axis=0)
    satisfying = y_grid[cdf_vals >= c]
    return satisfying[0] if len(satisfying) > 0 else 1.0

# calcualted deltas
def calculate_deltas_sampling_new(ideal_obs_another, c=[0.25, 0.5, 0.75], y_grid=np.linspace(1e-5, 1 - 1e-5, 500)):
    """
    Calculate the difference between two gamma values.
    """
    deltas = {}
    for c in [0.25, 0.5, 0.75]:
        gamma0 = estimate_gamma_from_sample_vec(ideal_obs_another, a=0, c=c, y_grid=y_grid)
        gamma1 = estimate_gamma_from_sample_vec(ideal_obs_another, a=1, c=c, y_grid=y_grid)
        delta = gamma1 - gamma0
        deltas[c] = delta
        print(f"delta_0,{c} = gamma_0,1,{c} - gamma_0,0,{c} ≈ {gamma1:.4f} - {gamma0:.4f} = {delta:.4f}")
    return deltas


if __name__ == "__main__":
    # 2.1.2 A causal interpretation
    # Create a directed graph
    dag = nx.DiGraph()

    # Add nodes with labels
    labels = {
        "Y": "Actual reward",
        "A": "Action",
        "Y1": "Counterfactual reward\n of action 1",
        "Y0": "Counterfactual reward\n of action 0",
        "W": "Context of action"
    }

    # Add edges to represent relationships
    dag.add_edges_from([
        ("A", "Y"),
        ("Y1", "Y"),
        ("Y0", "Y"),
        ("W", "A"),
        ("W", "Y1"),
        ("W", "Y0")
    ])

    # Define coordinates for nodes
    coords = {
        "W": (0, 0),
        "A": (-1, -1),
        "Y1": (1.5, -0.5),
        "Y0": (0.25, -0.5),
        "Y": (1, -1)
    }

    # Draw the graph
    plt.figure(figsize=(8, 6))
    nx.draw(
        dag,
        pos=coords,
        labels=labels,
        with_labels=True,
        node_size=3000,
        node_color="lightgrey",
        font_size=10,
        font_color="black",
        arrows=True
    )
    plt.title("Visualizing DAG")
    plt.show()

    # 2.1.3 A causal computation
    # instantiating the Experiment class
    experiment = Experiment()

    # Number of samples
    B = int(1e6)

    # Generate ideal observations
    ideal_obs = experiment.sample_from(B, ideal=True)

    # Compute psi_approx
    psi_approx = np.mean(ideal_obs[:, 2] - ideal_obs[:, 1])  # Column 2 is "Yone", Column 1 is "Yzero"
    print(psi_approx)

    # constructing a 95% confidence interval for psi_0
    # Compute standard deviation of the difference
    sd_approx = np.std(ideal_obs[:, 2] - ideal_obs[:, 1])  # Column 2 is "Yone", Column 1 is "Yzero"

    # Significance level
    alpha = 0.05

    # Compute confidence interval
    psi_approx_CI = psi_approx + np.array([-1, 1]) * norm.ppf(1 - alpha / 2) * sd_approx / np.sqrt(B)
    print(psi_approx_CI)

    # 2.2 Practice Problem 2
    deltas = calculate_deltas(experiment)
    """output:
    delta_0,0.25 = gamma_0,1,0.25 - gamma_0,0,0.25 ≈ 0.6413 - 0.5050 = 0.1363
    delta_0,0.5 = gamma_0,1,0.5 - gamma_0,0,0.5 ≈ 0.8176 - 0.7275 = 0.0902
    delta_0,0.75 = gamma_0,1,0.75 - gamma_0,0,0.75 ≈ 0.9319 - 0.8898 = 0.0421
    """

    # 2.2 Practice Problem 3
    deltas_sampling = calculate_deltas_sampling(ideal_obs)
    """output:
    delta_0,0.25 = gamma_0,1,0.25 - gamma_0,0,0.25 ≈ 0.6413 - 0.5070 = 0.1343
    delta_0,0.5 = gamma_0,1,0.5 - gamma_0,0,0.5 ≈ 0.8176 - 0.7275 = 0.0902
    delta_0,0.75 = gamma_0,1,0.75 - gamma_0,0,0.75 ≈ 0.9319 - 0.8898 = 0.0421
    """

    # 2.3.3 Another Experiment
    # Example usage
    another_experiment = AnotherExperiment()
    two_obs_another_experiment = another_experiment.sample_from(2, h=0)
    print(two_obs_another_experiment)
    psi_approx = np.mean(two_obs_another_experiment[:, 2] - two_obs_another_experiment[:, 1])  # Column 2 is "Yone", Column 1 is "Yzero"
    print(psi_approx)

    # 2.4 Practice Problem 1
    deltas = calculate_deltas_new(another_experiment)
    """output:
    delta_0,0.25 = gamma_0,1,0.25 - gamma_0,0,0.25 ≈ 0.2625 - 0.0762 = 0.1864
    delta_0,0.5 = gamma_0,1,0.5 - gamma_0,0,0.5 ≈ 0.4790 - 0.2245 = 0.2545
    delta_0,0.75 = gamma_0,1,0.75 - gamma_0,0,0.75 ≈ 0.7315 - 0.4830 = 0.2485
    """

    # 2.4 Practice Problem 2
    ideal_obs_another = another_experiment.sample_from(int(1e6), h=0, ideal=True)
    deltas = calculate_deltas_sampling_new(ideal_obs_another)
    """output:
    delta_0,0.25 = gamma_0,1,0.25 - gamma_0,0,0.25 ≈ 0.2625 - 0.0762 = 0.1864
    delta_0,0.5 = gamma_0,1,0.5 - gamma_0,0,0.5 ≈ 0.4790 - 0.2265 = 0.2525
    delta_0,0.75 = gamma_0,1,0.75 - gamma_0,0,0.75 ≈ 0.7315 - 0.4830 = 0.2485
    """

    # 2.4 Practice Problem 3

