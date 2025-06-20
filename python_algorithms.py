import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import PolynomialFeatures
# from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold


# -------------------
# working_model_base
# -------------------
class WorkingModelBase():
    def __init__(self):
        """
        Initialize the base working model.
        """
        self.model = None
        self.formula = None
        self.type_of_preds = "response"
        self.ML = False

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
                             family=sm.families.Binomial(), **kwargs).fit()
        return self.model


# -------------------
# working_model_G_one
# -------------------
class WorkingModelGOne(WorkingModelBase):
    def __init__(self):
        """
        Initialize the working model for Gbar with one feature.
        """
        self.model = None
        self.formula = "A ~ " + " + ".join(
                                            [f"I(W**{p})" for p in np.arange(0.5, 2.0, 0.5)] +
                                            [f"I(abs(W - 5/12)**{p})" for p in np.arange(0.5, 2.0, 0.5)]
                                        )
        self.type_of_preds = "response"
        self.ML = False

# -------------------
# working_model_G_two
# -------------------
class WorkingModelGTwo(WorkingModelBase):
    def __init__(self, powers=np.repeat(np.arange(0.25, 3.25, 0.25), 2)):
        """
        Initialize the working model for Gbar with two features.
        """
        self.model = None
        self.formula = "A ~ " + " + ".join(
            [f"I(W**{p})" for p in powers] +
            [f"I(abs(W - 5/12)**{p})" for p in powers]
        )
        self.type_of_preds = "response"
        self.ML = False

# ---------------------
# working_model_G_three
# ---------------------
class WorkingModelGThree(WorkingModelBase):
    def __init__(self):
        """
        Initialize the working model for Gbar with trigonometric and other transformations.
        """
        self.model = None
        self.formula = "A ~ " + " + ".join(
            [f"I({func}(W))" for func in ["np.cos", "np.sin", "np.sqrt", "np.log", "np.exp"]]
        )
        # self.formula = "A ~ " + " + ".join(
        #     [f"I({func}(W))" for func in ["cos", "sin", "sqrt", "log", "exp"]]
        # )
        self.type_of_preds = "response"
        self.ML = False

# -------------------
# working_model_Q_one
# -------------------
class WorkingModelQOne(WorkingModelBase):
    def __init__(self):
        """
        Initialize the working model for Qbar with interaction terms.
        """
        self.model = None
        self.formula = "Y ~ A * (" + " + ".join(
            [f"I(W**{p})" for p in np.arange(0.5, 2.0, 0.5)]
        ) + ")"
        self.type_of_preds = "response"
        self.ML = False
        self.stratify = False


# ----------------------------
# machine_learning_model_base
# ----------------------------
class MLModelBase():
    def __init__(self):
        """
        Initialize the base machine learning model.
        """
        self.model = None
        self.type_of_preds = "raw"
        self.ML = True
        self.stratify = True
        self.grid = {}
        self.control = {
            "method": "cv",  # Cross-validation
            "number": 10,  # Number of folds
            "predictionBounds": (0, 1),  # Prediction bounds
            "trim": True,  # Trimming option
            "allowParallel": True  # Allow parallel processing
        }

    def preprocess_data(self, dat, binary_label=False, **kwargs):
        if isinstance(dat, np.ndarray):
            dat = pd.DataFrame(dat, columns=["W", "A", "Y"])

        # Handle subsampling if specified
        if "Subsample" in kwargs:
            subsample_size = kwargs.pop("Subsample")
            dat = dat.sample(n=subsample_size, random_state=42)
        
        # Transform the label from continuous to binary if needed
        if binary_label:
            lab_enc = preprocessing.LabelEncoder()
            dat["Y"] = lab_enc.fit_transform(dat["Y"])
            print("The label Y is transform to {}".format(dat["Y"].dtype))

        X = dat[["A", "W"]].copy()
        y = dat["Y"]
        return X, y

    def cross_validate(self, X, y, **kwargs):
        """
        Perform cross-validation on the model.

        Parameters:
            X: pandas DataFrame or NumPy array containing the features.
            y: pandas Series or NumPy array containing the target.
            **kwargs: Additional arguments for the cross-validation.

        Returns:
            cv_results: Cross-validation results.
        """
        # StraifiedKFold can throw out error when the number of classes is less than the number of splits
        # cv = StratifiedKFold(n_splits=self.control["number"], shuffle=True, random_state=42)
        cv = KFold(n_splits=self.control["number"], shuffle=True, random_state=42)

        if self.control["allowParallel"]:
            n_jobs = -1
        else:
            n_jobs = 1
        grid_search = GridSearchCV(self.model, self.grid, cv=cv, n_jobs=n_jobs)
        estimator = grid_search.fit(X, y, **kwargs)
        return estimator.best_estimator_ # return the best estimator from grid search

    def fit(self, dat, **kwargs):
        """
        Fit the machine learning model to the data.

        Parameters:
            X: pandas DataFrame or NumPy array containing the features.
            y: pandas Series or NumPy array containing the target.
            **kwargs: Additional arguments for the model.

        Returns:
            None
        """
        X, y = self.preprocess_data(dat, **kwargs)
        
        # Use cross-validation if specified in control
        if self.control["method"] == "cv": 
            model = self.cross_validate(X, y, **kwargs)
        else:
            model = self.model
            model.fit(X, y)
        
        return model
    
    def predict(self, dat, model=None, **kwargs):
        """
        Predict using the fitted model.

        Parameters:
            X: pandas DataFrame containing the features.
            **kwargs: Additional arguments for the predict method.

        Returns:
            predictions: The predicted values.
        """
        X, y = self.preprocess_data(dat, **kwargs)

        if model is not None:
            model = model
        else:
            raise ValueError("Fitted Model must be provided for prediction.")

        predictions = model.predict(X)
        
        # Apply prediction bounds if specified
        if self.control["predictionBounds"]:
            lower, upper = self.control["predictionBounds"]
            predictions = np.clip(predictions, lower, upper)
        
        return predictions


# ---------
# kknn_algo
# ---------
class KknnAlgo(MLModelBase):
    def __init__(self, **kwargs):
        """
        Initialize the kknn algorithm.
        """
        self.grid = {"n_neighbors": [25],
                    "metric": ["minkowski"],
                    "weights": ["uniform"]}
        
        # self.model = KNeighborsClassifier(n_neighbors=self.grid["n_neighbors"][0],
        #                                   metric=self.grid["metric"][0],
        #                                   weights=self.grid["weights"][0],
        #                                   **kwargs)
        self.model = KNeighborsRegressor(n_neighbors=self.grid["n_neighbors"][0],
                                         metric=self.grid["metric"][0],
                                         weights=self.grid["weights"][0],
                                         **kwargs)
        self.type_of_preds = "raw"
        self.ML = True
        self.stratify = False
        
        self.control = {"method": "none",  # No cross-validation
                            "predictionBounds": (0, 1),  # Prediction bounds
                            "trim": True,  # Trimming option
                            "allowParallel": True,  # Allow parallel processing
                            }
    
    def preprocess_data(self, dat, binary_label=False, **kwargs):
        if isinstance(dat, np.ndarray):
            dat = pd.DataFrame(dat, columns=["W", "A", "Y"])

        # Handle subsampling if specified
        if "Subsample" in kwargs:
            subsample_size = kwargs.pop("Subsample")
            dat = dat.sample(n=subsample_size, random_state=42)
        
        # Transform the label from continuous to binary if needed
        if binary_label:
            lab_enc = preprocessing.LabelEncoder()
            dat["Y"] = lab_enc.fit_transform(dat["Y"])
            print("The label Y is transform to {}".format(dat["Y"].dtype))

        # Extract features and target
        X = dat[["A", "W"]].copy()
        X["A"] = 10 * X["A"] + X["W"]  # Apply the transformation I(10 * A + W)
        y = dat["Y"]
        return X, y


# ------------------
# boosting_tree_algo
# ------------------
class BoostingTreeAlgo(MLModelBase):
    def __init__(self, **kwargs):
        """
        Initialize the Boosting Tree algorithm class.
        """
        self.grid = {
            "n_estimators": [10, 20, 30],
            "learning_rate": [0.1, 0.2],
            "max_depth": [1, 2, 5]
        }

        # self.model = GradientBoostingClassifier(learning_rate=self.grid["learning_rate"][0],
        #                                         n_estimators=self.grid["n_estimators"][0],
        #                                         max_depth=self.grid["max_depth"][0],
        #                                         **kwargs)
        self.model = GradientBoostingRegressor(learning_rate=self.grid["learning_rate"][0],
                                                n_estimators=self.grid["n_estimators"][0],
                                                max_depth=self.grid["max_depth"][0],
                                                **kwargs)
        self.type_of_preds = "raw"
        self.ML = True
        self.stratify = True
        
        self.control = {
            "method": "cv",  # Cross-validation
            "number": 10,  # Number of folds
            "predictionBounds": (0, 1),  # Prediction bounds
            "trim": True,  # Trimming option
            "allowParallel": True  # Allow parallel processing
        }



# ------------------
# boosting_lm_algo
# ------------------
class BoostingLMAlgo(MLModelBase):
    def __init__(self, **kwargs):
        """
        Initialize the Boosting LM algorithm class.
        """
        self.grid = {
            "n_estimators": [50, 100],
            "learning_rate": [0.1, 0.2, 0.3]
        }
        
        self.model = GradientBoostingClassifier(learning_rate=self.grid["learning_rate"][0],
                                                n_estimators=self.grid["n_estimators"][0],
                                                **kwargs)
        self.type_of_preds = "raw"
        self.ML = True
        self.stratify = True
        
        self.control = {
            "method": "cv",  # Cross-validation
            "number": 10,  # Number of folds
            "predictionBounds": (0, 1),  # Prediction bounds
            "trim": True,  # Trimming option
            "allowParallel": True  # Allow parallel processing
        }