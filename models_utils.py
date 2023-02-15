# General Imports
from utils import *
from sklearnex import patch_sklearn

patch_sklearn()

# Dimensionality Reduction
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Scalers
from sklearn.preprocessing import StandardScaler

# Metrics
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

# Interpretability
import eli5
from lime.lime_tabular import LimeTabularExplainer
from sklearn import tree

# Hyperparameter Tuning
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer

# Feature & Model selection
from sklearn.feature_selection import RFECV

# Other
import pickle
from joblib import dump, load
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from graphviz import Source
from IPython.display import SVG

# Plotting
import plotly.express as px

# PyTorch
import torch

template = "plotly_dark"


def calculate_metrics_classification(y_true, y_pred, print_results=True):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    if print_results:
        print(f"F1 score: {f1}")
        print(f"Matthews Correlation Coefficient: {mcc}")
        print(f"Accuracy score: {accuracy}")
        print(f"Recall score: {recall}")
        print(f"Precision score: {precision}")
    else:
        return [recall, precision, f1, accuracy, mcc]


def calculate_metrics_regression(y_true, y_pred, print_results=True):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    if print_results:
        # The closer to 1 the better
        print(f"R2 Score: {r2}")
        # The closer to 0 the better
        print(f"Negated Mean Absolute Error: {-mae}")
    else:
        return [mae, r2]


def get_confidence_intervals(model, X, y, sample_size, category, n_sample=1000, interval=95, print_iterator=False):
    metrics = {}
    alpha = 100 - interval

    if category == "Classification":
        metrics["Recall"] = []
        metrics["Precision"] = []
        metrics["F1"] = []
        metrics["Accuracy"] = []
        metrics["MCC"] = []

        for i in range(n_sample):
            if print_iterator:
                print(i)
            sample_indices = np.random.randint(0, len(X), sample_size)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            metrics_list = calculate_metrics_classification(y_sample, model.predict(X_sample), print_results=False)
            metrics["Recall"].append(metrics_list[0])
            metrics["Precision"].append(metrics_list[1])
            metrics["F1"].append(metrics_list[2])
            metrics["Accuracy"].append(metrics_list[3])
            metrics["MCC"].append(metrics_list[4])

    elif category == "Regression":
        metrics["Negated-MAE"] = []
        metrics["R2"] = []

        for i in range(n_sample):
            if print_iterator:
                print(i)
            sample_indices = np.random.randint(0, len(X), sample_size)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            metrics_list = calculate_metrics_regression(y_sample, model.predict(X_sample), print_results=False)
            metrics["Negated-MAE"].append(-metrics_list[0])
            metrics["R2"].append(metrics_list[1])
    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")

    print(f"Metrics after {n_sample} bootstrapped samples of size {sample_size}")
    print("--------------------------------------------------------")
    for metric, values in metrics.items():
        median = np.percentile(values, 50)
        low_confidence_interval = np.percentile(values, alpha / 2)
        high_confidence_interval = np.percentile(values, 100 - alpha / 2)
        print(
            f"Median {metric}: {median:.2f} with a {interval}% confidence interval of [{low_confidence_interval:.2f},{high_confidence_interval:.2f}]")


def plot_confusion_matrix(y_true, y_pred):
    confusion_matrix_array = confusion_matrix(y_true, y_pred)
    fig = ff.create_annotated_heatmap(confusion_matrix_array, x=["Inactive", "Active"], y=["Inactive", "Active"],
                                      colorscale="blues", showscale=True)
    fig.update_layout(template=template, xaxis_title="Predicted Label", yaxis_title="True Label")
    fig.show()


def prediction_category_classification(df):
    if df['True Class'] == df['Prediction']:
        return 'Correct'
    elif (df['True Class'] == 0) and (df['Prediction'] == 1):
        return 'False Positive'
    else:
        return 'False Negative'


def error_analysis_classification(y_pred, feature_selection_columns):
    X_test_dataframe = load_from_pickle("Training_Test_Sets/Classification/X_test_feature_selection")
    y_test_series = load_from_pickle("Training_Test_Sets/Classification/y_test")

    # Combining data into one dataframe
    y_pred_series = pd.Series(y_pred, index=y_test_series.index)

    error_analysis_dataframe = pd.concat([X_test_dataframe, y_test_series], axis=1)
    error_analysis_dataframe = pd.concat([error_analysis_dataframe, y_pred_series], axis=1)
    error_analysis_dataframe.rename(columns={"Activity_Binary": "True Class", 0: "Prediction"}, inplace=True)
    error_analysis_dataframe["Is the prediction correct?"] = error_analysis_dataframe.apply(
        prediction_category_classification, axis=1)

    # Scaling
    scaler = StandardScaler()
    scaler.fit(error_analysis_dataframe.loc[:, feature_selection_columns])
    scaled_data = scaler.transform(error_analysis_dataframe.loc[:, feature_selection_columns])

    # PCA
    pca = PCA(n_components=2, random_state=0)
    pca.fit(scaled_data)
    pca_data = pca.transform(scaled_data)
    pca_dataframe_2d = pd.DataFrame(pca_data, columns=["PCA_Dimension_1", "PCA_Dimension_2"], index=y_test_series.index)

    # Joining dataframes
    error_analysis_dataframe = pd.concat([error_analysis_dataframe, pca_dataframe_2d], axis=1)

    # Plot
    fig = px.scatter(error_analysis_dataframe, x="PCA_Dimension_1", y="PCA_Dimension_2",
                     color="Is the prediction correct?",
                     symbol="Is the prediction correct?",
                     hover_data=['MW', 'TPSA', 'XLogP', 'NHD', 'NHA', 'NRB', 'True Class', 'Prediction'],
                     title="Correct Classifications vs Misclassifications")
    fig.show()

    # Useful stats
    print(
        f"Number of correct classifications: {len(error_analysis_dataframe[error_analysis_dataframe['Is the prediction correct?'] == 'Correct'])}")
    print(
        f"Number of misclassifications: {len(error_analysis_dataframe[error_analysis_dataframe['Is the prediction correct?'] != 'Correct'])}")
    print(
        f"False Positives (True class:0, Prediction:1): {len(error_analysis_dataframe[(error_analysis_dataframe['True Class'] == 0) & (error_analysis_dataframe['Prediction'] == 1)])}")
    print(
        f"False Negatives (True class:1, Prediction:0): {len(error_analysis_dataframe[(error_analysis_dataframe['True Class'] == 1) & (error_analysis_dataframe['Prediction'] == 0)])}")

    return error_analysis_dataframe.sort_values('Is the prediction correct?')


def model_weights(model, category, feature_selection_columns):
    if category == "Classification":
        return eli5.show_weights(model,
                                 feature_names=feature_selection_columns,
                                 target_names={1: "Active", 0: "Inactive"})
    elif category == "Regression":
        return eli5.show_weights(model, feature_names=feature_selection_columns)
    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")


def get_lime_explainer(category, X_train, y_train=None):
    if category == "Classification":
        return LimeTabularExplainer(training_data=X_train,
                                    mode='classification',
                                    feature_names=list(X_train.columns),
                                    training_labels=y_train,
                                    class_names=['Inactive', 'Active'],
                                    random_state=42)

    elif category == "Regression":
        return LimeTabularExplainer(training_data=X_train,
                                    mode='regression',
                                    feature_names=list(X_train.columns),
                                    random_state=42)
    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")


def visualise_decision_tree(decision_tree, feature_names, class_names, dot_file_save_path):
    if not os.path.exists("Dataset_Files/Baseline_Models/Classification/optimised_dtc.dot"):
        tree.export_graphviz(decision_tree,
                             feature_names=feature_names,
                             class_names=class_names,
                             out_file=dot_file_save_path,
                             filled=True)

    s = Source.from_file(dot_file_save_path, format='svg')
    s.view()


# Reference
# https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, path, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def normalise(dataframe, first_feature_column):
    scaler = StandardScaler()

    scaler.fit(dataframe.loc[:, first_feature_column:])
    X_normalised = scaler.transform(dataframe.loc[:, first_feature_column:])

    return X_normalised


def intersection_sanity_check(set1, set2):
    set1_proteins = set1.loc[:, "Protein_Accession"].tolist()
    set1_drugs = set1.loc[:, "Drug_CID"].tolist()

    set2_proteins = set2.loc[:, "Protein_Accession"].tolist()
    set2_drugs = set2.loc[:, "Drug_CID"].tolist()

    # Test Proteins
    for protein in set1_proteins:
        if protein in set2_proteins:
            return "Fail"

    # Test Drugs
    for drug in set1_drugs:
        if drug in set2_drugs:
            return "Fail"

    return "Pass"
