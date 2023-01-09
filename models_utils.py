from utils import *
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import mean_absolute_error, r2_score
from lime.lime_tabular import LimeTabularExplainer
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from joblib import dump, load
from sklearn.pipeline import Pipeline
import eli5
import plotly.express as px


def calculate_metrics_classification(y_true, y_pred):
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    print(f"F1 score: {f1}")
    print(f"Matthews Correlation Coefficient: {mcc}")
    print(f"Accuracy score: {accuracy}")
    print(f"Recall score: {recall}")
    print(f"Precision score: {precision}")


def calculate_metrics_regression(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # The closer to 1 the better
    print(f"R2 Score: {r2}")
    # The closer to 0 the better
    print(f"Negated Mean Absolute Error: {-mae}")


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
