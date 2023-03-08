from lime.lime_tabular import LimeTabularExplainer
import streamlit as st
import pandas as pd
import eli5


@st.cache_data
def get_baseline_classification_sets():
    X_train = pd.read_pickle("Streamlit_App/data/Training_Test_Sets/Classification/X_train_feature_selection.pkl")
    X_train.drop(columns=["Drug_CID", "Protein_Accession"], inplace=True)
    feature_selection_columns = X_train.loc[:, "MolecularWeight":].columns
    X_train = X_train.to_numpy()

    y_train = pd.read_pickle("Streamlit_App/data/Training_Test_Sets/Classification/y_train.pkl")
    y_train = y_train.to_numpy()

    return X_train, y_train, feature_selection_columns


@st.cache_data
def get_baseline_regression_sets():
    X_train = pd.read_pickle("Streamlit_App/data/Training_Test_Sets/Regression/X_train_feature_selection.pkl")
    X_train.drop(columns=["Protein_Accession", "Drug_CID", "Activity_Name"], inplace=True)
    feature_selection_columns = X_train.loc[:, "MolecularWeight":].columns
    X_train = X_train.to_numpy()

    y_train = pd.read_pickle("Streamlit_App/data/Training_Test_Sets/Regression/y_train.pkl")
    y_train.drop(columns=["Activity_Binary"], inplace=True)
    y_train = y_train.to_numpy()

    return X_train, y_train, feature_selection_columns


@st.cache_data
def get_enhanced_classification_sets():
    protein_embeddings_dataframe = pd.read_pickle("Streamlit_App/data/Datasets/Protein_Embeddings.pkl")
    X_train = pd.read_pickle("Streamlit_App/data/Training_Test_Sets/Classification/X_train_feature_selection.pkl")
    X_train = pd.merge(X_train.reset_index(), protein_embeddings_dataframe, on="Protein_Accession").set_index(
        'index')
    X_train.drop(columns=["Drug_CID", "Protein_Accession"], inplace=True)

    y_train = pd.read_pickle("Streamlit_App/data/Training_Test_Sets/Classification/y_train.pkl")
    y_train = y_train[X_train.index]

    feature_selection_columns = X_train.loc[:, "MolecularWeight":].columns

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    return X_train, y_train, feature_selection_columns


@st.cache_data
def get_enhanced_regression_sets():
    protein_embeddings_dataframe = pd.read_pickle("Streamlit_App/data/Datasets/Protein_Embeddings.pkl")
    X_train = pd.read_pickle("Streamlit_App/data/Training_Test_Sets/Regression/X_train_feature_selection.pkl")
    X_train = pd.merge(X_train.reset_index(), protein_embeddings_dataframe, on="Protein_Accession").set_index(
        'index')
    X_train.drop(columns=["Protein_Accession", "Drug_CID", "Activity_Name"], inplace=True)

    y_train = pd.read_pickle("Streamlit_App/data/Training_Test_Sets/Regression/y_train.pkl")
    y_train = y_train.loc[X_train.index, :]
    y_train.drop(columns=["Activity_Binary"], inplace=True)

    feature_selection_columns = X_train.loc[:, "MolecularWeight":].columns

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()

    return X_train, y_train, feature_selection_columns


def get_model_weights(model, category, feature_selection_columns):
    if category == "Classification":
        return eli5.explain_weights_df(model,
                                       feature_names=feature_selection_columns,
                                       target_names={1: "Active", 0: "Inactive"})

    elif category == "Regression":
        return eli5.explain_weights_df(model, feature_names=feature_selection_columns)

    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")


def get_lime_explainer(category, feature_selection_columns, X_train, y_train=None):
    if category == "Classification":
        return LimeTabularExplainer(training_data=X_train,
                                    mode='classification',
                                    feature_names=feature_selection_columns.to_list(),
                                    training_labels=y_train,
                                    class_names=['Inactive', 'Active'],
                                    random_state=42)

    elif category == "Regression":
        return LimeTabularExplainer(training_data=X_train,
                                    mode='regression',
                                    feature_names=feature_selection_columns.to_list(),
                                    random_state=42)
    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")
