from utils.model_interpretability_helpers import *
import streamlit as st
import numpy as np
import base64


# Reference
# https://discuss.streamlit.io/t/include-svg-image-as-part-of-markdown/1314
def render_svg(svg_file):
    with open(svg_file, "r") as f:
        lines = f.readlines()
        svg = "".join(lines)

        # Renders the given svg string
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
        return html


def title():
    st.set_page_config(layout="wide")
    st.title("Linking QSAR-Based Drug-Target Prediction with AlphaFold")


def make_prediction_section(category):
    if category == "Classification":
        st.markdown("##### Make Predictions")
        st.markdown("""
                    - Please choose a **Model**, enter a **PubChem Compound CID** number and choose a **Protein Accession**
                    - Please be patient when using K-Nearest Neighbour Classifier and Random Forest Classifier as these are loaded from Google Drive    
                    """)
    elif category == "Regression":
        st.markdown("##### Make Predictions")
        st.markdown("""
                    - Please choose a **Model**, enter a **PubChem Compound CID** number and choose a **Protein Accession**
                    - Please be patient when using K-Nearest Neighbour Regressor and Random Forest Regressor as these are loaded from Google Drive    
                    """)
    else:
        raise ValueError("Invalid category. Please choose 'Classification' or 'Regression'")


def user_inputs_section(key):
    drug_cid = st.number_input("PubChem Compound CID",
                               min_value=0,
                               help="The PubChem CID or ID refers to the unique identifier used to identify a compound present in the PubChem database. It is usually the first field below the compound's name.",
                               key=f"{key}_drug_cid")

    if key == "Baseline_Models":
        all_proteins = np.load("Streamlit_App/data/Datasets/All_Proteins_List.npy", allow_pickle=True)
        protein_accession = st.selectbox("Protein Accession",
                                         all_proteins,
                                         key=f"{key}_protein_accession")
    elif key == "Enhanced_Models":
        proteins_with_embeddings = np.load("Streamlit_App/data/Datasets/Proteins_With_Embedding_List.npy",
                                           allow_pickle=True)
        protein_accession = st.selectbox("Protein Accession",
                                         proteins_with_embeddings,
                                         key=f"{key}_protein_accession")
    else:
        raise ValueError("Invalid key. Please choose 'Baseline_Models' or 'Enhanced_Models'")

    return drug_cid, protein_accession


def classification_result_column_section(model, model_name, descriptors, key):
    st.markdown("##### Result")

    # Return Prediction Probability
    if model_name not in ["Dummy Classifier", "Linear Support Vector Classification"]:
        prediction_probability = model.predict_proba(descriptors.to_numpy().reshape(1, -1))
        st.markdown(
            f"Probability that the drug-protein pair has an **inactive** relationship: **{prediction_probability[0][0]:.5f}**")
        st.markdown(
            f"Probability that the drug-protein pair has an **active** relationship: **{prediction_probability[0][1]:.5f}**")

    # Return Prediction
    prediction = model.predict(descriptors.to_numpy().reshape(1, -1))
    if prediction == 1:
        st.markdown("""
                    The model has predicted that the drug-protein pair you have specified has an **Active Relationship**
                    """)
    else:
        st.markdown("""
                    The model has predicted that the drug-protein pair you have specified has an  **Inactive Relationship**
                    """)

    if key == "Baseline_Models":
        X_train, y_train, feature_selection_columns = get_baseline_classification_sets()
    elif key == "Enhanced_Models":
        X_train, y_train, feature_selection_columns = get_enhanced_classification_sets()
    else:
        raise ValueError("Invalid key. Please choose 'Baseline_Models' or 'Enhanced_Models'")

    # LIME Explainer
    # Removed for the moment due to RAM constraints
    # if model_name not in ["Dummy Classifier", "Support Vector Classification"]:
    #     explainer = get_lime_explainer("Classification", feature_selection_columns, X_train, y_train)
    #     exp = explainer.explain_instance(descriptors.astype(np.float64), model.predict_proba, num_features=20)
    #
    #     st.markdown("##### Prediction Explanation")
    #     st.pyplot(exp.as_pyplot_figure())

    # ELI5 Model Weights
    if model_name not in ["Dummy Classifier", "K-Nearest Neighbour Classifier"]:
        st.markdown("##### Model Weights")
        st.write(get_model_weights(model, "Classification", feature_selection_columns))


def regression_result_column_section(model, model_name, descriptors, key):
    st.markdown("##### Result")

    # Return Prediction
    predicted_logKd = model.predict(descriptors.to_numpy().reshape(1, -1))[0]
    st.markdown(f"Predicted **logKd: {predicted_logKd:.5f} (Kd: {np.exp(predicted_logKd):.5f})**")

    if key == "Baseline_Models":
        X_train, y_train, feature_selection_columns = get_baseline_regression_sets()
    elif key == "Enhanced_Models":
        X_train, y_train, feature_selection_columns = get_enhanced_regression_sets()
    else:
        raise ValueError("Invalid key. Please choose 'Baseline_Models' or 'Enhanced_Models'")

    # LIME Explainer
    # Removed for the moment due to RAM constraints
    # if model_name != "Dummy Regressor":
    #     explainer = get_lime_explainer("Regression", feature_selection_columns, X_train, y_train)
    #     exp = explainer.explain_instance(descriptors.astype(np.float64), model.predict, num_features=20)
    #
    #     st.markdown("##### Prediction Explanation")
    #     st.pyplot(exp.as_pyplot_figure())

    # ELI5 Model Weights
    if model_name not in ["Dummy Classifier", "K-Nearest Neighbour Regressor", "Random Forest Regressor"]:
        st.markdown("##### Model Weights")
        st.write(get_model_weights(model, "Regression", feature_selection_columns))
