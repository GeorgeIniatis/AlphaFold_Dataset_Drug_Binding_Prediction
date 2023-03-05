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


def user_inputs_section(key):
    drug_cid = st.number_input("PubChem Compound CID",
                               min_value=0,
                               help="The PubChem CID or ID refers to the unique identifier used to identify a compound present in the PubChem database. It is usually the first field below the compound's name.",
                               key=f"{key}_drug_cid")

    ## CHANGE TO ALL PROTEINS
    if key == "baseline":
        proteins_with_embeddings = np.load("Streamlit_App/data/Datasets/Proteins_With_Embedding_List.npy",
                                           allow_pickle=True)
        protein_accession = st.selectbox("Protein Accession",
                                         proteins_with_embeddings,
                                         key=f"{key}_protein_accession")
    elif key == "enhanced":
        proteins_with_embeddings = np.load("Streamlit_App/data/Datasets/Proteins_With_Embedding_List.npy",
                                           allow_pickle=True)
        protein_accession = st.selectbox("Protein Accession",
                                         proteins_with_embeddings,
                                         key=f"{key}_protein_accession")
    else:
        raise ValueError("Invalid key. Please choose between 'baseline' and 'enhanced'")

    return drug_cid, protein_accession


def classification_result_column_section(model, model_name, descriptors, key):
    st.markdown("##### Result")
    if model_name not in ["Dummy Classifier", "Linear Support Vector Classification"]:
        prediction_probability = model.predict_proba(descriptors)
        st.markdown(
            f"Probability that the drug-protein pair has an inactive relationship: **{prediction_probability[0][0]:.5f}**")
        st.markdown(
            f"Probability that the drug-protein pair has an active relationship: **{prediction_probability[0][1]:.5f}**")

    prediction = model.predict(descriptors)
    if prediction == 1:
        st.markdown("""
                    The model has predicted that the drug-protein pair you have specified has an **Active Relationship**
                    """)
    else:
        st.markdown("""
                    The model has predicted that the drug-protein pair you have specified has an  **Inactive Relationship**
                    """)

    # LIME Explainer
    if model_name not in ["Dummy Classifier", "Support Vector Classification"]:
        if key == 'baseline':
            X_train, y_train, feature_selection_columns = get_baseline_classification_sets()

            explainer = get_lime_explainer("Classification", feature_selection_columns, X_train, y_train)
            exp = explainer.explain_instance(descriptors, model.predict_proba, num_features=20)
        else:
            X_train, y_train, feature_selection_columns = get_enhanced_classification_sets()

            explainer = get_lime_explainer("Classification", feature_selection_columns, X_train, y_train)
            exp = explainer.explain_instance(descriptors, model.predict_proba, num_features=20)

        st.markdown("##### Prediction Explanation")
        st.pyplot(exp.as_pyplot_figure())

    # ELI5 Model Weights
    if model_name not in ["Dummy Classifier", "K-Nearest Neighbour Classifier"]:
        st.markdown("##### Model Weights")
        if key == "baseline":
            X_train, y_train, feature_selection_columns = get_baseline_classification_sets()
            st.write(get_model_weights(model, "Classification", feature_selection_columns))
        else:
            X_train, y_train, feature_selection_columns = get_enhanced_classification_sets()
            st.write(get_model_weights(model, "Classification", feature_selection_columns))
