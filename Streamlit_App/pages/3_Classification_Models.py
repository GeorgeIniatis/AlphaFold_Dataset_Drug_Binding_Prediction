from pathlib import Path
import pandas as pd
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.page_helpers import *
from utils.model_prediction_helpers import *

title()
useful_info = st.container()
baseline_models = st.container()
baseline_model_and_inputs, baseline_prediction_metrics = st.columns(2)
enhanced_models = st.container()
enhanced_model_and_inputs, enhanced_prediction_metrics = st.columns(2)
training_process = st.container()

with useful_info:
    st.subheader("Classification Models")
    st.markdown("""
                - Classification models make use of **"Activity_Binary"** as the label
                - All models were optimised using [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html) for **F1 score** using a **5-Fold Cross-Validation** except in the case of the Dummy Classifier model
                - Two different model categories:
                    - Baseline: Models with just the Drug and Protein Sequence Descriptors used as features
                    - Enhanced: Models with Drug and Protein Sequence Descriptors and Protein Structure Embeddings used as features
                - Training & Test sets:
                    - The same training and test sets were used by both model categories in order to properly compare their performances. However, we should mention that the Enhanced Models used slighly smaller versions of the sets as structural embeddings could not be created for every single protein present in them
                """)

with baseline_models:
    st.subheader("Baseline Models: Drug & Protein Sequence Descriptors")

    st.markdown("##### Training Performance")
    baseline_training_performance = pd.read_csv(
        "Streamlit_App/data/Metrics/Classification_Baseline_Models_Training_Metrics.csv", skiprows=1)
    st.write(baseline_training_performance)

    st.markdown("##### Testing Performance")
    baseline_testing_performance_ = pd.read_csv(
        "Streamlit_App/data/Metrics/Classification_Baseline_Models_Testing_Metrics.csv", skiprows=1)
    st.write(baseline_testing_performance_)

    with baseline_model_and_inputs:
        make_prediction_section("Classification")

        baseline_chosen_model = st.selectbox('Please choose a model to make predictions',
                                             classification_model_name_to_file.keys(),
                                             key="baseline_chosen_model")
        if baseline_chosen_model != "-":
            drug_cid, protein_accession = user_inputs_section(key="Baseline_Models")

            if st.button("Predict", key="baseline_button"):
                with baseline_prediction_metrics:
                    model = load_model(baseline_chosen_model, "Classification", "Baseline_Models")
                    drug_descriptors = get_chemical_descriptors(drug_cid, "Classification")

                    if isinstance(drug_descriptors, str):
                        st.markdown(f"""
                                    **{drug_descriptors}**
                                    """)
                    else:
                        protein_descriptors = get_protein_descriptors(protein_accession, "Classification",
                                                                      "Baseline_Models")

                        descriptors = pd.concat(objs=[drug_descriptors, protein_descriptors])
                        classification_result_column_section(model, baseline_chosen_model, descriptors,
                                                             "Baseline_Models")

with enhanced_models:
    st.subheader("Enhanced Models: Drug & Protein Sequence Descriptors & Protein Structural Embeddings")
    st.markdown("""
                * **The Neural network is not available due to data size constraints**
                """)

    st.markdown("##### Training Performance")
    baseline_training_performance = pd.read_csv(
        "Streamlit_App/data/Metrics/Classification_Enhanced_Models_Training_Metrics.csv", skiprows=1)
    st.write(baseline_training_performance)

    st.markdown("##### Testing Performance")
    baseline_testing_performance_ = pd.read_csv(
        "Streamlit_App/data/Metrics/Classification_Enhanced_Models_Testing_Metrics.csv", skiprows=1)
    st.write(baseline_testing_performance_)

    with enhanced_model_and_inputs:
        make_prediction_section("Classification")

        enhanced_chosen_model = st.selectbox('Please choose a model to make predictions',
                                             classification_model_name_to_file.keys(),
                                             key="enhanced_chosen_model")
        if enhanced_chosen_model != "-":
            drug_cid, protein_accession = user_inputs_section(key="Enhanced_Models")

            if st.button("Predict", key="enhanced_button"):
                with enhanced_prediction_metrics:
                    model = load_model(enhanced_chosen_model, "Classification", "Enhanced_Models")
                    drug_descriptors = get_chemical_descriptors(drug_cid, "Classification")

                    if isinstance(drug_descriptors, str):
                        st.markdown(f"""
                                    **{drug_descriptors}**
                                    """)
                    else:
                        protein_descriptors = get_protein_descriptors(protein_accession, "Classification",
                                                                      "Enhanced_Models")

                        descriptors = pd.concat(objs=[drug_descriptors, protein_descriptors])
                        classification_result_column_section(model, enhanced_chosen_model, descriptors,
                                                             "Enhanced_Models")

with training_process:
    st.subheader("Training & Testing Process Overview")
    st.markdown(render_svg("Streamlit_App/data/Plots/Model_Training.svg"), unsafe_allow_html=True)
