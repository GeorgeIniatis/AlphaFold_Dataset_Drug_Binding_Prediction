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
                - All models were optimised using [BayesSearchCV](https://scikit-optimize.github.io/stable/modules/generated/skopt.BayesSearchCV.html) for F1 score using a 5-Fold Cross-Validation except in the case of the Dummy Classifier
                - Two different model categories:
                    - Baseline: Models with just the Drug and Protein Sequence Descriptors used as features
                    - Enhanced: Models with Drug and Protein Sequence Descriptors and Protein Structure Embeddings used as features
                - Training & Test sets:
                    - The same training and test sets were used by both model categories in order to properly compare their performances. However, we should mention that the Enhanced Models used slighly smaller versions of the sets as structural embeddings could not be created for every single protein present in them.
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
        st.markdown("##### Make Predictions")
        st.markdown("""
                    - Please choose a **Model**, enter a **PubChem Compound CID** number and choose a **Protein Accession**
                    - Please be patient when using K-Nearest Neighbour Classifier and Random Forest Classifier as these are loaded from Google Drive    
                    """)

        baseline_chosen_model = st.selectbox('Please choose a model to make predictions',
                                             classification_model_name_to_file.keys(),
                                             key="baseline_chosen_model")
        if baseline_chosen_model != "-":
            drug_cid, protein_accession = user_inputs_section(key="baseline")

            if st.button("Predict", key="baseline_button"):
                with baseline_prediction_metrics:
                    model = load_model(baseline_chosen_model, "Classification", "Baseline_Models")
                    drug_descriptors = get_chemical_descriptors(drug_cid, "Classification")

                    if isinstance(drug_descriptors, str):
                        st.markdown(f"""
                                    **{drug_descriptors}**
                                    """)

                    # protein_descriptors


                    # classification_result_column_section(model, baseline_chosen_model, descriptors, "baseline")


#
# with cd_se_i_models:
#     st.subheader("Category 2: Models with Chemical Descriptors, Side Effects and Indications used as features")
#
#     st.markdown("##### Feature Selection")
#     st.markdown("""
#                     - Feature selection was used in order to find the most chemical descriptors, side effects and indications and to improve the models' training times
#                     - [Recursive Feature Elimination with Cross Validation (RFECV)](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) was used with a 10-Fold Cross Validation and a Random Forest Classifier optimised for F1 score
#                        - Features were reduced from 4353 to 217
#                        - All 6 chemical descriptors were kept
#                        - 196 of the side effects were kept
#                        - 15 of indications were kept
#                     """)
#     with cd_se_i_important_side_effects:
#         st.markdown("##### Important Side Effects")
#         st.dataframe(side_effects)
#
#     with cd_se_i_important_indications:
#         st.markdown("##### Important Indications")
#         st.dataframe(indications)
#
#     st.markdown("##### Training Performance With Feature Selection")
#     training_performance_cd_se_i = pd.read_csv(
#         "Streamlit_App/data/Metrics/Classification_Models_CD_SE_I_Training_Metrics.csv",
#         skiprows=1)
#     render_dataframe_as_table(training_performance_cd_se_i)
#
#     st.markdown("##### Testing Performance With Feature Selection")
#     testing_performance_cd_se_i = pd.read_csv(
#         "Streamlit_App/data/Metrics/Classification_Models_CD_SE_I_Testing_Metrics.csv",
#         skiprows=1)
#     render_dataframe_as_table(testing_performance_cd_se_i)
#
#     with cd_se_i_chosen_model_and_inputs:
#         st.markdown("##### Make Predictions")
#         st.markdown("""
#                         - Please choose a model, enter a decimal number for each of the 6 chemical descriptors and then pick one or multiple side effects and indications
#                         - Helpful definitions are available for each descriptor. To access them click the question mark icon
#                         """)
#
#         cd_se_i_chosen_model = st.selectbox('Please choose a model to make predictions',
#                                             cd_se_i_model_name_to_file.keys(),
#                                             key="cd_se_i")
#         if cd_se_i_chosen_model != "-":
#             mw, tpsa, xlogp, nhd, nha, nrb, side_effects_chosen, indications_chosen = user_inputs_section("cd_se_i")
#
#             if st.button("Predict", key="cd_se_i"):
#                 with cd_se_i_prediction_metrics:
#                     model = load(
#                         f"Streamlit_App/data/Classification_Models/CD_SE_I/{cd_se_i_model_name_to_file[cd_se_i_chosen_model]}")
#
#                     user_inputs = pd.DataFrame(columns=feature_selection_dataframe[0])
#                     user_inputs.loc[0, "MW"] = mw
#                     user_inputs.loc[0, "TPSA"] = tpsa
#                     user_inputs.loc[0, "XLogP"] = xlogp
#                     user_inputs.loc[0, "NHD"] = nhd
#                     user_inputs.loc[0, "NHA"] = nha
#                     user_inputs.loc[0, "NRB"] = nrb
#                     for side_effect in side_effects_chosen:
#                         user_inputs.loc[0, f"Side_Effect_{side_effect}"] = 1
#                     for indication in indications_chosen:
#                         user_inputs.loc[0, f"Indication_{indication}"] = 1
#                     user_inputs.fillna(0, inplace=True)
#
#                     result_column_section(model, cd_se_i_chosen_model, user_inputs, 'cd_se_i')

with training_process:
    st.subheader("Training & Testing Process Overview")
    st.markdown(render_svg("Streamlit_App/data/Plots/Model_Training.svg"), unsafe_allow_html=True)