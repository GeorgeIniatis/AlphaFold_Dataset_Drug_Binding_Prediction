from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.page_helpers import *

title()
introduction = st.container()
dataset_creation_process = st.container()
model_discussion = st.container()
training_process = st.container()

with introduction:
    st.subheader("Embedding Model")
    st.markdown("""
                * In order to construct our embeddings we decided to construct a neural network that would be trained for a specific classification task involving as much proteins as possible in order to maximise the number of created embeddings
                * Once the model was trained we would then extract the protein structural embeddings from one of the layers
                """)

with dataset_creation_process:
    st.subheader("Dataset Creation Process")
    st.markdown("""
                * Downloaded the human proteins from [AlphaFold](https://alphafold.ebi.ac.uk/download) (UP000005640) - **23391 Proteins**
                * Kept only F1 from AlphaFold. **Left with 19966 Proteins**
                * Retrieved all the protein accession numbers and sequences
                * Extracted molecular function keywords using [UniProt API Calls](https://www.uniprot.org/help/programmatic_access) for each protein. **Left with 11222 Proteins**
                * Extracted protein sequence descriptors using [Protr](https://cran.r-project.org/web/packages/protr/vignettes/protr.html) R library
                * Extracted protein sequence embeddings from [UniProt](https://www.uniprot.org/help/embeddings) for each protein
                * Extracted amino acids descriptors from protein sequences using [Peptides](https://www.rdocumentation.org/packages/Peptides/versions/2.4.4) R library (Converted to numpy arrays)
                * Extracted amino acid embeddings from [UniProt](https://www.uniprot.org/help/embeddings) (Converted to numpy array)
                * Extracted protein PSSM using [Protr](https://cran.r-project.org/web/packages/protr/vignettes/protr.html) R library (Converted to numpy arrays)
                * Extracted contact maps from protein 3D structures using this [nanoHUB tool](https://nanohub.org/resources/contactmaps) (Converted to numpy arrays)
                * One-hot encoded all those molecular functions
                * Protein Sequence UniProt embeddings each entry given a column
                * Used PCA to reduce Tripeptide Descriptors from **8000** to **4813**
                * Removed **20** entries for missing protein sequence descriptors. **Left with 11202 Proteins**
                * Decided to simplify problem from multi-label to one-class classification. The molecular function we would be trying to predict would be "DNA Binding" as it was the most prevalent
                """)

    st.subheader("Training & Test Sets")
    st.markdown("""
                * No test set needed since this process is a means to an end to get protein structural embeddings and we do not really care of the network's predictive performance in this particular task
                * Training Set Size: **11034 (168 lost due to incorrect graphs)**
                """)

    st.subheader("Feature Selection")
    st.markdown("""
                * Used [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) to reduce the protein sequence descriptors from **7757** to **144**
                """)

with model_discussion:
    st.subheader("Neural Network")
    st.markdown("""
                * The neural network would make use of protein graphs protein sequence descriptors.
                * Each protein graph was constructed using the protein contact map, the amino acid descriptors and embeddings, and the position-specific scoring matrix (PSSM).
                """)

with training_process:
    st.subheader("Neural Network Architecture Overview")
    st.markdown("""
                * The following plot provides a high-level view of the neural network's architecture
                * **The protein structural embeddings were extracted from the second dense linear layer before the concatenation with the protein sequence descriptors**
                """)
    st.markdown(render_svg("Streamlit_App/data/Plots/Embeddings_NN_Architecture.svg"), unsafe_allow_html=True)

