from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.page_helpers import *

title()
classification_dataset = st.container()
regression_dataset = st.container()
creation_process = st.container()

with classification_dataset:
    st.subheader("Classification Dataset with Feature Selection Sample")
    st.write(pd.read_pickle("Streamlit_App/data/Datasets/Classification_Dataset_Feature_Selection.pkl"))

    st.subheader("Counts")
    st.markdown(f"- **Total DTIs: 163080**  \n"
                f"- **Binding Count: 112597**  \n"
                f"- **Non-Binding Count: 50483**  \n"
                f"- **Class Imbalance: 2:1**  \n"
                f"- **Unique Proteins: 3459** \n"
                f"- **Unique Drugs: 99948** \n"
                f"- **Number of features: 388**")

    st.subheader("Principal Component Analysis")
    st.markdown(render_svg("Streamlit_App/data/Plots/Classification_PCA.svg"), unsafe_allow_html=True)

with regression_dataset:
    st.subheader("Regression Dataset with Feature Selection Sample")
    st.write(pd.read_pickle("Streamlit_App/data/Datasets/Regression_Dataset_Feature_Selection.pkl"))

    st.subheader("Counts")
    st.markdown(f"- **Total DTIs: 20372**  \n"
                f"- **Binding Count: 5365**  \n"
                f"- **Non-Binding Count:15007**  \n"
                f"- **Class Imbalance: 1:3**  \n"
                f"- **Unique Proteins: 841** \n"
                f"- **Unique Drugs: 2279** \n"
                f"- **Number of features: 693**")

    st.subheader("Principal Component Analysis")
    st.markdown(render_svg("Streamlit_App/data/Plots/Regression_PCA.svg"), unsafe_allow_html=True)

with creation_process:
    st.subheader("Dataset Creation Process")
    st.markdown(
        """
        * Downloaded the human proteins from [AlphaFold](https://alphafold.ebi.ac.uk/download) (UP000005640) - **23391 Proteins**
        * Retrieved all the protein accession numbers and sequences
        * Using these accession numbers [PubChem API Calls](https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest) were made to retrieve the protein's name
        * Removed those not found and any duplicates (Kept only F1 from AlphaFold). **Left with 19966 Proteins**
        * For each protein left [PubChem API Calls](https://pubchemdocs.ncbi.nlm.nih.gov/pug-rest) were used to retrieve their drug interactions. Max number of interactions per protein were set to 100. If the number of interactions exceeded that number a random selection of 100 interactions was made
        * Drug Target Interactions: **190028** , Unique Proteins: **3875**, Unique Drugs: **117585**
        * Extracted all drug descriptors available from [PubChem](https://pubchem.ncbi.nlm.nih.gov/) for each unique drug using API calls
        * Extracted protein sequence descriptors using [Protr](https://cran.r-project.org/web/packages/protr/vignettes/protr.html) R library
        * Extracted protein sequence embeddings from [UniProt](https://www.uniprot.org/help/embeddings) for each protein
        * Removed **26948** entries for missing descriptors either drug or protein ones. Dataset site after removal: **163080**
           * **Classification Problem**: Binding DTIs: **112597**, Non-Binding DTIs: **50483**, Class imbalance roughly: **2:1**
           * **Regression Problem (Kd)**: Binding DTIs: **5365**, Non-Binding DTIs: **15007**, Class imbalance roughly: **1:3**
        * Drug Molecular Fingerprints converted to binary, prefix and padding removed, and each given a column
        * Protein Sequence UniProt embeddings each entry given a column
        * Used PCA to reduce Tripeptide Descriptors from 8000 to 2616
        """)

    st.subheader("Training & Test Sets")
    st.markdown("""
                * Given the nature of our problem, many proteins can be associated with many drugs, we could not do the traditional 80/20 split.
                   * Decided to take a small subset of our dataset as our test set and remove any proteins and drugs associated with it from the training set. This has of course led to the loss of a substantial number of DTIs but this process would better allow us to evaluate the trained models' real world predictive performances.
                   * **Classification**: Training set size: **99705**, Test set size: **816**
                   * **Regression**: Training set size: **10956**, Test set size: **102**
                """)

    st.subheader("Feature Selection")
    st.markdown("""
                * Used [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) on a sample of 10000 entries from the **classification** training set to reduce the features from **6474** to **388**.
                * Used [RFECV](https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html) on a sample of 2000 entries from the **regression** training set to reduce the features from **6474** to **693**
                """)


