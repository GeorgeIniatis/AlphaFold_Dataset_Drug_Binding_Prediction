from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.page_helpers import *

title()
project_introduction = st.container()
about_us = st.container()

with project_introduction:
    st.subheader("Introduction")
    st.markdown("""
            Drug-target interactions (DTIs) refer to the interactions of chemical compounds and biological targets, 
            proteins in our case, inside the human body. They play a crucial role in drug discovery and pharmacology, 
            however, their experimental determination with methods, such as fluorescence assays, 
            is time-consuming and limited due to funding and the difficulty of purifying proteins.
            
            Unwanted or unexpected DTIs could cause severe side effects, therefore, the creation of in silico machine 
            learning models with high throughput that can quickly and confidently predict whether thousands of drugs and 
            proteins bind together and how much could be crucial for medicinal chemistry and drug development, 
            acting as a supplement to biological experiments.
            """)
    st.markdown("""
            **Original Aims**: The project aimed to gather publicly available data on known drug-target interactions and 
            place them into a new curated dataset. Then, using this new dataset, train multiple machine learning models 
            using simple QSAR descriptors derived from a drug's chemical properties and a protein's sequence and 3D 
            structural embeddings extracted from [AlphaFold](https://alphafold.ebi.ac.uk/) 
            protein structures to predict whether they bind together or not. 
            """)
    st.markdown("""
            **Actual Achievements**: A dataset of 163,080 publicly available DTIs was gathered from 
            [PubChem](https://pubchem.ncbi.nlm.nih.gov/). The models built were split up into two categories, 
            Classification and Regression. Classification models would try to predict whether a drug-protein pair would
            bind together or not and Regression models would try to predict the logKd value. 
            
            The models were then further split into "Baseline" and "Enhanced" with the former utilising just the QSAR
            descriptors from drug and proteins and the latter utilising the 3D structural embeddings in addition
            to the QSAR descriptors. This was naturally done in order to compare the effect, positive or negative, 
            of the created structural embeddings to a baseline.
            """)

with about_us:
    st.subheader("About")
    st.markdown("""
                - Created by George Iniatis as part of a 5th year computer science project at the University of Glasgow
                - Supervised by [Dr. Jake Lever](https://www.gla.ac.uk/schools/computing/staff/jakelever/)
                """)
    st.subheader("Useful Links")
    st.markdown("""
                - [GitHub Page](https://github.com/GeorgeIniatis/AlphaFold_Dataset_Drug_Binding_Prediction) 
                - [GitHub Wiki](https://github.com/GeorgeIniatis/AlphaFold_Dataset_Drug_Binding_Prediction/wiki) 
                - [Google Drive](https://drive.google.com/drive/folders/1VjRcpX_pHmt70I8neKLktm2N8S_dSptj?usp=share_link) - Holding all the trained models, datasets and embeddings
                """)