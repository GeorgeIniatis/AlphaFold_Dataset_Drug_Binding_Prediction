# Linking QSAR-Based Drug Target Prediction with AlphaFold
Drug-target interactions (DTIs) refer to the interactions of chemical compounds and biological targets, 
proteins in our case, inside the human body. They play a crucial role in drug discovery and pharmacology, 
however, their experimental determination is time-consuming and limited due to funding and the difficulty 
of purifying proteins.
            
Unwanted or unexpected DTIs could cause severe side effects. Therefore, the creation of in silico machine 
learning models with high throughput that can quickly and confidently predict whether thousands of drugs and 
proteins bind together and how much could be crucial for medicinal chemistry and drug development, 
acting as a supplement to biological experiments.

**Original Aims**: The project aimed to gather publicly available data on known DTIs and place them into 
a new curated dataset. 
Then, using this new dataset, train multiple machine learning models using simple QSAR descriptors derived 
from a drug's chemical properties and a protein's sequence and 3D structural information extracted 
from [AlphaFold](https://alphafold.ebi.ac.uk/) to predict whether they bind together or not. 

**Actual Achievements**: A dataset of 163,080 DTIs was gathered using a variety of databases, 
libraries and biochemical APIs, subsets of which were used to train both our classification and regression 
models, evaluated using dummy models, holdout test sets and model interpretability tools. 
Classification models would try to predict whether a drug-protein pair would
bind together or not and Regression models would try to predict the logKd value. 
            
The models were then further split into "Baseline" and "Enhanced" with the former utilising just the QSAR
descriptors from drug and proteins and the latter utilising the 3D structural embeddings in addition
to the QSAR descriptors. This was naturally done in order to compare the effect, positive or negative, 
of the created structural embeddings to a baseline.
            
Unfortunately, our embeddings seemed to have little effect on our baseline models, 
which reasonably falls down to our embeddings creation process. Even though our embeddings did not have a 
significant impact, our high-throughput models could still be used to uncover some 
interesting relationships between drugs and proteins that could be later confirmed or 
rejected by molecular docking simulations and actual experimental trials.


**Important Links**
- Dissertation discussing the project's life-cycle ([Dissertation Link](https://drive.google.com/file/d/1PxtbJ2dam5OzJq-qUniG39A37cO22_3X/view?usp=share_link))
- Google Drive holding our models and datasets ([Google Drive Link](https://drive.google.com/drive/folders/1VjRcpX_pHmt70I8neKLktm2N8S_dSptj?usp=share_link))
- Streamlit web application created to showcase all the different models and our work ([Web App Link](https://alphafold-dataset-drug-binding-prediction.streamlit.app/))
