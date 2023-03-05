from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.page_helpers import *


title()
references = st.container()

with references:
    st.subheader("Libraries")
    st.markdown("""
                - [Streamlit](https://streamlit.io/)
                - [Scikit-Learn](https://scikit-learn.org/stable/)
                - [Scikit-Optimize](https://scikit-optimize.github.io/stable/)
                - [Numpy](https://numpy.org/)
                - [Pandas](https://pandas.pydata.org/)
                - [Plotly](https://plotly.com/)
                - [ELI5](https://eli5.readthedocs.io/en/latest/overview.html)
                - [LIME](https://lime-ml.readthedocs.io/en/latest/)
                - [Protr](https://cran.r-project.org/web/packages/protr/vignettes/protr.html)
                - [Peptides](https://www.rdocumentation.org/packages/Peptides/versions/2.4.4)
                - [nanoHUB's Protein Contact Maps Library](https://nanohub.org/resources/contactmaps)
                 """)

    st.subheader("APIs")
    st.markdown("""
                - [PubChem's PUG REST](https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest)
                - [UniProt's REST API](https://www.uniprot.org/help/programmatic_access)
                """)

    st.subheader("Data Sources")
    st.markdown("""
                - [AlphaFold](https://alphafold.ebi.ac.uk/)
                - [PubChem](https://pubchem.ncbi.nlm.nih.gov/)
                - [UniProt](https://www.uniprot.org/)
                """)

    st.subheader("Tutorials")
    st.markdown(
        """
        - [Misra Turp's Streamlit Playlist](https://www.youtube.com/watch?v=-IM3531b1XU&list=PLM8lYG2MzHmTATqBUZCQW9w816ndU0xHc&ab_channel=M%C4%B1sraTurp)
        - [Renu Khandelwal's Article on LIME](https://towardsdatascience.com/developing-trust-in-machine-learning-models-predictions-e49b0064abab)
        """
    )
