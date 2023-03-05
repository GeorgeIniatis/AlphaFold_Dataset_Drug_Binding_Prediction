from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.page_helpers import *

title()

methodology_overview = st.container()

with methodology_overview:
    st.subheader("Methodology Overview")
    st.markdown("""
                    * The following plot provides a high-level view of the whole project as a whole
                    """)
    st.markdown(render_svg("Streamlit_App/data/Plots/Methodology.svg"), unsafe_allow_html=True)