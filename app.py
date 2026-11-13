"""DPE Lyon : application Streamlit de simulation.

Lancement (depuis la racine du projet) : python -m streamlit run app.py
Prérequis : modèles entraînés avec `python run.py` (dossier models/).

Ce fichier ne contient que la configuration et la navigation. Chaque page a
son propre fichier dans pages/ ; les éléments communs (modèles, échelle DPE)
sont dans pages/commun.py. Pour ajouter une page : créer pages/<nom>.py, puis
ajouter une ligne st.Page ci-dessous.
"""

import streamlit as st

st.set_page_config(page_title="DPE Lyon", page_icon="🏠", layout="wide")

st.sidebar.title("DPE Lyon")
st.sidebar.caption("Projet M2 SISE · Version de développement")
navigation = st.navigation([
    st.Page("pages/contexte.py", title="Contexte", icon=":material/home:", default=True, url_path="contexte"),
    st.Page("pages/exploration.py", title="Exploration", icon=":material/bar_chart:", url_path="exploration"),
    st.Page("pages/prediction.py", title="Prédiction", icon=":material/house:", url_path="prediction"),
])
navigation.run()
