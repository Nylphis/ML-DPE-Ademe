"""DPE Lyon : application Streamlit, configuration et menu.

Une page par fichier dans vues/ ; éléments communs dans vues/commun.py.
Nouvelle page : créer vues/<nom>.py, puis ajouter une ligne st.Page ci-dessous.
Dossier nommé vues/ et non pages/ : un dossier pages/ déclencherait le menu
automatique de Streamlit au démarrage.

Lancement (depuis la racine du projet) : python -m streamlit run app.py
Prérequis : modèles dans models/ (python run.py).
"""

import streamlit as st

st.set_page_config(page_title="DPE Lyon", page_icon="🏠", layout="wide")

st.sidebar.title("DPE Lyon")
st.sidebar.caption("Projet M2 SISE · Version de développement")
navigation = st.navigation([
    st.Page("vues/contexte.py", title="Contexte", icon=":material/home:", default=True, url_path="contexte"),
    st.Page("vues/exploration.py", title="Exploration", icon=":material/bar_chart:", url_path="exploration"),
    st.Page("vues/prediction.py", title="Prédiction", icon=":material/house:", url_path="prediction"),
    st.Page("vues/administration.py", title="Administration", icon=":material/settings:", url_path="administration"),
])
navigation.run()
