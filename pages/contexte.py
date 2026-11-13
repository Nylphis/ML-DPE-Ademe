"""Page Contexte : présentation du projet, des données et des performances des modèles."""

import streamlit as st

from dpe.config import FEATURES
from pages.commun import METADATA, MODELS, echelle_dpe, fmt_int

st.title("DPE Lyon")
st.caption("Explorer les logements lyonnais et estimer leur performance énergétique.")
st.header("Le projet")
st.write(
    "À partir des caractéristiques d’un logement, l’application estime "
    "son étiquette DPE et sa consommation annuelle d’énergie finale."
)

st.subheader("Les données")
donnees = METADATA.get("donnees", {})
a, b, c = st.columns(3)
a.metric("Diagnostics récupérés", fmt_int(donnees.get("n_brut")))
b.metric("Diagnostics conservés après nettoyage", fmt_int(donnees.get("n_clean")))
c.metric("Caractéristiques en entrée", len(FEATURES))
st.caption(
    "Logements (maisons et appartements) de Lyon et d’une partie de la métropole, "
    "DPE version 2.1 et suivantes. Les caractéristiques non renseignées dans un "
    "diagnostic sont conservées comme « inconnues » : le modèle apprend aussi à "
    "estimer sans elles."
)
par_type = donnees.get("par_type_batiment", {}).get("clean", {})
if par_type:
    st.caption("Répartition : " + ", ".join(
        f"{fmt_int(n)} {t}s" for t, n in sorted(par_type.items(), key=lambda x: -x[1])) + ".")

st.subheader("Deux objectifs")
texte, image = st.columns([2, 1])
texte.write(
    "Classification : estimer la classe DPE (A et B sont regroupées, la classe A "
    "étant trop rare). Régression : estimer la consommation annuelle d’énergie "
    "finale en kWh/an, toutes énergies confondues."
)
image.image(echelle_dpe(), caption="Les sept classes du DPE")
if MODELS:
    m = METADATA["metriques"]
    mae = fmt_int(round(m["regression"]["mae"]))
    st.caption(
        f"Performances sur le jeu de test : étiquette exacte dans "
        f"{m['classification']['accuracy']:.0%} des cas, à une classe près dans "
        f"{m['classification']['accuracy_1_classe']:.0%} des cas ; erreur moyenne "
        f"sur la consommation de {mae} kWh/an (R² = {m['regression']['r2']:.2f})."
    )
else:
    st.warning("Les modèles ne sont pas encore entraînés : lancez `python run.py`.")

st.link_button("Consulter les données DPE de l’ADEME", "https://data.ademe.fr/")
st.caption("Les estimations du modèle ne remplacent pas un diagnostic réglementaire.")
