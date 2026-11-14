"""Page Exploration : gabarit en attente des données et visualisations du groupe."""

import streamlit as st

from pages.commun import OPTIONS

st.title("Exploration des données")
st.caption("Emplacements prévus pour les données et visualisations du groupe.")
st.info("La source de données nettoyée n’est pas encore connectée à cette page.")
st.subheader("Filtres")
st.multiselect("Codes postaux", OPTIONS.get("code_postal_ban", []), disabled=True)
st.multiselect("Types de logement", OPTIONS.get("type_batiment", []), disabled=True)
st.caption("Les filtres seront activés après la connexion des données.")
st.subheader("Indicateurs clés")
for column, label in zip(st.columns(4), [
    "Nombre de diagnostics", "Surface médiane (m²)",
    "Consommation médiane (kWh/an)", "Part des classes F et G (%)",
]):
    column.metric(label, "—")
st.subheader("Visualisations prévues")
st.write(
    "Répartition des classes DPE (barres), distribution des surfaces (histogramme), "
    "consommation par classe DPE (boîtes à moustaches), "
    "surface et consommation (nuage de points)."
)
st.subheader("Carte interactive")
st.caption("La carte utilisera les coordonnées des diagnostics disponibles.")
st.subheader("Export des données et des graphiques")
st.caption("Les exports CSV des sélections et PNG des graphiques seront connectés aux visualisations.")
