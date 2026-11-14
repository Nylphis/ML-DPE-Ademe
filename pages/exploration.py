"""Page Exploration des données DPE (filtres, KPI, graphiques, carte, exports)."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from dpe.config import DATA_CLEAN, ORDINAL, ORDRE_ETIQUETTES
from pages.commun import COULEURS_DPE, fmt_int

TAILLE_ECHANTILLON = 3_000  # nb de points pour la carte et le nuage de points

# A et B sont regroupées dans les données, on prend la couleur du A pour "A-B"
COULEURS_ETIQUETTES = {"A-B": COULEURS_DPE["A"],
                       **{e: COULEURS_DPE[e] for e in ORDRE_ETIQUETTES if e != "A-B"}}
ECHELLE_COULEURS = alt.Scale(domain=ORDRE_ETIQUETTES,
                             range=[COULEURS_ETIQUETTES[e] for e in ORDRE_ETIQUETTES])

COLONNES = [
    "etiquette_dpe", "conso_5_usages_ef", "code_postal_ban", "nom_commune_ban",
    "latitude", "longitude", "type_batiment", "surface_habitable_logement",
    "periode_construction", "type_energie_principale_chauffage", "generateur",
    "type_energie_principale_ecs", "qualite_isolation_menuiseries",
]

# variables proposées dans l'onglet Facteurs
FACTEURS = {
    "periode_construction": "Période de construction",
    "type_energie_principale_chauffage": "Énergie du chauffage",
    "generateur": "Type de chauffage",
    "type_energie_principale_ecs": "Énergie de l'eau chaude",
    "qualite_isolation_menuiseries": "Isolation des fenêtres",
    "code_postal_ban": "Code postal",
    "type_batiment": "Type de logement",
}


# chargement des données
# cache_resource -> une seule copie en mémoire pour tout le monde (limite de 512 Mo sur Render)
@st.cache_resource
def charger_donnees() -> pd.DataFrame:
    df = pd.read_csv(DATA_CLEAN, usecols=COLONNES, dtype={"code_postal_ban": str})

    # catégories ordonnées pour garder le bon ordre dans les graphiques
    df["etiquette_dpe"] = pd.Categorical(df["etiquette_dpe"], categories=ORDRE_ETIQUETTES, ordered=True)
    for col, ordre in ORDINAL.items():
        df[col] = pd.Categorical(df[col], categories=ordre, ordered=True)

    # colonnes texte en category pour prendre moins de place
    for col in ["code_postal_ban", "nom_commune_ban", "type_batiment",
                "type_energie_principale_chauffage", "generateur", "type_energie_principale_ecs"]:
        df[col] = df[col].astype("category")

    for col in ["conso_5_usages_ef", "latitude", "longitude", "surface_habitable_logement"]:
        df[col] = df[col].astype("float32")

    # conso au m², pour comparer des logements de tailles différentes
    df["conso_m2"] = (df["conso_5_usages_ef"] / df["surface_habitable_logement"]).astype("float32")
    return df


df = charger_donnees()

SURFACE_MAX = int(np.ceil(df["surface_habitable_logement"].quantile(0.99)))
CONSO_BORNES = df["conso_5_usages_ef"].quantile([0.01, 0.99]).tolist()


st.title("Exploration des données")
st.caption(
    "Diagnostics de performance énergétique (DPE) des logements de Lyon et d'une partie de la métropole, "
    "source ADEME. Les filtres ci-dessous mettent à jour toute la page."
)

# ---------- Filtres ----------
st.subheader("Filtres")
col1, col2 = st.columns(2)
with col1:
    codes_postaux = st.multiselect(
        "Codes postaux",
        options=sorted(df["code_postal_ban"].cat.categories),
        placeholder="Tous les codes postaux",
    )
    type_logement = st.radio("Type de logement", ["Tous", "appartement", "maison"], horizontal=True)
with col2:
    surface_min, surface_max = st.slider(
        "Surface habitable (m²)", min_value=0, max_value=SURFACE_MAX, value=(0, SURFACE_MAX), step=5,
    )
    exclure_extremes = st.checkbox(
        "Exclure les 1 % de consommations les plus faibles et les plus fortes", value=True,
    )

# on filtre avec un masque, sans toucher au tableau en cache
masque = df["surface_habitable_logement"].between(surface_min, surface_max)
if surface_max == SURFACE_MAX:  # curseur au max : on garde aussi les grandes surfaces
    masque |= df["surface_habitable_logement"] > SURFACE_MAX
if codes_postaux:
    masque &= df["code_postal_ban"].isin(codes_postaux)
if type_logement != "Tous":
    masque &= df["type_batiment"] == type_logement
if exclure_extremes:
    masque &= df["conso_5_usages_ef"].between(*CONSO_BORNES)

df_f = df[masque]

if df_f.empty:
    st.warning("Aucun logement ne correspond à ces filtres. Élargissez la sélection.")
    st.stop()

# échantillon pour la carte et le nuage de points (afficher 200 000 points serait trop lourd)
echantillon = df_f.sample(min(TAILLE_ECHANTILLON, len(df_f)), random_state=0)

# ---------- KPI ----------
st.subheader("Indicateurs clés")
st.caption(f"{fmt_int(len(df_f))} logements retenus sur {fmt_int(len(df))}.")

part_fg = df_f["etiquette_dpe"].isin(["F", "G"]).mean() * 100
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Nombre de diagnostics", fmt_int(len(df_f)))
k2.metric("Surface médiane", f"{df_f['surface_habitable_logement'].median():.0f} m²")
k3.metric("Consommation médiane", f"{fmt_int(int(df_f['conso_5_usages_ef'].median()))} kWh/an")
k4.metric("Consommation médiane au m²", f"{df_f['conso_m2'].median():.0f} kWh/m²/an")
k5.metric("Part des classes F et G", f"{part_fg:.1f} %")

st.caption("Pour enregistrer un graphique en PNG : menu « ⋯ » en haut à droite du graphique, puis « Save as PNG ».")

# ---------- Onglets ----------
onglet_ensemble, onglet_conso, onglet_facteurs, onglet_carte, onglet_donnees = st.tabs(
    ["Vue d'ensemble", "Consommation", "Facteurs", "Carte", "Données"]
)

# Vue d'ensemble : répartition des étiquettes
with onglet_ensemble:
    repartition = (
        df_f["etiquette_dpe"].value_counts(sort=False)
        .rename_axis("etiquette").reset_index(name="logements")
    )
    repartition["part"] = repartition["logements"] / repartition["logements"].sum()
    repartition["etiquette"] = repartition["etiquette"].astype(str)

    graphique_barres = (
        alt.Chart(repartition, title="Répartition des étiquettes DPE")
        .mark_bar()
        .encode(
            x=alt.X("etiquette:N", sort=ORDRE_ETIQUETTES, title="Étiquette DPE"),
            y=alt.Y("logements:Q", title="Nombre de logements"),
            color=alt.Color("etiquette:N", scale=ECHELLE_COULEURS, legend=None),
            tooltip=[alt.Tooltip("etiquette:N", title="Étiquette"),
                     alt.Tooltip("logements:Q", title="Logements", format=","),
                     alt.Tooltip("part:Q", title="Part", format=".1%")],
        )
    )
    st.altair_chart(graphique_barres, width="stretch")

# Consommation : histogramme, boxplots, nuage de points
with onglet_conso:
    # on calcule les classes avec numpy pour envoyer seulement 40 lignes au graphique
    surfaces = df_f["surface_habitable_logement"].dropna()
    effectifs, bords = np.histogram(surfaces.clip(upper=SURFACE_MAX), bins=40)
    histo = pd.DataFrame({"debut": bords[:-1], "fin": bords[1:], "logements": effectifs})

    graphique_histo = (
        alt.Chart(histo, title="Distribution des surfaces habitables")
        .mark_bar(color="#4c78a8")
        .encode(
            x=alt.X("debut:Q", bin="binned", title="Surface habitable (m²)"),
            x2="fin:Q",
            y=alt.Y("logements:Q", title="Nombre de logements"),
            tooltip=[alt.Tooltip("debut:Q", title="De (m²)", format=".0f"),
                     alt.Tooltip("fin:Q", title="À (m²)", format=".0f"),
                     alt.Tooltip("logements:Q", title="Logements", format=",")],
        )
    )
    st.altair_chart(graphique_histo, width="stretch")

    # boxplots sur l'échantillon
    mesure = st.radio(
        "Mesure", ["Consommation au m² (kWh/m²/an)", "Consommation totale (kWh/an)"], horizontal=True,
    )
    colonne_mesure = "conso_m2" if mesure.startswith("Consommation au m²") else "conso_5_usages_ef"
    boites = echantillon[["etiquette_dpe", colonne_mesure]].dropna().copy()
    boites["etiquette_dpe"] = boites["etiquette_dpe"].astype(str)

    graphique_boites = (
        alt.Chart(boites, title=f"{mesure} selon l'étiquette DPE")
        .mark_boxplot(outliers=False, size=40)
        .encode(
            x=alt.X("etiquette_dpe:N", sort=ORDRE_ETIQUETTES, title="Étiquette DPE"),
            y=alt.Y(f"{colonne_mesure}:Q", title=mesure),
            color=alt.Color("etiquette_dpe:N", scale=ECHELLE_COULEURS, legend=None),
        )
    )
    st.altair_chart(graphique_boites, width="stretch")
    st.caption(f"Calculé sur un échantillon de {fmt_int(len(echantillon))} logements, sans les valeurs extrêmes.")

    nuage = echantillon[["surface_habitable_logement", "conso_5_usages_ef", "etiquette_dpe"]].dropna().copy()
    nuage["etiquette_dpe"] = nuage["etiquette_dpe"].astype(str)

    graphique_nuage = (
        alt.Chart(nuage, title="Surface et consommation")
        .mark_circle(size=25, opacity=0.6)
        .encode(
            x=alt.X("surface_habitable_logement:Q", title="Surface habitable (m²)"),
            y=alt.Y("conso_5_usages_ef:Q", title="Consommation (kWh/an)"),
            color=alt.Color("etiquette_dpe:N", scale=ECHELLE_COULEURS, title="Étiquette"),
            tooltip=[alt.Tooltip("surface_habitable_logement:Q", title="Surface (m²)", format=".0f"),
                     alt.Tooltip("conso_5_usages_ef:Q", title="Consommation (kWh/an)", format=",.0f"),
                     alt.Tooltip("etiquette_dpe:N", title="Étiquette")],
        )
        .interactive()
    )
    st.altair_chart(graphique_nuage, width="stretch")
    st.caption(f"{fmt_int(len(nuage))} logements tirés au hasard parmi {fmt_int(len(df_f))}.")

# Facteurs : conso et étiquettes selon une variable choisie
with onglet_facteurs:
    facteur = st.selectbox("Variable à étudier", list(FACTEURS), format_func=FACTEURS.get)
    libelle = FACTEURS[facteur]

    par_modalite = (
        df_f.groupby(facteur, observed=True)
        .agg(logements=("etiquette_dpe", "size"), conso_m2=("conso_m2", "median"))
        .reset_index()
    )
    par_modalite = par_modalite[par_modalite["logements"] >= 30]  # on enlève les modalités trop rares
    par_modalite[facteur] = par_modalite[facteur].astype(str)
    ordre_modalites = par_modalite[facteur].tolist()

    if par_modalite.empty:
        st.info("Pas assez de logements pour cette variable avec les filtres actuels.")
    else:
        graphique_conso = (
            alt.Chart(par_modalite, title=f"Consommation médiane au m² selon : {libelle.lower()}")
            .mark_bar(color="#4c78a8")
            .encode(
                y=alt.Y(f"{facteur}:N", sort=ordre_modalites, title=libelle),
                x=alt.X("conso_m2:Q", title="Consommation médiane (kWh/m²/an)"),
                tooltip=[alt.Tooltip(f"{facteur}:N", title=libelle),
                         alt.Tooltip("conso_m2:Q", title="kWh/m²/an", format=".0f"),
                         alt.Tooltip("logements:Q", title="Logements", format=",")],
            )
        )
        st.altair_chart(graphique_conso, width="stretch")

        # barres empilées : part de chaque étiquette par modalité
        parts = (
            pd.crosstab(df_f[facteur], df_f["etiquette_dpe"], normalize="index")
            .stack().rename("part").reset_index()
        )
        parts.columns = ["modalite", "etiquette", "part"]
        parts["modalite"] = parts["modalite"].astype(str)
        parts["etiquette"] = parts["etiquette"].astype(str)
        parts = parts[parts["modalite"].isin(ordre_modalites)]
        parts["ordre"] = parts["etiquette"].map({e: i for i, e in enumerate(ORDRE_ETIQUETTES)})

        graphique_empile = (
            alt.Chart(parts, title=f"Répartition des étiquettes selon : {libelle.lower()}")
            .mark_bar()
            .encode(
                y=alt.Y("modalite:N", sort=ordre_modalites, title=libelle),
                x=alt.X("part:Q", stack="normalize", axis=alt.Axis(format="%"), title="Part des logements"),
                color=alt.Color("etiquette:N", scale=ECHELLE_COULEURS, title="Étiquette"),
                order=alt.Order("ordre:Q"),
                tooltip=[alt.Tooltip("modalite:N", title=libelle),
                         alt.Tooltip("etiquette:N", title="Étiquette"),
                         alt.Tooltip("part:Q", title="Part", format=".1%")],
            )
        )
        st.altair_chart(graphique_empile, width="stretch")
        st.caption("Les modalités de moins de 30 logements ne sont pas affichées.")

# Carte
with onglet_carte:
    afficher_carte = st.checkbox("Afficher la carte", value=True)
    if afficher_carte:
        points = echantillon[["latitude", "longitude", "etiquette_dpe"]].dropna().copy()
        points["couleur"] = points["etiquette_dpe"].astype(str).map(COULEURS_ETIQUETTES)
        st.map(points, latitude="latitude", longitude="longitude", color="couleur", size=20)
        st.caption(
            f"{fmt_int(len(points))} logements affichés, tirés au hasard parmi {fmt_int(len(df_f))}. "
            "Couleurs : étiquette DPE, du vert (A-B) au rouge (G)."
        )

# Données + export CSV
with onglet_donnees:
    st.dataframe(df_f.head(100), width="stretch", hide_index=True)
    st.caption("Aperçu des 100 premières lignes de la sélection.")


def csv_selection() -> bytes:
    # appelée seulement au clic sur le bouton
    return df_f.to_csv(index=False).encode("utf-8-sig")


st.download_button(
    "Télécharger les données filtrées (CSV)",
    data=csv_selection,
    file_name="dpe_lyon_selection.csv",
    mime="text/csv",
)