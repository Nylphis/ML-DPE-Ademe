"""Page Contexte : projet, données, répartition des étiquettes, performances des modèles."""

import altair as alt
import pandas as pd
import streamlit as st

from dpe.config import DATA_CLEAN, FEATURES, ORDRE_ETIQUETTES
from vues.commun import COULEURS_DPE, METADATA, MODELS, echelle_dpe, fmt_int


@st.cache_data(show_spinner=False)
def repartition_etiquettes() -> pd.DataFrame:
    """Nombre et part de logements par étiquette.

    Lecture de la seule colonne etiquette_dpe : empreinte mémoire minimale.

    Returns
    -------
    DataFrame
        Colonnes etiquette, logements, part ; une ligne par étiquette.
    """
    etiquettes = pd.read_csv(DATA_CLEAN, usecols=["etiquette_dpe"])["etiquette_dpe"]
    effectifs = etiquettes.value_counts().reindex(ORDRE_ETIQUETTES, fill_value=0)
    return pd.DataFrame({"etiquette": effectifs.index, "logements": effectifs.values,
                         "part": effectifs.values / effectifs.sum()})

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

if DATA_CLEAN.exists():
    repartition = repartition_etiquettes()
    # Barre « A-B » : couleur du A
    couleurs = [COULEURS_DPE[e.split("-")[0]] for e in ORDRE_ETIQUETTES]
    graphique = (
        alt.Chart(repartition, title="Répartition des logements par étiquette DPE")
        .mark_bar()
        .encode(
            x=alt.X("etiquette:N", sort=ORDRE_ETIQUETTES, title="Étiquette DPE", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("part:Q", title="Part des logements", axis=alt.Axis(format="%")),
            color=alt.Color("etiquette:N", scale=alt.Scale(domain=ORDRE_ETIQUETTES, range=couleurs), legend=None),
            tooltip=[alt.Tooltip("etiquette:N", title="Étiquette"),
                     alt.Tooltip("logements:Q", title="Logements", format=","),
                     alt.Tooltip("part:Q", title="Part", format=".1%")],
        )
    )
    st.altair_chart(graphique, width="stretch")
    st.caption("Plus de détails (filtres, carte, consommation) sur la page Exploration.")

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
