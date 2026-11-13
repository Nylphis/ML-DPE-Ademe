"""DPE Lyon — application Streamlit de simulation.

Lancement (depuis la racine du projet) : python -m streamlit run app.py
Prérequis : modèles entraînés avec `python run.py` (dossier models/).

Le formulaire suit exactement le contrat d'entrée du modèle (dpe/config.py).
Les options proposées sont lues dans models/metadata.json, donc toujours
cohérentes avec les catégories vues à l'entraînement. Seuls les champs de
REQUIRED sont obligatoires : un champ laissé vide est envoyé au modèle comme
« inconnu », une situation qu'il a apprise sur les données ADEME.
"""

import csv
import io
import json
import logging
import math
from typing import Any

import joblib
import streamlit as st

from dpe.config import BINARY, CLF_PATH, FEATURES, METADATA_PATH, REG_PATH, REQUIRED
from dpe.features import to_model_frame

st.set_page_config(page_title="DPE Lyon", page_icon="🏠", layout="wide")

LABELS = {
    "code_postal_ban": "Code postal",
    "type_batiment": "Type de logement",
    "surface_habitable_logement": "Surface habitable (m²)",
    "periode_construction": "Période de construction",
    "nombre_niveau_logement": "Nombre de niveaux du logement",
    "type_energie_principale_chauffage": "Énergie principale du chauffage",
    "generateur": "Type de chauffage principal",
    "type_installation_chauffage": "Installation de chauffage",
    "type_energie_principale_ecs": "Énergie principale de l’eau chaude",
    "type_installation_ecs": "Installation d’eau chaude",
    "isolation_toiture": "Toiture isolée",
    "qualite_isolation_menuiseries": "Qualité de l’isolation des fenêtres",
    "logement_traversant": "Logement traversant",
    "climatisation": "Climatisation",
}

LIBELLES_ETIQUETTE = {"A-B": "A ou B"}

# Jamais renseignés pour les maisons dans les données ADEME : non transmis au modèle pour une maison
CHAMPS_APPARTEMENT = ["type_installation_chauffage", "type_installation_ecs"]
AIDE_APPARTEMENT = "Ignoré pour une maison : ce champ n’est jamais renseigné pour les maisons dans les données ADEME."

# État propre à chaque session. Aucun renseignement n'est écrit sur le disque.
st.session_state.setdefault("validated_housing", None)
st.session_state.setdefault("prediction_result", None)


# ------------------------------------------------
# Chargement des modèles (une seule fois par processus)
# ------------------------------------------------
@st.cache_resource
def load_models() -> tuple[Any, Any, dict] | None:
    if not all(p.exists() for p in (CLF_PATH, REG_PATH, METADATA_PATH)):
        return None
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return joblib.load(CLF_PATH), joblib.load(REG_PATH), metadata


MODELS = load_models()
METADATA: dict = MODELS[2] if MODELS else {}
OPTIONS: dict[str, list] = METADATA.get("options", {})


def predict_logement(features: dict[str, Any]) -> dict[str, Any]:
    """Passe une ligne aux deux pipelines (prétraitement inclus)."""
    clf, reg, _ = MODELS
    X = to_model_frame(features)
    return {
        "etiquette_dpe": str(clf.predict(X)[0]),
        "conso_5_usages_ef": float(reg.predict(X)[0]),
    }


# ------------------------------------------------
# Widgets
# ------------------------------------------------
def label(field: str) -> str:
    return LABELS[field] + (" *" if field in REQUIRED else "")


def placeholder(field: str) -> str:
    return "Choisir une valeur" if field in REQUIRED else "Je ne sais pas"


def select_category(field: str, help_text: str | None = None) -> Any:
    return st.selectbox(
        label(field),
        options=OPTIONS[field],
        index=None,
        placeholder=placeholder(field),
        help=help_text,
        key=field,
        persist_state="session",
    )


def select_binary(field: str, help_text: str) -> int | None:
    return st.selectbox(
        label(field),
        options=[0, 1],
        index=None,
        format_func=lambda x: {0: "Non", 1: "Oui"}[x],
        placeholder=placeholder(field),
        help=help_text,
        key=field,
        persist_state="session",
    )


def validate_housing(data: dict[str, Any]) -> list[str]:
    errors = []
    for field in REQUIRED:
        if data.get(field) is None:
            errors.append(f"Veuillez renseigner : {LABELS[field]}.")
    for field, values in OPTIONS.items():
        if data.get(field) is not None and data[field] not in values:
            errors.append(f"Valeur non reconnue : {LABELS[field]}.")

    surface = data.get("surface_habitable_logement")
    if surface is not None and not (isinstance(surface, (int, float)) and math.isfinite(surface) and surface > 0):
        errors.append("La surface habitable doit être supérieure à zéro.")
    levels = data.get("nombre_niveau_logement")
    if levels is not None and (not isinstance(levels, int) or isinstance(levels, bool) or levels < 1):
        errors.append("Le nombre de niveaux doit être un entier supérieur ou égal à 1.")
    for field in BINARY:
        if data.get(field) not in (None, 0, 1):
            errors.append(f"Valeur non reconnue : {LABELS[field]}.")
    return errors


def housing_csv(data: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FEATURES)
    writer.writeheader()
    writer.writerow(data)
    return stream.getvalue()


def sans_objet(payload: dict[str, Any], field: str) -> bool:
    return payload["type_batiment"] == "maison" and field in CHAMPS_APPARTEMENT


def display_value(field: str, value: Any) -> str:
    if value is None:
        return "Non renseigné (inconnu)"
    if field in BINARY:
        return "Oui" if value == 1 else "Non"
    return str(value)


COULEURS_DPE = {"A": "#009c6d", "B": "#52b153", "C": "#78bd76", "D": "#f4e70f",
                "E": "#f0b40f", "F": "#eb8235", "G": "#d7221f"}


def echelle_dpe(etiquette: str | None = None) -> str:
    """Image SVG de l'échelle DPE (A à G). Si une étiquette est donnée, les autres classes sont estompées."""
    surlignees = etiquette.split("-") if etiquette else list(COULEURS_DPE)
    barres = []
    for i, (lettre, couleur) in enumerate(COULEURS_DPE.items()):
        y, largeur = 24 + i * 34, 90 + i * 35
        opacite = 1 if lettre in surlignees else 0.25
        texte = "#333" if lettre == "D" else "#fff"
        barres.append(
            f'<g opacity="{opacite}"><rect x="0" y="{y}" width="{largeur}" height="28" rx="4" fill="{couleur}"/>'
            f'<text x="{largeur - 12}" y="{y + 20}" text-anchor="end" font-size="18" font-weight="bold" '
            f'fill="{texte}">{lettre}</text></g>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="282" font-family="sans-serif">'
        '<text x="0" y="14" font-size="13" fill="#888">Logement économe</text>'
        + "".join(barres)
        + '<text x="0" y="278" font-size="13" fill="#888">Logement énergivore</text></svg>'
    )


def fmt_int(n: Any) -> str:
    return f"{n:,}".replace(",", " ") if isinstance(n, int) else "—"


# ------------------------------------------------
# Pages
# ------------------------------------------------
def contexte() -> None:
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


def exploration() -> None:
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


def prediction() -> None:
    st.title("Prédiction")
    if not MODELS:
        st.error("Les modèles ne sont pas disponibles. Lancez `python run.py` puis relancez l’application.")
        return

    st.write(
        "Décrivez votre logement. Les champs marqués * sont obligatoires ; "
        "laissez vides ceux que vous ne connaissez pas."
    )
    data: dict[str, Any] = {}
    with st.form("formulaire_logement", clear_on_submit=False):
        st.subheader("Le logement")
        left, right = st.columns(2)
        with left:
            data["code_postal_ban"] = select_category("code_postal_ban")
            data["surface_habitable_logement"] = st.number_input(
                label("surface_habitable_logement"), min_value=0.1,
                value=None, step=1.0, placeholder="Ex. : 75",
                key="surface_habitable_logement", persist_state="session",
            )
            data["nombre_niveau_logement"] = st.number_input(
                label("nombre_niveau_logement"), min_value=1, value=None, step=1,
                placeholder="Je ne sais pas",
                help="Nombre de niveaux à l’intérieur du logement : 1 pour un logement sur un seul niveau, 2 pour un duplex. Ce n’est pas le numéro de l’étage.",
                key="nombre_niveau_logement", persist_state="session",
            )
        with right:
            data["type_batiment"] = select_category("type_batiment")
            data["periode_construction"] = select_category("periode_construction")

        st.divider()
        st.subheader("Le chauffage")
        data["type_energie_principale_chauffage"] = select_category("type_energie_principale_chauffage")
        data["generateur"] = select_category("generateur")
        data["type_installation_chauffage"] = select_category("type_installation_chauffage", AIDE_APPARTEMENT)

        st.divider()
        st.subheader("L’eau chaude sanitaire")
        st.caption("L’eau chaude sanitaire (ECS) est l’eau chaude utilisée aux robinets et à la douche.")
        left, right = st.columns(2)
        with left:
            data["type_energie_principale_ecs"] = select_category("type_energie_principale_ecs")
        with right:
            data["type_installation_ecs"] = select_category("type_installation_ecs", AIDE_APPARTEMENT)

        st.divider()
        st.subheader("L’isolation et la climatisation")
        left, right = st.columns(2)
        with left:
            data["isolation_toiture"] = select_binary(
                "isolation_toiture", "Indiquez si la toiture du logement est isolée."
            )
            data["logement_traversant"] = select_binary(
                "logement_traversant", "Un logement traversant possède des ouvertures sur deux façades opposées."
            )
        with right:
            data["qualite_isolation_menuiseries"] = select_category("qualite_isolation_menuiseries")
            data["climatisation"] = select_binary(
                "climatisation", "Indiquez si au moins une pièce du logement est climatisée. "
                "Si vous ne savez pas, le modèle considère le logement comme non climatisé, "
                "comme dans les données ADEME."
            )
        submitted = st.form_submit_button(
            "Valider les informations", key="submit_housing", type="primary", width="stretch",
        )

    if submitted:
        errors = validate_housing(data)
        if errors:
            # Une soumission invalide ne conserve pas un ancien résultat visible.
            st.session_state["validated_housing"] = None
            st.session_state["prediction_result"] = None
            for error in errors:
                st.error(error)
        else:
            payload = {name: data[name] for name in FEATURES}
            for field in CHAMPS_APPARTEMENT:
                if sans_objet(payload, field):
                    payload[field] = None

            st.session_state["validated_housing"] = payload
            st.session_state["prediction_result"] = None

            try:
                st.session_state["prediction_result"] = predict_logement(payload)
            except Exception:  # noqa: BLE001
                # Détail de l'erreur dans la console du serveur, pas dans la page
                logging.exception("Échec de la prédiction")
                st.error("L’estimation n’a pas pu être calculée.")

    payload = st.session_state["validated_housing"]
    if payload is None:
        st.caption("Complétez le formulaire, puis validez les informations.")
        return

    st.success("Les informations ont bien été prises en compte.")
    st.subheader("Estimation")
    result = st.session_state["prediction_result"]
    left, right = st.columns(2)
    if result is None:
        left.metric("Étiquette DPE estimée", "—")
        right.metric("Énergie finale estimée (kWh/an)", "—")
    else:
        etiquette = result["etiquette_dpe"]
        conso = result["conso_5_usages_ef"]
        left.metric("Étiquette DPE estimée", LIBELLES_ETIQUETTE.get(etiquette, etiquette))
        left.image(echelle_dpe(etiquette))
        right.metric(
            "Énergie finale estimée (kWh/an)", f"{conso:,.0f}".replace(",", " "),
            help=f"Soit environ {conso / payload['surface_habitable_logement']:,.0f} kWh/m²/an.".replace(",", " "),
        )
        st.caption(
            "L’étiquette et la consommation sont estimées par deux modèles distincts : "
            "elles peuvent ne pas être parfaitement cohérentes entre elles."
        )
        inconnus = [LABELS[f] for f in FEATURES if payload[f] is None and not sans_objet(payload, f)]
        if inconnus:
            st.info(
                f"{len(inconnus)} caractéristique(s) non renseignée(s) : "
                + ", ".join(inconnus).lower()
                + ". L’estimation est d’autant moins précise que le nombre de champs vides est grand."
            )

    st.subheader("Dernières informations validées")
    st.table(
        [{"Caractéristique": LABELS[name],
          "Valeur": "Sans objet pour une maison" if sans_objet(payload, name) else display_value(name, payload[name])}
         for name in FEATURES],
        alt="Récapitulatif des caractéristiques du logement validées",
    )

    st.subheader("Exporter les informations validées")
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Télécharger en CSV", housing_csv(payload).encode("utf-8-sig"),
            file_name="logement.csv", mime="text/csv", key="download_csv",
        )
    with right:
        st.download_button(
            "Télécharger en JSON", json.dumps(payload, ensure_ascii=False, indent=2),
            file_name="logement.json", mime="application/json", key="download_json",
        )


st.sidebar.title("DPE Lyon")
st.sidebar.caption("Projet M2 SISE · Version de développement")
navigation = st.navigation([
    st.Page(contexte, title="Contexte", icon=":material/home:", default=True, url_path="contexte"),
    st.Page(exploration, title="Exploration", icon=":material/bar_chart:", url_path="exploration"),
    st.Page(prediction, title="Prédiction", icon=":material/house:", url_path="prediction"),
])
navigation.run()
