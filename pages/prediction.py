"""Page Prédiction : formulaire du logement, estimation de l'étiquette et de la consommation.

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

import streamlit as st

from dpe.config import BINARY, FEATURES, REQUIRED
from dpe.features import to_model_frame
from pages.commun import MODELS, OPTIONS, echelle_dpe

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


# ------------------------------------------------
# Page
# ------------------------------------------------
st.title("Prédiction")
if not MODELS:
    st.error("Les modèles ne sont pas disponibles. Lancez `python run.py` puis relancez l’application.")
    st.stop()

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
    st.stop()

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
