"""DPE Lyon — squelette Streamlit et formulaire complet de 15 variables.

Lancement : python -m streamlit run app.py
Les catégories sont issues du CSV du groupe, limité aux maisons/appartements.
Les cibles ne sont jamais demandées à l'utilisateur.
"""

import csv
import io
import json
import math
from typing import Any

import streamlit as st

st.set_page_config(page_title="DPE Lyon", page_icon="🏠", layout="wide")

# Ordre contractuel des 15 entrées. À reprendre dans le pipeline de prédiction.
FEATURES = [
    "code_postal_ban", "type_batiment", "surface_habitable_logement",
    "periode_construction", "nombre_niveau_logement",
    "type_energie_principale_chauffage", "type_generateur_chauffage_principal",
    "type_installation_chauffage", "type_energie_principale_ecs",
    "type_installation_ecs", "isolation_toiture",
    "qualite_isolation_menuiseries", "type_ventilation", "surface_climatisee",
    "logement_traversant",
]

OPTIONS = {'code_postal_ban': ['69001',
                     '69002',
                     '69003',
                     '69004',
                     '69005',
                     '69006',
                     '69007',
                     '69008',
                     '69009'],
 'type_batiment': ['appartement', 'maison'],
 'periode_construction': ['avant 1948',
                          '1948-1974',
                          '1975-1977',
                          '1978-1982',
                          '1983-1988',
                          '1989-2000',
                          '2001-2005',
                          '2006-2012',
                          '2013-2021',
                          'après 2021'],
 'type_energie_principale_chauffage': ['Bois – Bûches',
                                       'Bois – Granulés (pellets) ou briquettes',
                                       'Bois – Plaquettes forestières',
                                       'Fioul domestique',
                                       'GPL',
                                       'Gaz naturel',
                                       'Réseau de Chauffage urbain',
                                       'Électricité'],
 'type_generateur_chauffage_principal': ['Autres émetteurs à effet joule',
                                         'Chaudière bois granulés 1978-1994',
                                         'Chaudière bois granulés 2004-2012',
                                         'Chaudière bois granulés 2013-2019',
                                         'Chaudière bois plaquette après 2019',
                                         'Chaudière fioul basse température 1991-2015',
                                         'Chaudière fioul classique 1970-1975',
                                         'Chaudière fioul classique 1981-1990',
                                         'Chaudière fioul standard 1991-2015',
                                         'Chaudière fioul à condensation 1996-2015',
                                         'Chaudière gaz basse température 1991-2000',
                                         'Chaudière gaz basse température 2001-2015',
                                         'Chaudière gaz basse température après 2015',
                                         'Chaudière gaz classique 1981-1985',
                                         'Chaudière gaz classique 1986-1990',
                                         'Chaudière gaz classique avant 1981',
                                         'Chaudière gaz standard 1991-2000',
                                         'Chaudière gaz standard 2001-2015',
                                         'Chaudière gaz standard après 2015',
                                         'Chaudière gaz à condensation 1981-1985',
                                         'Chaudière gaz à condensation 1986-2000',
                                         'Chaudière gaz à condensation 2001-2015',
                                         'Chaudière gaz à condensation après 2015',
                                         'Chaudière gpl/propane/butane à condensation 2001-2015',
                                         'Chaudière électrique',
                                         'Chaudière(s) bois multi bâtiment modélisée comme un '
                                         'réseau de chaleur',
                                         'Chaudière(s) gaz multi bâtiment modélisée comme un '
                                         'réseau de chaleur',
                                         'Convecteur électrique NFC, NF** et NF***',
                                         'PAC air/air installée avant 2008',
                                         'PAC air/air installée entre 2008 et 2014',
                                         'PAC air/air installée à partir de 2015',
                                         'PAC air/eau installée après 2017',
                                         'PAC air/eau installée avant 2008',
                                         'PAC géothermique installée avant 2008',
                                         'Panneau rayonnant électrique NFC, NF** et NF***',
                                         'Plancher ou plafond rayonnant électrique avec régulation '
                                         'terminale',
                                         'Plancher ou plafond rayonnant électrique sans régulation '
                                         'terminale',
                                         'Poêle à granulés flamme verte installé entre 2012 et '
                                         '2019',
                                         'Radiateur électrique à accumulation',
                                         'Réseau de chaleur isolé',
                                         'Réseau de chaleur non isolé',
                                         'convecteur bi-jonction',
                                         'radiateur à gaz indépendant ou autonome après 2006',
                                         'radiateur à gaz indépendant ou autonome avant 2006',
                                         'radiateur électrique NFC, NF** et NF***',
                                         'réseau de chaleur non répertorié ou inconnu'],
 'type_installation_chauffage': ['collectif', 'individuel', 'mixte (collectif-individuel)'],
 'type_energie_principale_ecs': ['Bois – Bûches',
                                 'Bois – Granulés (pellets) ou briquettes',
                                 'Fioul domestique',
                                 'GPL',
                                 'Gaz naturel',
                                 'Non affecté',
                                 'Réseau de Chauffage urbain',
                                 'Électricité'],
 'type_installation_ecs': ['collectif', 'individuel', 'mixte (collectif-individuel)'],
 'qualite_isolation_menuiseries': ['bonne', 'insuffisante', 'moyenne', 'très bonne'],
 'type_ventilation': ['Non affecté',
                      'VMC Basse pression Auto-réglable',
                      'VMC Basse pression Hygro A',
                      'VMC Basse pression Hygro B',
                      'VMC DF collective avec échangeur avant 2013',
                      'VMC DF collective avec échangeur à partir de 2013',
                      'VMC DF sans échangeur après 2012',
                      'VMC DF sans échangeur avant 2013',
                      'VMC SF Auto réglable après 2012',
                      'VMC SF Auto réglable avant 1982',
                      'VMC SF Auto réglable de 1982 à 2000',
                      'VMC SF Auto réglable de 2001 à 2012',
                      'VMC SF Gaz après 2012',
                      'VMC SF Gaz avant  2001',
                      'VMC SF Gaz de 2001 à 2012',
                      'VMC SF Hygro A après 2012',
                      'VMC SF Hygro A avant 2001',
                      'VMC SF Hygro A de 2001 à 2012',
                      'VMC SF Hygro B après 2012',
                      'VMC SF Hygro B avant  2001',
                      'VMC SF Hygro B de 2001 à 2012',
                      'Ventilation hybride après 2012',
                      'Ventilation hybride avant  2001',
                      'Ventilation hybride de 2001 à 2012',
                      'Ventilation mécanique sur conduit existant avant 2013',
                      'Ventilation mécanique sur conduit existant à partir de 2013',
                      'Ventilation naturelle par conduit',
                      "Ventilation naturelle par conduit avec entrées d'air hygro",
                      "Ventilation par entrées d'air hautes et basses",
                      'Ventilation par ouverture des fenêtres']}

LABELS = {
    "code_postal_ban": "Code postal",
    "type_batiment": "Type de logement",
    "surface_habitable_logement": "Surface habitable (m²)",
    "periode_construction": "Période de construction",
    "nombre_niveau_logement": "Nombre de niveaux du logement",
    "type_energie_principale_chauffage": "Énergie principale du chauffage",
    "type_generateur_chauffage_principal": "Équipement principal de chauffage",
    "type_installation_chauffage": "Installation de chauffage",
    "type_energie_principale_ecs": "Énergie principale de l’eau chaude",
    "type_installation_ecs": "Installation d’eau chaude",
    "isolation_toiture": "Toiture isolée",
    "qualite_isolation_menuiseries": "Qualité de l’isolation des fenêtres",
    "type_ventilation": "Type de ventilation",
    "surface_climatisee": "Surface climatisée (m²)",
    "logement_traversant": "Logement traversant",
}

# État propre à chaque session. Aucun renseignement n'est écrit sur le disque.
st.session_state.setdefault("validated_housing", None)
st.session_state.setdefault("prediction_result", None)


def predict_logement(features: dict[str, Any]) -> dict[str, Any] | None:
    """POINT DE BRANCHEMENT POUR QUENTIN.

    features contient exactement FEATURES, dans cet ordre.
    Les informations inconnues valent None (à convertir selon le pipeline).
    Charger les pipelines complets avec st.cache_resource, puis leur passer
    une seule ligne. Ne pas réentraîner les modèles dans cette fonction.

    Retour attendu, après branchement :
    {"etiquette_dpe": str, "conso_5_usages_ef": float}
    La consommation doit être exprimée en kWh/an d'énergie finale.
    Tant que les pipelines ne sont pas fournis, aucune valeur n'est inventée.
    """
    return None


def select_category(field: str, required: bool = False) -> Any:
    """Affiche un libellé lisible et conserve la catégorie brute du CSV."""
    return st.selectbox(
        LABELS[field] + (" *" if required else ""),
        options=OPTIONS[field],
        index=None,
        placeholder="Choisir une valeur" if required else "Je ne sais pas / non renseigné",
        key=field,
        persist_state="session",
    )


def select_binary(field: str, help_text: str) -> int | None:
    # Aucun choix = inconnu ; 0 = non ; 1 = oui. Ne pas confondre vide et zéro.
    return st.selectbox(
        LABELS[field],
        options=[None, 0, 1],
        format_func=lambda x: {None: "Je ne sais pas", 0: "Non", 1: "Oui"}[x],
        help=help_text,
        key=field,
        persist_state="session",
    )


def validate_housing(data: dict[str, Any]) -> list[str]:
    errors = []
    for field in ("code_postal_ban", "type_batiment"):
        if data.get(field) not in OPTIONS[field]:
            errors.append(f"Veuillez renseigner : {LABELS[field]}.")
    for field in OPTIONS:
        if data.get(field) is not None and data[field] not in OPTIONS[field]:
            errors.append(f"Valeur non reconnue : {LABELS[field]}.")

    surface = data.get("surface_habitable_logement")
    valid_surface = isinstance(surface, (int, float)) and math.isfinite(surface) and surface > 0
    if not valid_surface:
        errors.append("La surface habitable doit être renseignée et supérieure à zéro.")
    cooled = data.get("surface_climatisee")
    if cooled is not None:
        if not isinstance(cooled, (int, float)) or not math.isfinite(cooled) or cooled < 0:
            errors.append("La surface climatisée doit être positive ou nulle.")
        elif isinstance(surface, (int, float)) and valid_surface and cooled > surface:
            errors.append("La surface climatisée ne peut pas dépasser la surface habitable.")
    levels = data.get("nombre_niveau_logement")
    if levels is not None and (not isinstance(levels, int) or isinstance(levels, bool) or levels < 1):
        errors.append("Le nombre de niveaux doit être un entier supérieur ou égal à 1.")
    for field in ("isolation_toiture", "logement_traversant"):
        if data.get(field) not in (None, 0, 1):
            errors.append(f"Valeur non reconnue : {LABELS[field]}.")
    return errors


def housing_csv(data: dict[str, Any]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=FEATURES)
    writer.writeheader()
    writer.writerow(data)
    return stream.getvalue()


def contexte() -> None:
    st.title("DPE Lyon")
    st.caption("Explorer les logements lyonnais et préparer une estimation énergétique.")
    st.header("Le projet")
    st.write(
        "À partir des caractéristiques d’un logement, l’application vise à estimer "
        "son étiquette DPE de A à G et sa consommation annuelle d’énergie finale."
    )
    st.subheader("Les données")
    a, b, c = st.columns(3)
    a.metric("Diagnostics dans le fichier initial", "10 000")
    b.metric("Diagnostics de logements retenus", "9 921")
    c.metric("Caractéristiques en entrée", "15")
    st.caption(
        "Effectifs du fichier initial fourni par le groupe : les 79 diagnostics "
        "d’immeubles entiers sont exclus. Un diagnostic n’est pas nécessairement "
        "un logement unique ; les réévaluations doivent être traitées dans le nettoyage commun."
    )
    st.subheader("Deux objectifs")
    st.write(
        "Classification : estimer une classe DPE. Régression : estimer la consommation "
        "annuelle d’énergie finale en kWh/an, toutes énergies confondues."
    )
    st.info(
        "Version de développement : le formulaire est opérationnel. "
        "Les modèles et les visualisations de données restent à connecter."
    )
    st.link_button("Consulter les données DPE de l’ADEME", "https://data.ademe.fr/")
    st.caption("Les futures estimations du modèle ne remplacent pas un diagnostic réglementaire.")


def exploration() -> None:
    st.title("Exploration des données")
    st.caption("Emplacements prévus pour les données et visualisations du groupe.")
    st.info("La source de données nettoyée n’est pas encore connectée à cette page.")
    st.subheader("Filtres")
    st.multiselect("Codes postaux", OPTIONS["code_postal_ban"], disabled=True)
    st.multiselect("Types de logement", OPTIONS["type_batiment"], disabled=True)
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
    st.write(
        "Décrivez votre logement. Les champs marqués d’un astérisque sont obligatoires. "
        "Pour les autres, laissez le champ vide si vous ne connaissez pas la réponse."
    )
    data: dict[str, Any] = {}
    with st.form("formulaire_logement", clear_on_submit=False):
        st.subheader("Le logement")
        left, right = st.columns(2)
        with left:
            data["code_postal_ban"] = select_category("code_postal_ban", required=True)
            data["surface_habitable_logement"] = st.number_input(
                LABELS["surface_habitable_logement"] + " *", min_value=0.1,
                value=None, step=1.0, placeholder="Ex. : 75",
                key="surface_habitable_logement", persist_state="session",
            )
            data["nombre_niveau_logement"] = st.number_input(
                LABELS["nombre_niveau_logement"], min_value=1, value=None, step=1,
                placeholder="Ex. : 1",
                help="Nombre de niveaux à l’intérieur du logement : 1 pour un logement sur un seul niveau, 2 pour un duplex. Ce n’est pas le numéro de l’étage.",
                key="nombre_niveau_logement", persist_state="session",
            )
        with right:
            data["type_batiment"] = select_category("type_batiment", required=True)
            data["periode_construction"] = select_category("periode_construction")

        st.divider()
        st.subheader("Le chauffage")
        data["type_energie_principale_chauffage"] = select_category("type_energie_principale_chauffage")
        data["type_generateur_chauffage_principal"] = select_category("type_generateur_chauffage_principal")
        data["type_installation_chauffage"] = select_category("type_installation_chauffage")

        st.divider()
        st.subheader("L’eau chaude sanitaire")
        st.caption("L’eau chaude sanitaire (ECS) est l’eau chaude utilisée aux robinets et à la douche.")
        left, right = st.columns(2)
        with left:
            data["type_energie_principale_ecs"] = select_category("type_energie_principale_ecs")
        with right:
            data["type_installation_ecs"] = select_category("type_installation_ecs")

        st.divider()
        st.subheader("L’isolation, la ventilation et la climatisation")
        left, right = st.columns(2)
        with left:
            data["isolation_toiture"] = select_binary(
                "isolation_toiture", "Indiquez si la toiture est isolée, uniquement si vous disposez de cette information."
            )
            data["type_ventilation"] = select_category("type_ventilation")
            data["logement_traversant"] = select_binary(
                "logement_traversant", "Un logement traversant possède des ouvertures sur deux façades opposées."
            )
        with right:
            data["qualite_isolation_menuiseries"] = select_category("qualite_isolation_menuiseries")
            data["surface_climatisee"] = st.number_input(
                LABELS["surface_climatisee"], min_value=0.0, value=None, step=1.0,
                placeholder="0 si aucune climatisation",
                help="Indiquez 0 si aucune pièce n’est climatisée ; laissez vide si vous ne savez pas.",
                key="surface_climatisee", persist_state="session",
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
            st.session_state["validated_housing"] = payload
            st.session_state["prediction_result"] = None
            try:
                st.session_state["prediction_result"] = predict_logement(payload)
            except Exception:
                st.error("L’estimation n’a pas pu être calculée. Veuillez réessayer ultérieurement.")

    payload = st.session_state["validated_housing"]
    if payload is None:
        st.caption("Complétez le formulaire, puis validez les informations.")
        return

    st.success("Les informations ont bien été prises en compte.")
    st.subheader("Dernières informations validées")
    st.caption("Ce récapitulatif correspond au dernier envoi du formulaire.")
    st.table(
        [{"Caractéristique": LABELS[name], "Valeur": display_value(name, payload[name])} for name in FEATURES],
        alt="Récapitulatif des quinze caractéristiques du logement validées",
    )
    st.subheader("Estimation")
    result = st.session_state["prediction_result"]
    if result is None:
        left, right = st.columns(2)
        left.metric("Étiquette DPE estimée", "—")
        right.metric("Énergie finale estimée (kWh/an)", "—")
        st.info("Les modèles ne sont pas encore connectés. Aucune prédiction n’a été calculée.")
    else:
        left, right = st.columns(2)
        left.metric("Étiquette DPE estimée", str(result["etiquette_dpe"]))
        right.metric("Énergie finale estimée (kWh/an)", f"{float(result['conso_5_usages_ef']):,.0f}")

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


def display_value(field: str, value: Any) -> str:
    if value is None:
        return "Non renseigné"
    if field in ("isolation_toiture", "logement_traversant"):
        return "Oui" if value == 1 else "Non"
    return str(value)


st.sidebar.title("DPE Lyon")
st.sidebar.caption("Projet M2 SISE · Version de développement")
navigation = st.navigation([
    st.Page(contexte, title="Contexte", icon=":material/home:", default=True, url_path="contexte"),
    st.Page(exploration, title="Exploration", icon=":material/bar_chart:", url_path="exploration"),
    st.Page(prediction, title="Prédiction", icon=":material/house:", url_path="prediction"),
])
navigation.run()
