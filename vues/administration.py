"""Page Administration : lancer la mise à jour des données et le réentraînement.

Les traitements ne tournent pas ici (512 Mo sur Render) : la page lance le
workflow GitHub Actions .github/workflows/refresh.yml, qui pousse les fichiers
mis à jour sur main ; Render redéploie alors l'appli.

Configuration (variables d'environnement, jamais dans le code) :
- GITHUB_TOKEN  : jeton GitHub limité au dépôt, droit « Actions : lecture et écriture » ;
- GITHUB_DEPOT  : dépôt au format propriétaire/nom (par défaut Nylphis/ML-DPE-Ademe) ;
- GITHUB_BRANCHE : branche sur laquelle lancer le workflow (par défaut main).
"""

import json
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import requests
import streamlit as st

from dpe.config import CLEAN_STATS
from vues.commun import METADATA

# Page ouverte à tous : identifiants volontairement publics et préremplis
IDENTIFIANT, MOT_DE_PASSE = "admin", "admin"

PARIS = ZoneInfo("Europe/Paris")  # GitHub et les fichiers donnent l'heure en UTC

API_GITHUB = "https://api.github.com"
WORKFLOW = "refresh.yml"
DEPOT = os.environ.get("GITHUB_DEPOT", "Nylphis/ML-DPE-Ademe")
BRANCHE = os.environ.get("GITHUB_BRANCHE", "main")
JETON = os.environ.get("GITHUB_TOKEN")
URL_WORKFLOW = f"{API_GITHUB}/repos/{DEPOT}/actions/workflows/{WORKFLOW}"
EN_TETES = {"Accept": "application/vnd.github+json"} | ({"Authorization": f"Bearer {JETON}"} if JETON else {})

# Id de la dernière exécution au moment du lancement : tant que GitHub n'en a pas créé une nouvelle, on attend
CLE_RUN_AVANT = "run_avant_lancement"


@st.cache_data(ttl=10, show_spinner=False)
def dernier_run() -> dict | None:
    """Dernière exécution du workflow (mise en cache 10 s pour ménager l'API GitHub)."""
    reponse = requests.get(f"{URL_WORKFLOW}/runs", headers=EN_TETES,
                           params={"branch": BRANCHE, "per_page": 1}, timeout=15)
    reponse.raise_for_status()
    runs = reponse.json()["workflow_runs"]
    return runs[0] if runs else None


def lancer(complet: bool, reentrainer: bool) -> None:
    """Demande à GitHub de lancer le workflow (réponse 204 si accepté)."""
    reponse = requests.post(
        f"{URL_WORKFLOW}/dispatches", headers=EN_TETES, timeout=15,
        # Valeurs en texte, comme dans la documentation de l'API ; le workflow les lit comme du texte
        json={"ref": BRANCHE, "inputs": {"complet": str(complet).lower(), "reentrainer": str(reentrainer).lower()}},
    )
    reponse.raise_for_status()


def en_cours(run: dict | None) -> bool:
    """Vrai si une exécution vient d'être demandée ou n'est pas terminée."""
    id_run = run["id"] if run else None
    lancement = st.session_state.get(CLE_RUN_AVANT)
    # Lancement demandé, GitHub n'a pas encore créé l'exécution (on abandonne l'attente après 2 minutes)
    if lancement and lancement["id"] == id_run and time.time() - lancement["heure"] < 120:
        return True
    st.session_state.pop(CLE_RUN_AVANT, None)
    return run is not None and run["status"] != "completed"


def date_fr(iso: str | None) -> str:
    """« 2026-11-13T13:49:59+00:00 » devient « 13/11/2026 à 14:49 » (heure de Paris)."""
    if not iso:
        return "inconnue"
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).astimezone(PARIS).strftime("%d/%m/%Y à %H:%M")


@st.fragment(run_every="15s")
def suivi() -> None:
    """Affiche l'exécution en cours et recharge la page quand elle est terminée."""
    dernier_run.clear()
    try:
        run = dernier_run()
    except requests.RequestException:
        st.warning("GitHub est injoignable, nouvel essai dans quelques secondes.")
        return
    if not en_cours(run):
        st.rerun()
    lien = f" [Suivre sur GitHub]({run['html_url']})" if run and run["status"] != "completed" else ""
    st.info("Mise à jour en cours sur GitHub : prévoir quelques minutes de base, une quinzaine avec une "
            "reconstruction complète, plus l’entraînement s’il est demandé. Si des fichiers changent, l’appli "
            "redémarre ensuite (environ 1 minute). Vous pouvez quitter cette page." + lien,
            icon=":material/hourglass_top:")


# ------------------------------------------------
# Page
# ------------------------------------------------
st.title("Administration")
st.write(
    "Mettre à jour les données depuis l’API de l’ADEME et réentraîner les modèles. "
    "Les traitements tournent sur GitHub Actions, puis l’application est redéployée."
)

st.subheader("État actuel")
stats = json.loads(CLEAN_STATS.read_text(encoding="utf-8")) if CLEAN_STATS.exists() else {}
maj = stats.get("derniere_maj_incrementale", {})
a, b = st.columns(2)
a.metric("Diagnostics dans les données", f"{stats.get('n_clean', 0):,}".replace(",", " "))
b.metric("Dernière mise à jour incrémentale", date_fr(maj.get("date")).split(" à ")[0])
st.caption(f"Modèles entraînés le {date_fr(METADATA.get('date_entrainement'))}.")

try:
    run = dernier_run()
except requests.RequestException:
    run = None
    st.warning("Impossible de joindre GitHub : l’état des mises à jour n’est pas disponible.")

if run and not en_cours(run):
    texte = f"Dernière exécution le {date_fr(run['created_at'])} : "
    if run["conclusion"] == "success":
        st.success(texte + f"réussie. [Détails]({run['html_url']})")
    else:
        st.error(texte + f"échec ({run['conclusion']}). [Voir les logs]({run['html_url']})")

st.subheader("Lancer une mise à jour")
if not JETON:
    st.info("Lancement non configuré : la variable d’environnement GITHUB_TOKEN est absente.")
    st.stop()
if en_cours(run):
    suivi()
    st.stop()

with st.form("formulaire_administration"):
    gauche, droite = st.columns(2)
    identifiant = gauche.text_input("Identifiant", value=IDENTIFIANT)
    mot_de_passe = droite.text_input("Mot de passe", value=MOT_DE_PASSE, type="password")
    complet = st.checkbox(
        "Reconstruction complète",
        help="Retélécharge toutes les données (environ 12 minutes) au lieu des seuls DPE modifiés. "
             "Seule une reconstruction complète retire les DPE supprimés par l’ADEME.",
    )
    reentrainer = st.checkbox("Réentraîner les modèles", help="Quelques minutes de plus.")
    envoye = st.form_submit_button("Lancer la mise à jour", type="primary", icon=":material/refresh:")

if envoye:
    if (identifiant, mot_de_passe) != (IDENTIFIANT, MOT_DE_PASSE):
        st.error("Identifiant ou mot de passe incorrect.")
    else:
        try:
            # Dernière exécution relue sans cache juste avant le lancement : c'est elle qu'on attend de voir remplacée
            dernier_run.clear()
            avant = dernier_run()
            lancer(complet, reentrainer)
        except requests.RequestException:
            st.error("GitHub a refusé le lancement : vérifier GITHUB_TOKEN, GITHUB_DEPOT et GITHUB_BRANCHE.")
        else:
            st.session_state[CLE_RUN_AVANT] = {"id": avant["id"] if avant else None, "heure": time.time()}
            dernier_run.clear()
            st.rerun()
