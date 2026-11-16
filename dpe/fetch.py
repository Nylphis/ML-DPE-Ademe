"""Récupération des DPE via l'API ADEME (data-fair).

Deux modes :
- complet (fetch) : tous les DPE du périmètre ; pas de nouvel appel si le
  fichier brut correspond déjà à la même requête (cache, FETCH_INFO) ;
- incrémental (fetch_nouveaux) : DPE créés ou modifiés depuis la date de
  modification la plus récente du fichier nettoyé.

Écriture page par page dans un fichier .part renommé à la fin : un
téléchargement interrompu n'écrase pas le fichier précédent. Nouvelles
tentatives automatiques en cas d'échec d'une page.
"""

import json
import logging
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from dpe.config import DATA_CLEAN, DATA_NOUVEAUX, DATA_RAW, DATE_MAJ, FETCH_INFO

log = logging.getLogger(__name__)

URL_BASE = "https://data.ademe.fr/data-fair/api/v1/datasets/meg-83tjwtg8dyz4vv7h1dqe/lines"
URL_DATASET = "https://data.ademe.fr/data-fair/api/v1/datasets/meg-83tjwtg8dyz4vv7h1dqe"

COLONNES_API = [
    "numero_dpe",
    "date_etablissement_dpe",
    "date_derniere_modification_dpe",
    "numero_dpe_remplace",
    "version_dpe",
    "methode_application_dpe",
    "numero_dpe_immeuble_associe",
    "etiquette_dpe",
    "conso_5_usages_ef",
    "code_postal_ban",
    "type_batiment",
    "surface_habitable_logement",
    "periode_construction",
    "nombre_niveau_logement",
    "type_energie_principale_chauffage",
    "type_generateur_chauffage_principal",
    "type_installation_chauffage",
    "type_energie_principale_ecs",
    "type_installation_ecs",
    "isolation_toiture",
    "qualite_isolation_menuiseries",
    "surface_climatisee",
    "logement_traversant",
    "type_generateur_n1_installation_n1",
    # Localisation : découpage train/test par adresse, carte, filtres
    "identifiant_ban",
    "nom_commune_ban",
    "_geopoint",
]

CODES_POSTAUX = [
    "69001", "69002", "69003", "69004", "69005", "69006", "69007", "69008", "69009",
    "69100", "69500", "69300", "69130", "69150", "69600", "69140", "69700",
]


PAGE_SIZE = 10000
TENTATIVES = 5


def _requete(max_lignes, depuis=None):
    """Paramètres de la requête, aussi utilisés comme clé de cache.

    Parameters
    ----------
    max_lignes : int or None
        Taille de l'échantillon ; None pour tout récupérer.
    depuis : str, optional
        Date AAAA-MM-JJ : seulement les DPE modifiés ce jour-là ou après.

    Returns
    -------
    dict
        Clés "qs", "select" et "max_lignes".
    """
    qs = (
        f"code_postal_ban:({' OR '.join(CODES_POSTAUX)})"
        " AND version_dpe:[2.1 TO *]"
        " AND NOT type_batiment:immeuble"
    )
    if depuis:
        qs += f" AND {DATE_MAJ}:[{depuis} TO *]"
    return {
        "qs": qs,
        "select": ",".join(COLONNES_API),
        "max_lignes": max_lignes,
    }


def deja_telecharge(output=DATA_RAW, max_lignes=None) -> dict | None:
    """Trace du téléchargement si le fichier brut correspond à la requête.

    Returns
    -------
    dict or None
        Contenu de FETCH_INFO, ou None si fichier absent, requête différente
        ou téléchargement inachevé.
    """
    if not (output.exists() and FETCH_INFO.exists()):
        return None
    info = json.loads(FETCH_INFO.read_text(encoding="utf-8"))
    if info.get("requete") != _requete(max_lignes) or not info.get("termine"):
        return None
    return info


def _get(url, params):
    """GET JSON avec nouvelles tentatives (attente 2, 4, 8, 16 s).

    Raises
    ------
    requests.RequestException
        Après TENTATIVES échecs.
    """
    for tentative in range(1, TENTATIVES + 1):
        try:
            response = requests.get(url, params=params, timeout=120)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            if tentative == TENTATIVES:
                raise
            attente = 2 ** tentative
            log.warning("Échec de la requête (%s), nouvel essai dans %d s", exc, attente)
            time.sleep(attente)


def _telecharger(requete, output, max_lignes=None):
    """Écrit dans `output` toutes les pages de la requête.

    Returns
    -------
    n_lignes : int
        Lignes écrites.
    total : int or None
        Total annoncé par l'API.
    """
    # Colonnes demandées absentes de l'API : avertissement seulement
    schema = _get(URL_DATASET, None)["schema"]
    cles_api = {champ["key"] for champ in schema}
    absentes = [c for c in COLONNES_API if c not in cles_api]
    if absentes:
        log.warning("Colonnes absentes de l'API : %s", absentes)

    output.parent.mkdir(parents=True, exist_ok=True)
    partiel = output.with_suffix(output.suffix + ".part")
    url = URL_BASE
    params = {"qs": requete["qs"], "select": requete["select"], "size": PAGE_SIZE}
    n_lignes, total, debut = 0, None, time.time()

    with open(partiel, "w", encoding="utf-8", newline="") as f:
        while url and (max_lignes is None or n_lignes < max_lignes):
            data = _get(url, params)
            total = data.get("total", total)  # total présent sur la première page seulement
            lignes = data["results"]
            if max_lignes is not None:
                lignes = lignes[: max_lignes - n_lignes]
            if not lignes:
                break
            # Même ordre de colonnes sur toutes les pages
            pd.DataFrame(lignes, columns=COLONNES_API).to_csv(f, index=False, header=(n_lignes == 0))
            n_lignes += len(lignes)
            log.info("Lignes récupérées : %d / %s (%.0f s)", n_lignes, total, time.time() - debut)

            url = data.get("next")  # pagination par curseur : paramètres inclus dans next
            params = None
            time.sleep(0.2)  # ménage l'API

    partiel.replace(output)
    return n_lignes, total


def fetch(output=DATA_RAW, refetch=False, max_lignes=None):
    """Téléchargement complet des DPE du périmètre.

    Parameters
    ----------
    output : Path, default=DATA_RAW
        Fichier CSV brut.
    refetch : bool, default=False
        Retélécharger même si le cache correspond.
    max_lignes : int, optional
        Échantillon (ex. 10000) ; None pour tout récupérer.

    Returns
    -------
    int
        Nombre de lignes du fichier brut.
    """
    requete = _requete(max_lignes)
    info = deja_telecharge(output, max_lignes)
    if info and not refetch:
        log.info("Données brutes déjà présentes (%d lignes, téléchargées le %s) : "
                 "pas de nouvel appel à l'API. Utiliser --refetch pour forcer.",
                 info["n_lignes"], info["date"])
        return info["n_lignes"]

    n_lignes, total = _telecharger(requete, output, max_lignes)
    FETCH_INFO.write_text(json.dumps({
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_lignes": n_lignes,
        "total_api": total,
        "termine": True,
        "requete": requete,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Données brutes sauvegardées : %s (%d lignes)", output, n_lignes)
    return n_lignes


def date_derniere_maj(clean_path=DATA_CLEAN) -> str:
    """Date de modification la plus récente du fichier nettoyé.

    Returns
    -------
    str
        Date AAAA-MM-JJ.

    Raises
    ------
    ValueError
        Colonne DATE_MAJ absente : reconstruction complète nécessaire.
    """
    colonnes = pd.read_csv(clean_path, nrows=0).columns
    if DATE_MAJ not in colonnes:
        raise ValueError(f"{clean_path.name} n'a pas de colonne {DATE_MAJ} : faire d'abord "
                         "une reconstruction complète (python run.py fetch clean --refetch).")
    return pd.read_csv(clean_path, usecols=[DATE_MAJ])[DATE_MAJ].max()


def fetch_nouveaux(output=DATA_NOUVEAUX, clean_path=DATA_CLEAN):
    """Téléchargement incrémental : DPE créés ou modifiés depuis la dernière mise à jour.

    Date de départ incluse : les DPE de ce jour-là reviennent à chaque passage,
    dédoublonnés ensuite par clean.run_incremental.

    Returns
    -------
    int
        Nombre de lignes téléchargées (0 : fichier vide).
    """
    depuis = date_derniere_maj(clean_path)
    log.info("Mise à jour incrémentale : DPE modifiés depuis le %s", depuis)
    n_lignes, _ = _telecharger(_requete(None, depuis), output)
    log.info("Nouvelles lignes : %s (%d lignes)", output, n_lignes)
    return n_lignes
