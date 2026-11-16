"""Nettoyage des DPE bruts vers data/processed/dpe_clean.csv.

Deux modes :
- complet (run) : nettoyage de tout le fichier brut ;
- incrémental (run_incremental) : nettoyage des DPE créés ou modifiés, puis
  fusion dans le fichier existant.

Libellés gardés bruts : l'encodage est fait dans les pipelines (train.py), ce
qui permet à l'appli et à l'API d'envoyer les saisies telles quelles. Valeurs
manquantes conservées (encodées « inconnues » par les pipelines) : seules les
lignes sans cible, code postal ou type de logement sont retirées. Un dropna
complet retirerait toutes les maisons, dont certains champs ne sont jamais renseignés.
"""

import json
import logging
from datetime import datetime, timezone

import pandas as pd

from dpe.config import (CLEAN_STATS, DATA_CLEAN, DATA_NOUVEAUX, DATA_RAW, EXTRA, FEATURES,
                        GROUP, ID_DPE, TARGETS)
from dpe.features import famille_generateur

log = logging.getLogger(__name__)


def clean(dfExistant: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Nettoie un tableau de DPE bruts.

    Parameters
    ----------
    dfExistant : DataFrame
        DPE bruts (colonnes COLONNES_API de fetch.py).

    Returns
    -------
    df : DataFrame
        Colonnes TARGETS + FEATURES + EXTRA.
    stats : dict
        Effectifs, taux de manquants, groupes d'adresses.

    Raises
    ------
    ValueError
        Colonne attendue absente du fichier brut.
    """
    stats = {"n_brut": len(dfExistant)}

    # Noms de colonnes : BOM, espaces
    dfExistant = dfExistant.copy()
    dfExistant.columns = (
        dfExistant.columns
        .str.replace("﻿", "")
        .str.strip()
        .str.replace(" ", "_")
    )

    # Périmètre (version et immeubles déjà filtrés par l'API : contrôle redondant)
    dfFiltre = dfExistant[
        (dfExistant["type_batiment"] != "immeuble")
        & (dfExistant["methode_application_dpe"] != "dpe appartement généré à partir des données DPE immeuble")
        & (dfExistant["version_dpe"] >= 2.1)
    ].copy()
    stats["n_perimetre"] = len(dfFiltre)

    ls_columns = [
        # cibles
        "etiquette_dpe",
        "conso_5_usages_ef",
        # variables du formulaire
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
        # hors modèle
        "numero_dpe",
        "date_derniere_modification_dpe",
        "identifiant_ban",
        "nom_commune_ban",
        "_geopoint",
    ]

    manquantes = [c for c in ls_columns if c not in dfFiltre.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes du fichier brut : {manquantes}")

    dfClean = dfFiltre[ls_columns].copy()

    # Groupe = adresse BAN ; DPE sans adresse : groupe à lui seul
    dfClean["groupe"] = dfClean["identifiant_ban"].fillna("dpe_" + dfClean["numero_dpe"].astype(str))

    # Coordonnées pour la carte : _geopoint = "lat,lon"
    coords = dfClean["_geopoint"].str.split(",", n=1, expand=True).reindex(columns=[0, 1])
    dfClean["latitude"] = pd.to_numeric(coords[0], errors="coerce")
    dfClean["longitude"] = pd.to_numeric(coords[1], errors="coerce")

    stats["pct_sans_adresse"] = round(dfClean["identifiant_ban"].isna().mean() * 100, 1)
    dfClean = dfClean.drop(columns=["identifiant_ban", "_geopoint"])

    # Surface climatisée -> climatisation 0/1
    dfClean["climatisation"] = (dfClean["surface_climatisee"].fillna(0) > 0).astype(int)
    dfClean = dfClean.drop(columns=["surface_climatisee"])

    # Générateur principal, complété par celui de l'installation n1 (DPE 2.1)
    dfClean["generateur"] = dfClean["type_generateur_chauffage_principal"].fillna(
        dfClean["type_generateur_n1_installation_n1"]
    )
    dfClean = dfClean.drop(columns=["type_generateur_chauffage_principal",
                                    "type_generateur_n1_installation_n1"])

    # Familles de générateurs ; non renseigné : reste NaN (distinct de « Autre »)
    connu = dfClean["generateur"].notna()
    dfClean.loc[connu, "generateur"] = dfClean.loc[connu, "generateur"].apply(famille_generateur)

    # Manquants conservés (« inconnus » dans les pipelines) ; lignes sans cible retirées
    stats["manquants_pct"] = _taux_manquants(dfClean)
    dfClean = dfClean.dropna(subset=TARGETS + ["code_postal_ban", "type_batiment"])

    # A et B regroupées (A trop rare)
    dfClean["etiquette_dpe"] = dfClean["etiquette_dpe"].replace({"A": "A-B", "B": "A-B"})

    dfClean = _types(dfClean)
    stats["par_type_batiment"] = {"perimetre": dfFiltre["type_batiment"].value_counts().to_dict()}
    _stats_clean(dfClean, stats)
    return dfClean[TARGETS + FEATURES + EXTRA], stats


def _types(df: pd.DataFrame) -> pd.DataFrame:
    """Fixe les types des colonnes (modifie `df` en place et le renvoie)."""
    # Texte : catégorie, pas grandeur continue
    df["code_postal_ban"] = df["code_postal_ban"].astype(str)
    # Int64 : entier acceptant les manquants
    for col in ["isolation_toiture", "logement_traversant", "nombre_niveau_logement"]:
        df[col] = df[col].astype("Int64")
    return df


def _stats_clean(df: pd.DataFrame, stats: dict) -> None:
    """Ajoute à `stats` les statistiques du fichier nettoyé (effectifs, groupes d'adresses)."""
    stats["n_complet"] = int(df[FEATURES].notna().all(axis=1).sum())  # lignes sans aucun manquant
    stats["n_clean"] = len(df)
    tailles = df[GROUP].value_counts()
    stats["groupes"] = {"n": int(len(tailles)), "taille_moyenne": round(float(tailles.mean()), 2),
                        "taille_max": int(tailles.max()),
                        "pct_lignes_groupe_2plus": round(float((tailles[tailles > 1].sum()) / len(df) * 100), 1)}
    stats["par_type_batiment"]["clean"] = df["type_batiment"].value_counts().to_dict()


def _taux_manquants(df: pd.DataFrame) -> dict:
    """Pourcentage de manquants par variable, global et par type de bâtiment."""
    cols = [c for c in FEATURES if c in df.columns]
    taux = {"tous": (df[cols].isna().mean() * 100).round(1).to_dict()}
    for type_bat, groupe in df.groupby("type_batiment"):
        taux[type_bat] = (groupe[cols].isna().mean() * 100).round(1).to_dict()
    return taux


def run(input_path=DATA_RAW, output_path=DATA_CLEAN):
    """Nettoyage complet : écrit le CSV nettoyé et CLEAN_STATS.

    Returns
    -------
    DataFrame
        DPE nettoyés.
    """
    df_brut = pd.read_csv(input_path)
    df_clean, stats = clean(df_brut)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    CLEAN_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    log.info("Nettoyage : %d brutes -> %d dans le périmètre -> %d conservées "
             "(%d sans aucune valeur manquante)",
             stats["n_brut"], stats["n_perimetre"], stats["n_clean"], stats["n_complet"])
    log.info("Par type de bâtiment (périmètre -> conservées) : %s",
             {t: f"{n} -> {stats['par_type_batiment']['clean'].get(t, 0)}"
              for t, n in stats["par_type_batiment"]["perimetre"].items()})
    log.info("Groupes (adresses) : %s | DPE sans adresse : %s %%",
            stats["groupes"], stats["pct_sans_adresse"])
    for groupe, taux in stats["manquants_pct"].items():
        non_nuls = {c: v for c, v in taux.items() if v > 0}
        log.info("Valeurs manquantes (%%) – %s : %s", groupe, non_nuls or "aucune")
    return df_clean


def _meme_contenu(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    """Vrai si les deux tables ont les mêmes lignes une fois écrites en CSV (ordre ignoré)."""
    def texte(df):
        return df.sort_values(ID_DPE).to_csv(index=False)
    return len(a) == len(b) and texte(a) == texte(b)


def run_incremental(input_path=DATA_NOUVEAUX, output_path=DATA_CLEAN):
    """Nettoie les DPE créés ou modifiés et les fusionne dans le fichier nettoyé.

    Returns
    -------
    DataFrame or None
        Fichier fusionné ; None si rien de nouveau (fichier non réécrit).

    Notes
    -----
    - DPE retéléchargé : remplace sa version précédente (même numero_dpe), ou
      la retire s'il ne passe plus le nettoyage.
    - DPE remplacé (numero_dpe_remplace) : retiré, comme dans le jeu ADEME.
    - DPE supprimé par l'ADEME : non détecté ; retiré seulement par une
      reconstruction complète (python run.py fetch clean --refetch).
    - Statistiques des données brutes (n_brut, taux de manquants…) : celles de
      la dernière reconstruction complète. Statistiques du fichier nettoyé :
      recalculées.
    """
    if input_path.stat().st_size == 0:
        log.info("Aucun DPE nouveau ou modifié : fichier nettoyé inchangé.")
        return None
    df_brut = pd.read_csv(input_path, dtype={ID_DPE: str, "numero_dpe_remplace": str})
    df_nouveaux, _ = clean(df_brut)
    a_remplacer = set(df_brut[ID_DPE]) | set(df_brut["numero_dpe_remplace"].dropna())

    df_existant = pd.read_csv(output_path, dtype={"code_postal_ban": str, GROUP: str, ID_DPE: str})
    a_retirer = df_existant[ID_DPE].isin(a_remplacer)
    # DPE du jour de départ retéléchargés à chaque passage : si inchangés,
    # rien n'est réécrit (pas de commit inutile du workflow)
    if _meme_contenu(_types(df_existant[a_retirer].copy()), df_nouveaux):
        log.info("%d DPE retéléchargés, tous identiques : fichier nettoyé inchangé.", len(df_brut))
        return None
    df_clean = _types(pd.concat([df_existant[~a_retirer], df_nouveaux], ignore_index=True))
    df_clean.to_csv(output_path, index=False)

    stats = json.loads(CLEAN_STATS.read_text(encoding="utf-8"))
    _stats_clean(df_clean, stats)
    stats["derniere_maj_incrementale"] = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_telecharges": len(df_brut),
        "n_nettoyes": len(df_nouveaux),
        "n_existants_remplaces_ou_retires": int(a_retirer.sum()),
    }
    CLEAN_STATS.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    log.info("Mise à jour incrémentale : %d téléchargées -> %d nettoyées ; "
             "%d lignes existantes remplacées ou retirées ; %d lignes au total",
             len(df_brut), len(df_nouveaux), int(a_retirer.sum()), len(df_clean))
    return df_clean
