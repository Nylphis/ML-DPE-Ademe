"""Nettoyage des DPE bruts (repris des sections A.1 à A.9 du notebook).

Seules les cellules de contrôle (print, aperçus, appels à values_agg) ont été
retirées. Différence volontaire avec le notebook : l'encodage ordinal de la
section A.10 n'est plus fait ici mais dans le pipeline sklearn (train.py),
pour que l'appli puisse envoyer les libellés bruts au modèle.

Second changement : le dropna() final ne porte plus que sur les cibles. Il
retirait toutes les maisons (certains champs ne sont jamais renseignés pour
elles). Les valeurs manquantes sont désormais encodées comme « inconnues »
dans le pipeline (voir build_preprocessor dans train.py).
"""

import json
import logging

import pandas as pd

from dpe.config import CLEAN_STATS, DATA_CLEAN, DATA_RAW, EXTRA, FEATURES, TARGETS
from dpe.features import famille_generateur

log = logging.getLogger(__name__)


def clean(dfExistant: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"n_brut": len(dfExistant)}

    # 1. Normalisation des noms de colonnes
    dfExistant = dfExistant.copy()
    dfExistant.columns = (
        dfExistant.columns
        .str.replace("﻿", "")
        .str.strip()
        .str.replace(" ", "_")
    )

    # 3. Filtre du périmètre (version et immeubles déjà filtrés par l'API, gardés par sécurité)
    dfFiltre = dfExistant[
        (dfExistant["type_batiment"] != "immeuble")
        & (dfExistant["methode_application_dpe"] != "dpe appartement généré à partir des données DPE immeuble")
        & (dfExistant["version_dpe"] >= 2.1)
    ].copy()
    stats["n_perimetre"] = len(dfFiltre)

    # 4. Sélection des variables
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
        "identifiant_ban",
        "nom_commune_ban",
        "_geopoint",
    ]

    manquantes = [c for c in ls_columns if c not in dfFiltre.columns]
    if manquantes:
        raise ValueError(f"Colonnes absentes du fichier brut : {manquantes}")

    dfClean = dfFiltre[ls_columns].copy()

    # Groupe d'évaluation : tous les DPE d'une même adresse BAN.
    # Un DPE non géolocalisé forme un groupe à lui seul.
    dfClean["groupe"] = dfClean["identifiant_ban"].fillna("dpe_" + dfClean["numero_dpe"].astype(str))

    # Coordonnées pour la carte (_geopoint = "lat,lon")
    coords = dfClean["_geopoint"].str.split(",", n=1, expand=True).reindex(columns=[0, 1])
    dfClean["latitude"] = pd.to_numeric(coords[0], errors="coerce")
    dfClean["longitude"] = pd.to_numeric(coords[1], errors="coerce")

    stats["pct_sans_adresse"] = round(dfClean["identifiant_ban"].isna().mean() * 100, 1)
    dfClean = dfClean.drop(columns=["identifiant_ban", "_geopoint", "numero_dpe"])

    # Climatisation (0/1) à la place de la surface climatisée
    dfClean["climatisation"] = (dfClean["surface_climatisee"].fillna(0) > 0).astype(int)
    dfClean = dfClean.drop(columns=["surface_climatisee"])

    # 6. Générateur : principal, complété par l'installation n1 pour les versions DPE 2.1
    dfClean["generateur"] = dfClean["type_generateur_chauffage_principal"].fillna(
        dfClean["type_generateur_n1_installation_n1"]
    )
    dfClean = dfClean.drop(columns=["type_generateur_chauffage_principal",
                                    "type_generateur_n1_installation_n1"])

    # 7. Regroupement des générateurs en familles
    # (un générateur non renseigné reste NaN, au lieu de tomber dans « Autre »)
    connu = dfClean["generateur"].notna()
    dfClean.loc[connu, "generateur"] = dfClean.loc[connu, "generateur"].apply(famille_generateur)

    # 9. Derniers ajustements
    # Changement par rapport au notebook : plus de dropna() sur les variables
    # explicatives. Il supprimait toutes les maisons, pour lesquelles certains
    # champs ne sont jamais renseignés. Les NaN sont conservés ici et encodés
    # comme « inconnus » dans le pipeline (train.py). Seules les lignes sans
    # cible sont retirées.
    stats["manquants_pct"] = _taux_manquants(dfClean)
    dfClean = dfClean.dropna(subset=TARGETS + ["code_postal_ban", "type_batiment"])
    stats["n_complet"] = int(dfClean[FEATURES].notna().all(axis=1).sum())  # ce que gardait le dropna

    # Fusion des classes A et B (A trop rare)
    dfClean["etiquette_dpe"] = dfClean["etiquette_dpe"].replace({"A": "A-B", "B": "A-B"})

    # Code postal en catégorie (sinon traité comme une grandeur continue)
    dfClean["code_postal_ban"] = dfClean["code_postal_ban"].astype(str)

    # 10. Types (l'encodage ordinal est fait dans le pipeline).
    # Int64 (entier pandas) accepte les valeurs manquantes, contrairement à int.
    for col in ["isolation_toiture", "logement_traversant", "nombre_niveau_logement"]:
        dfClean[col] = dfClean[col].astype("Int64")

    stats["n_clean"] = len(dfClean)
    tailles = dfClean["groupe"].value_counts()
    stats["groupes"] = {"n": int(len(tailles)), "taille_moyenne": round(float(tailles.mean()), 2),
                        "taille_max": int(tailles.max()),
                        "pct_lignes_groupe_2plus": round(float((tailles[tailles > 1].sum()) / len(dfClean) * 100), 1)}
    stats["par_type_batiment"] = {
        "perimetre": dfFiltre["type_batiment"].value_counts().to_dict(),
        "clean": dfClean["type_batiment"].value_counts().to_dict(),
    }
    return dfClean[TARGETS + FEATURES + EXTRA], stats


def _taux_manquants(df: pd.DataFrame) -> dict:
    """% de valeurs manquantes par variable, globalement et par type de bâtiment."""
    cols = [c for c in FEATURES if c in df.columns]
    taux = {"tous": (df[cols].isna().mean() * 100).round(1).to_dict()}
    for type_bat, groupe in df.groupby("type_batiment"):
        taux[type_bat] = (groupe[cols].isna().mean() * 100).round(1).to_dict()
    return taux


def run(input_path=DATA_RAW, output_path=DATA_CLEAN):
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
