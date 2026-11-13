"""Chemins, variables et hyperparamètres communs au pipeline et à l'appli.

Toute modification du contrat d'entrée du modèle (FEATURES) se fait ici.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------
# Chemins
# ------------------------------------------------
DATA_RAW = ROOT / "data" / "raw" / "dpe03existant.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "dpe_clean.csv"
CLEAN_STATS = ROOT / "data" / "processed" / "clean_stats.json"
# Trace du dernier téléchargement : évite de rappeler l'API inutilement
FETCH_INFO = ROOT / "data" / "raw" / "fetch_info.json"
# Mise à jour incrémentale : DPE créés ou modifiés depuis la dernière mise à jour
DATA_NOUVEAUX = ROOT / "data" / "raw" / "dpe_nouveaux.csv"

MODELS_DIR = ROOT / "models"
CLF_PATH = MODELS_DIR / "classification_etiquette.joblib"
REG_PATH = MODELS_DIR / "regression_conso.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

# ------------------------------------------------
# Cibles
# ------------------------------------------------
TARGET_CLF = "etiquette_dpe"
TARGET_REG = "conso_5_usages_ef"
TARGETS = [TARGET_CLF, TARGET_REG]

# Classes A et B fusionnées dans le notebook (A trop rare)
ORDRE_ETIQUETTES = ["A-B", "C", "D", "E", "F", "G"]

# ------------------------------------------------
# Variables d'entrée du modèle (contrat avec l'appli)
# ------------------------------------------------
CATEGORICAL = [
    "code_postal_ban",
    "type_batiment",
    "type_energie_principale_chauffage",
    "type_installation_chauffage",
    "type_energie_principale_ecs",
    "type_installation_ecs",
    "generateur",
]

# Variables ordinales : l'ordre des listes fixe l'encodage 0, 1, 2…
ORDINAL = {
    "periode_construction": [
        "avant 1948", "1948-1974", "1975-1977", "1978-1982", "1983-1988",
        "1989-2000", "2001-2005", "2006-2012", "2013-2021", "après 2021",
    ],
    "qualite_isolation_menuiseries": ["insuffisante", "moyenne", "bonne", "très bonne"],
}

# Variables numériques écrêtées aux quantiles 1 % / 99 % (régression uniquement)
CLIPPED = ["surface_habitable_logement", "nombre_niveau_logement"]

# Variables binaires 0/1
BINARY = ["isolation_toiture", "logement_traversant", "climatisation"]

# Valeur donnée aux catégories non renseignées (remplace le dropna du notebook)
VALEUR_INCONNUE = "inconnu"

# Champs que l'utilisateur doit obligatoirement renseigner dans l'appli.
# Les autres peuvent rester vides : le pipeline les traite comme « inconnus ».
REQUIRED = ["code_postal_ban", "type_batiment", "surface_habitable_logement"]

# Ordre des colonnes identique à celui du notebook
FEATURES = [
    "code_postal_ban",
    "type_batiment",
    "surface_habitable_logement",
    "periode_construction",
    "nombre_niveau_logement",
    "type_energie_principale_chauffage",
    "type_installation_chauffage",
    "type_energie_principale_ecs",
    "type_installation_ecs",
    "isolation_toiture",
    "qualite_isolation_menuiseries",
    "logement_traversant",
    "climatisation",
    "generateur",
]

# Colonnes gardées dans le CSV nettoyé, qui ne sont pas des variables du modèle
GROUP = "groupe"   # adresse BAN : sert à découper train/test sans séparer les logements d'une même adresse
# Identifiant et date de dernière modification : servent à la mise à jour incrémentale
ID_DPE = "numero_dpe"
DATE_MAJ = "date_derniere_modification_dpe"
EXTRA = [GROUP, "latitude", "longitude", "nom_commune_ban", ID_DPE, DATE_MAJ]

# ------------------------------------------------
# Entraînement (valeurs reprises du notebook)
# ------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 1 / 3
CLIP_QUANTILES = (0.01, 0.99)

# Type de modèle : "rf" (forêt aléatoire) ou "hgb" (HistGradientBoosting)
CLF_ESTIMATOR = "hgb"
REG_ESTIMATOR = "hgb"

# Choisis avec compare_models.py : validation croisée par adresse BAN, 50 000 lignes, 3 plis.
# Classification : meilleur F1 macro, compromis entre accuracy et rappel des classes rares (A-B, G).
CLF_PARAMS = {
    "learning_rate": 0.05,
    "max_leaf_nodes": 255,
    "max_iter": 500,
    "early_stopping": True,
    "class_weight": "balanced",
}
# Régression : meilleur sur RMSE, MAE et R², 0,3 Mo
REG_PARAMS = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "max_iter": 500,
    "early_stopping": True,
}