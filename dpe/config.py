"""Configuration commune au pipeline, à l'appli et à l'API.

Chemins, cibles, variables d'entrée des modèles et hyperparamètres.
Le contrat d'entrée des modèles (FEATURES) se modifie uniquement ici ;
tout changement impose un réentraînement.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ------------------------------------------------
# Chemins
# ------------------------------------------------
DATA_RAW = ROOT / "data" / "raw" / "dpe03existant.csv"
DATA_CLEAN = ROOT / "data" / "processed" / "dpe_clean.csv"
CLEAN_STATS = ROOT / "data" / "processed" / "clean_stats.json"
FETCH_INFO = ROOT / "data" / "raw" / "fetch_info.json"  # trace du dernier téléchargement complet (cache)
DATA_NOUVEAUX = ROOT / "data" / "raw" / "dpe_nouveaux.csv"  # DPE créés ou modifiés (mise à jour incrémentale)

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

# A et B regroupées : classe A trop rare
ORDRE_ETIQUETTES = ["A-B", "C", "D", "E", "F", "G"]

# ------------------------------------------------
# Variables d'entrée des modèles
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

# Ordre des modalités = ordre de l'encodage (0, 1, 2…)
ORDINAL = {
    "periode_construction": [
        "avant 1948", "1948-1974", "1975-1977", "1978-1982", "1983-1988",
        "1989-2000", "2001-2005", "2006-2012", "2013-2021", "après 2021",
    ],
    "qualite_isolation_menuiseries": ["insuffisante", "moyenne", "bonne", "très bonne"],
}

# Écrêtées aux quantiles CLIP_QUANTILES (régression uniquement)
CLIPPED = ["surface_habitable_logement", "nombre_niveau_logement"]

# Variables 0/1
BINARY = ["isolation_toiture", "logement_traversant", "climatisation"]

# Modalité des catégories non renseignées
VALEUR_INCONNUE = "inconnu"

# Champs obligatoires (appli et API) ; les autres, vides, valent « inconnu »
REQUIRED = ["code_postal_ban", "type_batiment", "surface_habitable_logement"]

# Jamais renseignés pour les maisons dans les données ADEME : ignorés pour une maison
CHAMPS_APPARTEMENT = ["type_installation_chauffage", "type_installation_ecs"]

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

# Colonnes du CSV nettoyé hors variables des modèles
GROUP = "groupe"  # adresse BAN : découpage train/test sans séparer une même adresse
ID_DPE = "numero_dpe"  # identifiant (dédoublonnage incrémental)
DATE_MAJ = "date_derniere_modification_dpe"  # point de départ de la mise à jour incrémentale
EXTRA = [GROUP, "latitude", "longitude", "nom_commune_ban", ID_DPE, DATE_MAJ]

# ------------------------------------------------
# Entraînement
# ------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 1 / 3
CLIP_QUANTILES = (0.01, 0.99)

# Modèle : "rf" (forêt aléatoire) ou "hgb" (HistGradientBoosting)
CLF_ESTIMATOR = "hgb"
REG_ESTIMATOR = "hgb"

# Hyperparamètres choisis avec compare_models.py (validation croisée par adresse, 3 plis).
# Classification : meilleur F1 macro, compromis accuracy / rappel des classes rares.
CLF_PARAMS = {
    "learning_rate": 0.05,
    "max_leaf_nodes": 255,
    "max_iter": 500,
    "early_stopping": True,
    "class_weight": "balanced",
}
# Régression : meilleurs RMSE, MAE et R²
REG_PARAMS = {
    "learning_rate": 0.1,
    "max_leaf_nodes": 63,
    "max_iter": 500,
    "early_stopping": True,
}
