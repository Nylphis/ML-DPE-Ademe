"""Entraînement des deux modèles et export pour l'appli.

Reprend les sections B (classification) et C/E (régression) du notebook.
Changements :
- get_dummies remplacé par un OneHotEncoder dans un Pipeline sklearn, pour
  que le prétraitement soit sauvegardé avec le modèle ;
- découpage train/test par adresse (StratifiedGroupKFold) au lieu d'un tirage
  aléatoire : 96 % des DPE partagent leur adresse avec d'autres, souvent
  quasi identiques, ce qui gonflait les scores ;
- un seul découpage, commun aux deux modèles (même jeu de test) ;
- type de modèle choisi dans config.py (forêt aléatoire ou HistGradientBoosting).
"""

import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, r2_score, recall_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from dpe.config import (BINARY, CATEGORICAL, CLEAN_STATS, CLF_ESTIMATOR,
                        CLF_PARAMS, CLF_PATH, CLIP_QUANTILES, CLIPPED,
                        DATA_CLEAN, FEATURES, GROUP, METADATA_PATH, MODELS_DIR,
                        ORDINAL, ORDRE_ETIQUETTES, RANDOM_STATE, REG_ESTIMATOR,
                        REG_PARAMS, REG_PATH, REQUIRED, TARGET_CLF, TARGET_REG,
                        TEST_SIZE, VALEUR_INCONNUE)
from dpe.features import QuantileClipper, to_model_frame

log = logging.getLogger(__name__)

# Modèles disponibles : nom court (config.py) -> (classification, régression)
ESTIMATORS = {
    "rf": (RandomForestClassifier, RandomForestRegressor),
    "hgb": (HistGradientBoostingClassifier, HistGradientBoostingRegressor),
}


def categories_from(df: pd.DataFrame) -> dict[str, list]:
    """Catégories observées sur tout le jeu nettoyé, comme get_dummies dans le notebook.

    Sans cela, une catégorie rare présente uniquement dans le jeu de test
    (ex. « Bois – Plaquettes d'industrie ») fait échouer la prédiction.
    « inconnu » est toujours ajoutée, même si la variable n'a aucun manquant
    dans les données : l'appli peut alors l'envoyer sans erreur.
    """
    return {c: sorted(df[c].dropna().unique().tolist()) + [VALEUR_INCONNUE]
            for c in CATEGORICAL}


def build_preprocessor(categories: dict[str, list], clip_numeric: bool) -> ColumnTransformer:
    """Prétraitement commun aux deux modèles.

    - categories fixées sur tout le jeu + drop="first" reproduisent
      get_dummies(drop_first=True) appliqué avant le split dans le notebook.
    - handle_unknown="error" : une catégorie jamais vue fait échouer la
      prédiction au lieu d'être silencieusement encodée comme la catégorie
      de référence (le bug « chauffage au bois »).

    Valeurs manquantes (remplace le dropna du notebook) :
    - catégorielles : catégorie « inconnu », encodée comme les autres ;
    - ordinales : -1, en dessous de la première modalité ;
    - binaires : -1 (ni 0 ni 1) ;
    - numériques : médiane du jeu d'entraînement + colonne indicatrice
      « valeur manquante » (seulement si la variable a des manquants à l'entraînement).
    Les forêts aléatoires peuvent isoler ces valeurs dans leurs propres branches.
    """
    categorical = Pipeline([
        ("inconnu", SimpleImputer(strategy="constant", fill_value=VALEUR_INCONNUE)),
        ("onehot", OneHotEncoder(categories=[categories[c] for c in CATEGORICAL],
                                 drop="first", handle_unknown="error", sparse_output=False)),
    ])
    # np.nan en dernière position : autorise les manquants, encodés -1 (encoded_missing_value)
    ordinal = OrdinalEncoder(categories=[ORDINAL[c] + [np.nan] for c in ORDINAL],
                             encoded_missing_value=-1)
    numeric_steps = [("clip", QuantileClipper(*CLIP_QUANTILES))] if clip_numeric else []
    numeric = Pipeline(numeric_steps + [
        ("mediane", SimpleImputer(strategy="median", add_indicator=True)),
    ])
    binary = SimpleImputer(strategy="constant", fill_value=-1)

    return ColumnTransformer([
        ("cat", categorical, CATEGORICAL),
        ("ord", ordinal, list(ORDINAL)),
        ("num", numeric, CLIPPED),
        ("bin", binary, BINARY),
    ])


def make_model(task: str, estimator: str, params: dict, categories: dict[str, list]) -> Pipeline:
    """Pipeline complet (prétraitement + modèle), partagé avec compare_models.py.

    task : "clf" ou "reg" (l'écrêtage des variables numériques ne sert qu'en régression).
    estimator : clé de ESTIMATORS ("rf" ou "hgb").
    """
    classe = ESTIMATORS[estimator][0 if task == "clf" else 1]
    extra = {"n_jobs": -1} if estimator == "rf" else {}  # le HGB est déjà parallélisé
    return Pipeline([
        ("prep", build_preprocessor(categories, clip_numeric=(task == "reg"))),
        ("model", classe(random_state=RANDOM_STATE, **extra, **params)),
    ])


def split_par_adresse(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Découpage train/test sans qu'une adresse soit des deux côtés.

    Un pli de StratifiedGroupKFold : les groupes (adresses) restent entiers et la
    proportion des étiquettes est à peu près conservée. TEST_SIZE = 1/3 donne
    3 plis, donc environ 33 % en test.
    """
    cv = StratifiedGroupKFold(n_splits=round(1 / TEST_SIZE), shuffle=True,
                              random_state=RANDOM_STATE)
    i_train, i_test = next(cv.split(df, df[TARGET_CLF], groups=df[GROUP]))
    return df.iloc[i_train], df.iloc[i_test]


def clip_target(y_train: pd.Series) -> tuple[pd.Series, tuple[float, float]]:
    """Écrête la cible de régression sur le train uniquement (le test garde ses vraies valeurs)."""
    low, high = y_train.quantile(list(CLIP_QUANTILES))
    return y_train.clip(low, high), (float(low), float(high))


def classification_metrics(y_true, y_pred) -> dict:
    rang = {e: i for i, e in enumerate(ORDRE_ETIQUETTES)}
    ecart = np.abs(pd.Series(y_true).map(rang).values - pd.Series(y_pred).map(rang).values)
    # Rappel par classe : part des vrais logements de la classe que le modèle retrouve.
    # labels= fixe l'ordre ; zero_division=0 si une classe n'est jamais prédite.
    rappels = recall_score(y_true, y_pred, labels=ORDRE_ETIQUETTES, average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "accuracy_1_classe": float((ecart <= 1).mean()),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        **{f"recall_{e}": float(r) for e, r in zip(ORDRE_ETIQUETTES, rappels)},
    }


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_classifier(train: pd.DataFrame, test: pd.DataFrame, categories: dict[str, list]):
    model = make_model("clf", CLF_ESTIMATOR, CLF_PARAMS, categories)
    model.fit(train[FEATURES], train[TARGET_CLF])
    metrics = classification_metrics(test[TARGET_CLF], model.predict(test[FEATURES]))
    metrics.update(n_train=len(train), n_test=len(test))
    return model, metrics


def train_regressor(train: pd.DataFrame, test: pd.DataFrame, categories: dict[str, list]):
    y_train, bornes = clip_target(train[TARGET_REG])
    model = make_model("reg", REG_ESTIMATOR, REG_PARAMS, categories)
    model.fit(train[FEATURES], y_train)
    metrics = regression_metrics(test[TARGET_REG], model.predict(test[FEATURES]))
    metrics.update(bornes_cible=list(bornes), n_train=len(train), n_test=len(test))
    return model, metrics


def build_metadata(categories: dict[str, list], clf_metrics: dict, reg_metrics: dict) -> dict:
    """Informations lues par l'appli : contrat d'entrée, options, métriques."""
    # « inconnu » n'est pas proposé dans le formulaire : un champ laissé vide en tient lieu
    options = {c: [v for v in values if v != VALEUR_INCONNUE] for c, values in categories.items()}
    options.update({c: list(levels) for c, levels in ORDINAL.items()})

    stats = json.loads(CLEAN_STATS.read_text(encoding="utf-8")) if CLEAN_STATS.exists() else {}
    return {
        "date_entrainement": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "binary": BINARY,
        "required": REQUIRED,
        "options": options,
        "etiquettes": ORDRE_ETIQUETTES,
        "donnees": stats,
        "modeles": {
            "classification": {"estimateur": CLF_ESTIMATOR, "params": CLF_PARAMS},
            "regression": {"estimateur": REG_ESTIMATOR, "params": REG_PARAMS},
        },
        "evaluation": "jeu de test séparé par adresse BAN (StratifiedGroupKFold)",
        "metriques": {"classification": clf_metrics, "regression": reg_metrics},
    }


def run(input_path=DATA_CLEAN):
    df = pd.read_csv(input_path, dtype={"code_postal_ban": str, GROUP: str})
    # Mêmes types que les saisies de l'appli (colonnes texte en object, NaN inclus)
    df = pd.concat([df[[TARGET_CLF, TARGET_REG, GROUP]], to_model_frame(df)], axis=1)

    categories = categories_from(df)

    train, test = split_par_adresse(df)
    assert not set(train[GROUP]) & set(test[GROUP]), "une adresse est à la fois dans le train et le test"
    repartition = pd.DataFrame({
        "train": train[TARGET_CLF].value_counts(normalize=True),
        "test": test[TARGET_CLF].value_counts(normalize=True),
    }).reindex(ORDRE_ETIQUETTES).mul(100).round(1)
    log.info("Découpage par adresse : %d lignes / %d adresses en train, %d lignes / %d adresses en test",
             len(train), train[GROUP].nunique(), len(test), test[GROUP].nunique())
    log.info("Répartition des étiquettes (%%) :\n%s", repartition.T.to_string())

    clf, clf_metrics = train_classifier(train, test, categories)
    log.info("Classification : accuracy %.3f | ±1 classe %.3f | F1 macro %.3f",
             clf_metrics["accuracy"], clf_metrics["accuracy_1_classe"], clf_metrics["f1_macro"])

    log.info("Rappel par classe : %s", " | ".join(
        f"{e} {clf_metrics[f'recall_{e}']:.2f}" for e in ORDRE_ETIQUETTES))

    reg, reg_metrics = train_regressor(train, test, categories)
    log.info("Régression : RMSE %.1f | MAE %.1f | R² %.3f",
             reg_metrics["rmse"], reg_metrics["mae"], reg_metrics["r2"])

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, CLF_PATH, compress=3)
    joblib.dump(reg, REG_PATH, compress=3)

    metadata = build_metadata(categories, clf_metrics, reg_metrics)
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    for path in (CLF_PATH, REG_PATH):
        log.info("Modèle sauvegardé : %s (%.1f Mo)", path.name, path.stat().st_size / 1e6)
    return clf, reg, metadata
