"""Transformations réutilisées par le nettoyage et par les pipelines sauvegardés.

Ce module doit rester importable par l'appli : les pipelines .joblib
référencent QuantileClipper, et joblib a besoin de retrouver la classe au chargement.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from dpe.config import CATEGORICAL, FEATURES, ORDINAL


def to_model_frame(rows) -> pd.DataFrame:
    """Met des saisies (dict, liste de dict ou DataFrame) au format attendu par les pipelines.

    Les champs absents ou à None deviennent NaN, que le pipeline traite comme
    « inconnu ». Les colonnes texte sont forcées en object : sans cela, une
    colonne entièrement vide serait typée float et l'imputation échouerait.
    """
    if isinstance(rows, pd.DataFrame):
        X = rows.reindex(columns=FEATURES).copy()
    else:
        X = pd.DataFrame([rows] if isinstance(rows, dict) else rows, columns=FEATURES)
    texte = CATEGORICAL + list(ORDINAL)
    X[texte] = X[texte].astype(object).where(X[texte].notna(), np.nan)
    autres = [c for c in FEATURES if c not in texte]
    X[autres] = X[autres].apply(pd.to_numeric).astype(float)  # float comme à l’entraînement
    return X


def famille_generateur(libelle):
    """Regroupe les libellés de générateurs en familles (code repris du notebook).

    L'ordre des tests compte : "PAC hybride" doit tomber dans PAC,
    "chaudière électrique" avant le cas électrique générique.
    """
    s = str(libelle).lower()

    if "pac " in s or s.startswith("pac") or "pompe à chaleur" in s:
        return "PAC"
    if "réseau de chaleur" in s:
        return "Réseau de chaleur"
    if "chaudière" in s and "électrique" in s:
        return "Électrique direct"
    if "chaudière" in s and ("gaz" in s or "gpl" in s) and "condensation" in s:
        return "Chaudière gaz condensation"
    if ("chaudière" in s and ("gaz" in s or "gpl" in s)) or "combustion gaz" in s:
        return "Chaudière gaz non condensation"
    if "chaudière" in s and "fioul" in s:
        return "Chaudière fioul"
    if ("bois" in s or "granulés" in s or "bûche" in s or "plaquette" in s
            or "foyer fermé" in s or "insert" in s or "cuisinière" in s):
        return "Bois"
    if "accumulation" in s:
        return "Électrique accumulation"
    if "électrique" in s or "effet joule" in s or "convecteur" in s:
        return "Électrique direct"
    if "poêle" in s or "radiateur à gaz" in s:
        return "Poêle / radiateur autonome"
    return "Autre"


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Écrête chaque colonne entre ses quantiles calculés sur le jeu d'entraînement.

    Reproduit le clipping du notebook (section C.4), mais à l'intérieur du
    pipeline : les bornes apprises sont sauvegardées avec le modèle et
    réappliquées à la saisie de l'utilisateur.
    """

    def __init__(self, low=0.01, high=0.99):
        self.low = low
        self.high = high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.low, axis=0)
        self.upper_ = np.nanquantile(X, self.high, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)
