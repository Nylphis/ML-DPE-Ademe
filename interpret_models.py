"""Interprétation des modèles enregistrés, pour le rapport de ML.

Jeu de test reconstitué à l'identique (même découpage par adresse que
dpe/train.py). Sorties dans reports/ :
- matrice_confusion.png et .csv : en % de chaque vraie classe (diagonale = rappel) ;
- importance_variables.png et .csv : importance par permutation des 14
  variables, pour les deux modèles.

Permutation plutôt que feature_importances_ (absent du HistGradientBoosting) :
importance mesurée sur le jeu de test, par variable d'origine et non par
colonne encodée.

Prérequis : pip install matplotlib==3.11.2 (absent de requirements.txt, pour ne
pas alourdir l'image Docker de l'appli, qui n'en a pas besoin).
Lancement : python interpret_models.py   (environ 2 minutes)
"""

import json
import logging

import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from dpe.config import (CLF_PATH, DATA_CLEAN, FEATURES, GROUP, METADATA_PATH, ORDRE_ETIQUETTES,
                        RANDOM_STATE, REG_PATH, ROOT, TARGET_CLF, TARGET_REG)
from dpe.features import to_model_frame
from dpe.train import classification_metrics, regression_metrics, split_par_adresse

matplotlib.use("Agg")  # fichiers images, sans fenêtre
log = logging.getLogger(__name__)

REPORTS = ROOT / "reports"
N_REPETITIONS = 5  # mélanges par variable (moyenne et écart-type)

NOMS = {
    "code_postal_ban": "Code postal",
    "type_batiment": "Type de logement",
    "surface_habitable_logement": "Surface habitable",
    "periode_construction": "Période de construction",
    "nombre_niveau_logement": "Nombre de niveaux",
    "type_energie_principale_chauffage": "Énergie du chauffage",
    "generateur": "Type de chauffage",
    "type_installation_chauffage": "Installation de chauffage",
    "type_energie_principale_ecs": "Énergie de l'eau chaude",
    "type_installation_ecs": "Installation d'eau chaude",
    "isolation_toiture": "Toiture isolée",
    "qualite_isolation_menuiseries": "Isolation des fenêtres",
    "logement_traversant": "Logement traversant",
    "climatisation": "Climatisation",
}


def jeu_de_test() -> pd.DataFrame:
    """Jeu de test de l'entraînement (même préparation et même découpage que train.run)."""
    df = pd.read_csv(DATA_CLEAN, dtype={"code_postal_ban": str, GROUP: str})
    df = pd.concat([df[[TARGET_CLF, TARGET_REG, GROUP]], to_model_frame(df)], axis=1)
    _, test = split_par_adresse(df)
    return test


def verifier_scores(clf_metrics: dict, reg_metrics: dict) -> None:
    """Compare l'accuracy et la MAE recalculées à metadata.json.

    Écart : modèles entraînés sur d'autres données (avertissement dans le log).
    """
    attendus = json.loads(METADATA_PATH.read_text(encoding="utf-8"))["metriques"]
    for nom, obtenu, attendu in [("accuracy", clf_metrics["accuracy"], attendus["classification"]["accuracy"]),
                                 ("MAE", reg_metrics["mae"], attendus["regression"]["mae"])]:
        if abs(obtenu - attendu) > 1e-6:
            log.warning("%s recalculée %.4f ≠ %.4f dans metadata.json : jeu de test différent "
                        "de celui de l'entraînement (données modifiées depuis ?)", nom, obtenu, attendu)
        else:
            log.info("%s identique à metadata.json (%.4f) : même jeu de test", nom, obtenu)


def matrice_confusion(y_true: pd.Series, y_pred) -> None:
    """Écrit la matrice de confusion : effectifs (CSV) et % par vraie classe (PNG)."""
    effectifs = confusion_matrix(y_true, y_pred, labels=ORDRE_ETIQUETTES)
    pd.DataFrame(effectifs, index=ORDRE_ETIQUETTES, columns=ORDRE_ETIQUETTES).rename_axis(
        index="vraie classe", columns="classe prédite").to_csv(REPORTS / "matrice_confusion.csv")

    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=ORDRE_ETIQUETTES, normalize="true", values_format=".0%",
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set(title="Matrice de confusion (en % de chaque vraie classe)",
           xlabel="Classe prédite", ylabel="Vraie classe")
    fig.tight_layout()
    fig.savefig(REPORTS / "matrice_confusion.png", dpi=150)
    plt.close(fig)


def importances(clf, reg, X, y_clf, y_reg) -> pd.DataFrame:
    """Importance par permutation des variables, pour les deux modèles.

    Score : F1 macro (classification), MAE (régression).

    Returns
    -------
    DataFrame
        Index : variables ; colonnes : moyenne et écart-type par modèle.
        Aussi écrit dans reports/importance_variables.csv.
    """
    resultats = {}
    for cle, modele, y, score in [("classification", clf, y_clf, "f1_macro"),
                                  ("regression", reg, y_reg, "neg_mean_absolute_error")]:
        log.info("Permutation (%s, %d répétitions par variable)...", cle, N_REPETITIONS)
        r = permutation_importance(modele, X, y, scoring=score, n_repeats=N_REPETITIONS,
                                   random_state=RANDOM_STATE, n_jobs=1)
        resultats[f"{cle}_moyenne"] = r.importances_mean
        resultats[f"{cle}_ecart_type"] = r.importances_std
    table = pd.DataFrame(resultats, index=X.columns).rename_axis("variable")
    table.to_csv(REPORTS / "importance_variables.csv", float_format="%.5f")
    return table


def graphique_importances(table: pd.DataFrame) -> None:
    """Écrit le graphique des importances (deux panneaux, barres horizontales)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, cle, titre, unite in [
        (axes[0], "classification", "Étiquette DPE", "baisse du F1 macro"),
        (axes[1], "regression", "Consommation", "hausse de l'erreur moyenne (kWh/an)"),
    ]:
        t = table.sort_values(f"{cle}_moyenne")
        ax.barh([NOMS[v] for v in t.index], t[f"{cle}_moyenne"], xerr=t[f"{cle}_ecart_type"],
                color="#4c78a8")
        ax.set(title=titre, xlabel=unite)
        ax.axvline(0, color="grey", linewidth=0.8)
    fig.suptitle("Importance des variables (permutation sur le jeu de test)")
    fig.tight_layout()
    fig.savefig(REPORTS / "importance_variables.png", dpi=150)
    plt.close(fig)


def main() -> None:
    """Produit les quatre fichiers de reports/."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    REPORTS.mkdir(exist_ok=True)
    test = jeu_de_test()
    X = test[FEATURES]
    clf, reg = joblib.load(CLF_PATH), joblib.load(REG_PATH)
    log.info("Jeu de test : %d lignes", len(test))

    y_pred = clf.predict(X)
    clf_metrics = classification_metrics(test[TARGET_CLF], y_pred)
    reg_metrics = regression_metrics(test[TARGET_REG], reg.predict(X))
    verifier_scores(clf_metrics, reg_metrics)

    matrice_confusion(test[TARGET_CLF], y_pred)
    table = importances(clf, reg, X, test[TARGET_CLF], test[TARGET_REG])
    graphique_importances(table)
    log.info("Fichiers écrits dans %s : matrice_confusion.png/.csv, importance_variables.png/.csv", REPORTS)


if __name__ == "__main__":
    main()
