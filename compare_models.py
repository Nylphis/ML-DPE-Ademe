"""Comparaison des modèles candidats : forêt aléatoire et HistGradientBoosting.

Chaque candidat est évalué deux fois, sur le même échantillon :
- validation croisée par adresse (StratifiedGroupKFold) : l'évaluation retenue ;
- validation croisée aléatoire (StratifiedKFold) : seulement pour mesurer
  combien les DPE d'une même adresse gonflaient les scores.

Utilise exactement le pipeline de train.py (make_model). Pour chaque candidat :
scores, temps d'entraînement, taille du modèle sur l'échantillon et taille
estimée sur le jeu complet.

Lancement (depuis la racine, après `python run.py clean`) :
    python compare_models.py                  # 50 000 lignes
    python compare_models.py --lignes 20000   # plus rapide
Résultats : reports/comparaison_modeles.csv
"""

import argparse
import io
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from dpe.config import (DATA_CLEAN, FEATURES, GROUP, RANDOM_STATE, ROOT,
                        TARGET_CLF, TARGET_REG)
from dpe.features import to_model_frame
from dpe.train import (categories_from, classification_metrics, clip_target,
                       make_model, regression_metrics)

N_PLIS = 3
SORTIE = ROOT / "reports" / "comparaison_modeles.csv"

# (nom affiché, type de modèle, paramètres)
CANDIDATS = {
    "clf": [
        ("RF leaf=1", "rf", {"n_estimators": 100, "min_samples_leaf": 1, "class_weight": "balanced"}),
        ("RF leaf=1 ms=0.3", "rf", {"n_estimators": 100, "min_samples_leaf": 1, "max_samples": 0.3, "class_weight": "balanced"}),
        ("RF leaf=5", "rf", {"n_estimators": 100, "min_samples_leaf": 5, "class_weight": "balanced"}),
        ("RF leaf=20", "rf", {"n_estimators": 100, "min_samples_leaf": 20, "class_weight": "balanced"}),
        ("HGB lr=0.1 f=63", "hgb", {"learning_rate": 0.1, "max_leaf_nodes": 63, "max_iter": 500, "early_stopping": True, "class_weight": "balanced"}),
        ("HGB lr=0.05 f=127", "hgb", {"learning_rate": 0.05, "max_leaf_nodes": 127, "max_iter": 500, "early_stopping": True, "class_weight": "balanced"}),
        ("HGB lr=0.05 f=255", "hgb", {"learning_rate": 0.05, "max_leaf_nodes": 255, "max_iter": 500, "early_stopping": True, "class_weight": "balanced"}),
    ],
    "reg": [
        ("RF notebook (d=15, leaf=5)", "rf", {"n_estimators": 100, "max_depth": 15, "min_samples_leaf": 5}),
        ("RF leaf=1", "rf", {"n_estimators": 100, "min_samples_leaf": 1}),
        ("RF leaf=20", "rf", {"n_estimators": 100, "min_samples_leaf": 20}),
        ("HGB lr=0.1 f=63", "hgb", {"learning_rate": 0.1, "max_leaf_nodes": 63, "max_iter": 500, "early_stopping": True}),
        ("HGB lr=0.05 f=127", "hgb", {"learning_rate": 0.05, "max_leaf_nodes": 127, "max_iter": 500, "early_stopping": True}),
        ("HGB lr=0.05 f=255", "hgb", {"learning_rate": 0.05, "max_leaf_nodes": 255, "max_iter": 500, "early_stopping": True}),
    ],
}
SCORE_PRINCIPAL = {"clf": "f1_macro", "reg": "rmse"}


def charger() -> pd.DataFrame:
    df = pd.read_csv(DATA_CLEAN, dtype={"code_postal_ban": str, GROUP: str})
    return pd.concat([df[[TARGET_CLF, TARGET_REG, GROUP]], to_model_frame(df)], axis=1)


def echantillon_par_adresse(df: pd.DataFrame, n_lignes: int) -> pd.DataFrame:
    """Tire des adresses entières au hasard jusqu'à atteindre environ n_lignes."""
    if n_lignes >= len(df):
        return df
    tailles = df[GROUP].value_counts().sample(frac=1, random_state=RANDOM_STATE)
    gardees = tailles.index[tailles.cumsum() <= n_lignes]
    return df[df[GROUP].isin(gardees)]


def taille_mo(model) -> float:
    buffer = io.BytesIO()
    joblib.dump(model, buffer, compress=3)
    return buffer.tell() / 1e6


def evaluer(task, estimator, params, ech, plis, categories) -> dict:
    """Validation croisée manuelle (permet d'écrêter la cible sur chaque train)."""
    scores, temps = [], []
    for i_train, i_test in plis:
        train, test = ech.iloc[i_train], ech.iloc[i_test]
        model = make_model(task, estimator, params, categories)
        debut = time.time()
        if task == "clf":
            model.fit(train[FEATURES], train[TARGET_CLF])
            scores.append(classification_metrics(test[TARGET_CLF], model.predict(test[FEATURES])))
        else:
            y_train, _ = clip_target(train[TARGET_REG])
            model.fit(train[FEATURES], y_train)
            scores.append(regression_metrics(test[TARGET_REG], model.predict(test[FEATURES])))
        temps.append(time.time() - debut)
    moyennes = pd.DataFrame(scores).mean().to_dict()
    moyennes.update(temps_s=float(np.mean(temps)), taille_mo=taille_mo(model), n_fit=len(train))
    return moyennes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lignes", type=int, default=50_000, help="taille de l'échantillon")
    args = parser.parse_args(argv)

    df = charger()
    categories = categories_from(df)  # sur tout le jeu, comme dans train.py
    ech = echantillon_par_adresse(df, args.lignes).reset_index(drop=True)
    n_train_complet = len(df) * (1 - 1 / N_PLIS)  # taille du train dans train.py
    print(f"Échantillon : {len(ech)} lignes, {ech[GROUP].nunique()} adresses "
          f"(jeu complet : {len(df)} lignes)\n")

    # Mêmes plis pour tous les candidats, stratifiés sur l'étiquette dans les deux cas
    y = ech[TARGET_CLF]
    decoupages = {
        "adresse": list(StratifiedGroupKFold(N_PLIS, shuffle=True, random_state=RANDOM_STATE)
                        .split(ech, y, groups=ech[GROUP])),
        "aleatoire": list(StratifiedKFold(N_PLIS, shuffle=True, random_state=RANDOM_STATE).split(ech, y)),
    }

    lignes = []
    for task, candidats in CANDIDATS.items():
        score = SCORE_PRINCIPAL[task]
        for nom, estimator, params in candidats:
            res = {"tache": task, "modele": nom}
            for decoupage, plis in decoupages.items():
                r = evaluer(task, estimator, params, ech, plis, categories)
                res.update({f"{k}_{decoupage}": v for k, v in r.items()
                            if k not in ("taille_mo", "n_fit", "temps_s")})
                if decoupage == "adresse":
                    res.update(temps_s=r["temps_s"], taille_mo_ech=r["taille_mo"])
                    # La taille d'une forêt croît avec le nombre de lignes, pas celle du HGB
                    facteur = n_train_complet / r["n_fit"] if estimator == "rf" else 1
                    res["taille_mo_complet_estimee"] = r["taille_mo"] * facteur
            res["ecart_aleatoire_adresse"] = res[f"{score}_aleatoire"] - res[f"{score}_adresse"]
            lignes.append(res)
            print(f"[{task}] {nom:28s} {score} adresse {res[f'{score}_adresse']:9.3f} | "
                  f"aléatoire {res[f'{score}_aleatoire']:9.3f} | {res['temps_s']:5.1f} s | "
                  f"~{res['taille_mo_complet_estimee']:6.1f} Mo sur le jeu complet")

    resultats = pd.DataFrame(lignes)
    SORTIE.parent.mkdir(parents=True, exist_ok=True)
    resultats.to_csv(SORTIE, index=False)

    pd.set_option("display.width", 200)
    colonnes = {
        "clf": ["modele", "f1_macro_adresse", "f1_macro_aleatoire", "accuracy_adresse",
                "accuracy_1_classe_adresse", "recall_A-B_adresse", "recall_G_adresse",
                "temps_s", "taille_mo_complet_estimee"],
        "reg": ["modele", "rmse_adresse", "rmse_aleatoire", "mae_adresse", "r2_adresse",
                "temps_s", "taille_mo_complet_estimee"],
    }
    for task, cols in colonnes.items():
        tri = resultats[resultats["tache"] == task].sort_values(
            cols[1], ascending=(task == "reg"))
        print(f"\n=== {'Classification' if task == 'clf' else 'Régression'} "
              f"(validation croisée par adresse, {N_PLIS} plis) ===")
        print(tri[cols].round(3).to_string(index=False))
    print(f"\nRésultats complets : {SORTIE}")


if __name__ == "__main__":
    main()
