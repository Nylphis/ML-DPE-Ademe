# tune_clf.py
import io, time, joblib, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from dpe.config import FEATURES, RANDOM_STATE, TARGET_CLF, TARGET_REG
from dpe.features import to_model_frame
from dpe.train import build_preprocessor, categories_from

df = pd.read_csv("data/processed/dpe_clean.csv", dtype={"code_postal_ban": str})
df = pd.concat([df[[TARGET_CLF, TARGET_REG]], to_model_frame(df)], axis=1)
cats = categories_from(df)  # sur tout le jeu, comme dans train.py
ech, _ = train_test_split(df, train_size=50_000, stratify=df[TARGET_CLF], random_state=RANDOM_STATE)

pipe = Pipeline([("prep", build_preprocessor(cats, clip_numeric=False)),
                 ("rf", RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                               random_state=RANDOM_STATE, n_jobs=-1))])
grid = GridSearchCV(pipe, {"rf__min_samples_leaf": [1, 5, 10, 20, 50],
                           "rf__max_samples": [0.3, 0.5, None]},
                    scoring="f1_macro", cv=3, n_jobs=1)
grid.fit(ech[FEATURES], ech[TARGET_CLF])

res = pd.DataFrame(grid.cv_results_)
res = res[["param_rf__min_samples_leaf", "param_rf__max_samples", "mean_test_score", "mean_fit_time"]]
# taille de chaque combinaison, refittée sur l'échantillon
tailles = []
for p in grid.cv_results_["params"]:
    m = pipe.set_params(**p).fit(ech[FEATURES], ech[TARGET_CLF])
    b = io.BytesIO(); joblib.dump(m, b, compress=3); tailles.append(b.tell() / 1e6)
res["taille_mo_50k"] = tailles


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score

for lr, leaves in [(0.1, 31), (0.1, 63), (0.05, 127)]:
    hgb = Pipeline([("prep", build_preprocessor(cats, clip_numeric=False)),
                    ("hgb", HistGradientBoostingClassifier(
                        learning_rate=lr, max_leaf_nodes=leaves, max_iter=500,
                        early_stopping=True, class_weight="balanced",
                        random_state=RANDOM_STATE))])
    t = time.time()
    f1 = cross_val_score(hgb, ech[FEATURES], ech[TARGET_CLF], scoring="f1_macro", cv=3).mean()
    m = hgb.fit(ech[FEATURES], ech[TARGET_CLF])
    b = io.BytesIO(); joblib.dump(m, b, compress=3)
    print(f"HGB lr={lr} leaves={leaves} : F1 {f1:.3f} | {time.time()-t:.0f} s | {b.tell()/1e6:.1f} Mo")


print(res.sort_values("mean_test_score", ascending=False).to_string(index=False))