import pandas as pd
from dpe.config import FEATURES, TARGET_CLF

df = pd.read_csv("data/processed/dpe_clean.csv", dtype={"code_postal_ban": str})
dup = df.duplicated(FEATURES, keep=False)
print(f"Lignes dont les 14 variables existent ailleurs : {dup.mean():.1%}")

# Parmi ces doublons, quelle part a la même étiquette que la majorité de son groupe ?
g = df[dup].groupby(FEATURES, dropna=False)[TARGET_CLF]
accord = g.transform(lambda s: s.eq(s.mode().iloc[0]).mean())
print(f"Accord d'étiquette dans les groupes de doublons : {accord.mean():.1%}")