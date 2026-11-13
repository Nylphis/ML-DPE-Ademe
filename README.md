# DPE Lyon

Application Streamlit qui estime l'**étiquette DPE** (A-B à G) et la **consommation annuelle d'énergie finale** (kWh/an) d'un logement de Lyon et d'une partie de la métropole. Les modèles sont entraînés sur les DPE de l'ADEME (logements existants, DPE version 2.1 et suivantes).

En ligne : https://ml-dpe-ademe.onrender.com

## Pages

| Page | Contenu |
|---|---|
| Contexte | Le projet, les données utilisées, les deux objectifs (classification, régression), les performances des modèles, l'échelle DPE |
| Exploration | Indicateurs, graphiques, carte et filtres sur les données (en cours) |
| Prédiction | Formulaire du logement (3 champs obligatoires), étiquette et consommation estimées, export CSV et JSON |

## Structure

```
app.py                      # configuration et menu de l'appli
pages/                      # une page par fichier, éléments communs dans commun.py
dpe/                        # pipeline : config, fetch (API ADEME), clean, train, features
run.py                      # lance le pipeline : python run.py [fetch] [clean] [train]
compare_models.py           # comparaison des modèles candidats -> reports/
data/processed/             # données nettoyées et statistiques
models/                     # modèles entraînés (.joblib) et metadata.json
reports/                    # résultats de la comparaison des modèles
.github/workflows/          # mise à jour mensuelle des données (GitHub Actions)
Dockerfile                  # image de l'appli
requirements.txt            # dépendances (Python 3.12)
```
