# Essais initiaux de modèles — Xilian Jing

Ce dossier archive les trois modèles testés par Xilian pendant la phase initiale
du projet. Ce ne sont pas les pipelines HGB actuellement utilisés dans l’application.

## Contenu et ordre d’exécution

Exécuter les notebooks avec ce dossier comme répertoire de travail :

1. `01_inspection_donnees.ipynb` : inspection et préparation des données, partition
   par groupes de bâtiments, **classification par régression logistique**, validation
   croisée, sélection des hyperparamètres et évaluation sur le test.
   Lit `dpe_lyon_raw.csv` et produit `dpe_donnees_communes.csv` et
   `classification_resultats_finaux.csv`.
2. `02_ridge.ipynb` : Ridge, validation croisée par groupes, sélection d’alpha et
   évaluation sur le test. Lit `dpe_donnees_communes.csv` et produit notamment
   `ridge_resultats_finaux.csv`.
3. `03_arbre_regression.ipynb` : arbre de régression, sélection des paramètres,
   évaluation et comparaison avec Ridge. Lit le fichier commun et les résultats
   Ridge ; la dernière cellule de synthèse lit aussi les résultats de classification.

Les fichiers CSV d’entrée et les trois petits tableaux de résultats sont inclus
pour rendre les dépendances explicites. Les `.joblib` et images sont produits
par les notebooks ; ils ne sont pas nécessaires pour lire les résultats archivés.
Ne pas remplacer les dossiers `data/` ou `models/` de l’application par ces fichiers.

## Protocole de ces essais

- Fichier initial : 10 000 diagnostics de Lyon, 226 colonnes.
- Exclusion de 79 diagnostics d’immeubles entiers ; 9 921 lignes conservées.
- Configuration finale de ces notebooks : **15 entrées**, classification en
  **7 classes A, B, C, D, E, F, G**, régression sur `conso_5_usages_ef` (kWh EF/an).
- Les premières cellules d’inspection explorent aussi l’énergie primaire
  (`conso_5_usages_ep`) et une ancienne sélection de huit variables. La
  configuration à partir de la cellule qui définit les 15 entrées les remplace.
  Les scores finaux ci-dessous correspondent bien à cette configuration à 15 entrées
  et, en régression, à l’énergie finale.
- Groupes construits par liens entre `id_rnb` et `numero_dpe_immeuble_associe`.
  Il s’agit des groupes de bâtiments identifiés, pas de la méthode BAN utilisée
  ultérieurement par le pipeline final.
- `GroupShuffleSplit(test_size=0.30, random_state=42)` : 6 915 lignes pour
  l’apprentissage de classification, 3 006 pour le test. Une cible de régression
  manque dans le train : 6 914 lignes d’apprentissage en régression.
- Vérification du fichier commun : aucun identifiant de groupe n’est présent à
  la fois dans le train et le test. Cela ne démontre pas que tout lien d’adresse
  possible a été identifié.
- Imputation et encodage dans des `Pipeline` sklearn ; standardisation pour
  la régression logistique et Ridge. Réglage des hyperparamètres sur le train
  par validation croisée en 3 plis, avec groupes.
- Logistique : sélection par F1 macro, `C=10`, `class_weight=None`.
- Ridge : sélection par RMSE, `alpha=10`.
- Arbre : sélection par RMSE, `max_depth=20`, `min_samples_leaf=20`.

## Résultats de test enregistrés

| Modèle de classification | Accuracy | Accuracy à ±1 classe | F1 macro | F1 pondéré |
|---|---:|---:|---:|---:|
| Régression logistique | 0,517 | 0,908 | 0,301 | 0,495 |

| Modèle de régression | RMSE (kWh EF/an) | MAE (kWh EF/an) | R² |
|---|---:|---:|---:|
| Ridge | 8 120,910 | 3 012,212 | 0,333 |
| Arbre de régression | 8 252,371 | 2 818,669 | 0,312 |

Les résultats sont ceux enregistrés lors des essais initiaux. Ils ne doivent pas
être comparés directement aux scores du pipeline HGB actuel : données,
variables, définition des classes et groupes de séparation diffèrent.
Ridge peut produire des prédictions négatives. Les extrêmes du test pèsent
fortement sur le RMSE : le notebook Ridge rapporte que les dix plus grosses
erreurs représentent 79,1 % de la somme des erreurs au carré.

## Vérification lors de la préparation de l’archive

- Lecture des trois notebooks et contrôle de leur syntaxe Python : aucun
  résultat d’erreur sauvegardé dans les cellules.
- Les RMSE, MAE et R² des deux régressions ont été recalculés à partir des
  fichiers de prédictions existants et correspondent aux tableaux ci-dessus.
- Le F1 de classification a été recoupé avec le tableau et les sorties du
  notebook ; la classification n’a pas été entièrement réentraînée pour cette archive.
- Le dernier récapitulatif de `03_arbre_regression.ipynb` utilisait un chemin
  absolu vers un ancien dossier du bureau. Il est remplacé par `Path(".")`.
  Seule la sortie de cette cellule modifiée est effacée ; les autres résultats
  historiques sont conservés. Les sources originales restent inchangées.
- Les notebooks n’ont pas été réexécutés intégralement pour préparer ce dossier.

## Environnement

Les notebooks historiques indiquent Python 3.13.5 et le noyau `base`.
Ils utilisent pandas, NumPy, scikit-learn, matplotlib, seaborn, joblib et un
environnement Jupyter/IPython. Les versions exactes des paquets lors des essais
ne sont pas attestées dans ces notebooks ; aucune version historique n’est inventée.
Un environnement récent différent peut donner des avertissements ou des écarts.
Cela n’impose aucun changement au `requirements.txt` commun de l’application.
