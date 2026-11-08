"""Orchestrateur du pipeline DPE.

Exemples :
    python run.py                       # fetch + clean + train
    python run.py clean train           # sans passer par l'étape fetch
    python run.py train                 # réentraîne sur data/processed/dpe_clean.csv
    python run.py --refetch             # force un nouveau téléchargement complet
    python run.py --max-lignes 10000    # échantillon rapide (téléchargé une fois, puis réutilisé)

L'étape fetch ne rappelle pas l'API si data/raw contient déjà le résultat de
la même requête : relancer `python run.py` après un téléchargement complet
ne fait que le nettoyage et l'entraînement.
"""

import argparse
import logging
import sys

STEPS = ["fetch", "clean", "train"]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pipeline DPE Lyon")
    # Pas de default=STEPS : selon la version de Python, argparse valide la
    # liste par défaut entière contre `choices` et la rejette.
    parser.add_argument("steps", nargs="*", choices=STEPS,
                        help="étapes à lancer, dans l'ordre (par défaut : toutes)")
    parser.add_argument("--refetch", action="store_true",
                        help="retélécharger même si les données brutes sont déjà là")
    parser.add_argument("--max-lignes", type=int, default=None,
                        help="limiter le téléchargement à un échantillon (par défaut : tout)")
    args = parser.parse_args(argv)

    if (args.refetch or args.max_lignes) and "fetch" not in steps:
        parser.error("--refetch et --max-lignes ne servent qu'avec l'étape fetch")
        
    steps = args.steps or STEPS

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")

    # Imports différés : `python run.py train` ne requiert pas `requests`
    for step in STEPS:
        if step not in steps:
            continue
        logging.info("=== Étape %s ===", step)
        if step == "fetch":
            from dpe.fetch import fetch
            fetch(refetch=args.refetch, max_lignes=args.max_lignes)
        elif step == "clean":
            from dpe.clean import run as clean
            clean()
        elif step == "train":
            from dpe.train import run as train
            train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
