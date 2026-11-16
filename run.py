"""Orchestrateur du pipeline DPE : fetch, clean, train.

Exemples :
    python run.py                             # fetch + clean + train
    python run.py clean train                 # sans téléchargement
    python run.py train                       # réentraînement sur dpe_clean.csv
    python run.py --refetch                   # téléchargement complet forcé
    python run.py --max-lignes 10000          # échantillon (téléchargé une fois, puis réutilisé)
    python run.py fetch clean --incremental   # DPE créés ou modifiés depuis la dernière mise à jour

Étape fetch sans appel à l'API si data/raw contient déjà le résultat de la même requête.
"""

import argparse
import logging
import sys

STEPS = ["fetch", "clean", "train"]


def main(argv=None):
    """Point d'entrée en ligne de commande.

    Parameters
    ----------
    argv : list of str, optional
        Arguments (par défaut : sys.argv).

    Returns
    -------
    int
        Code de sortie (0).
    """
    parser = argparse.ArgumentParser(description="Pipeline DPE Lyon")
    # Pas de default=STEPS : argparse refuserait la liste entière contre `choices`
    parser.add_argument("steps", nargs="*", choices=STEPS,
                        help="étapes à lancer, dans l'ordre (par défaut : toutes)")
    parser.add_argument("--refetch", action="store_true",
                        help="retélécharger même si les données brutes sont déjà là")
    parser.add_argument("--max-lignes", type=int, default=None,
                        help="limiter le téléchargement à un échantillon (par défaut : tout)")
    parser.add_argument("--incremental", action="store_true",
                        help="fetch et clean : seulement les DPE créés ou modifiés depuis la dernière mise à jour")
    args = parser.parse_args(argv)
    steps = args.steps or STEPS

    if (args.refetch or args.max_lignes) and "fetch" not in steps:
        parser.error("--refetch et --max-lignes ne servent qu'avec l'étape fetch")
    if args.incremental and (args.refetch or args.max_lignes):
        parser.error("--incremental ne se combine pas avec --refetch ni --max-lignes")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s : %(message)s")

    # Imports différés : chaque étape ne charge que ses dépendances
    for step in STEPS:
        if step not in steps:
            continue
        logging.info("=== Étape %s ===", step)
        if step == "fetch":
            from dpe.fetch import fetch, fetch_nouveaux
            if args.incremental:
                fetch_nouveaux()
            else:
                fetch(refetch=args.refetch, max_lignes=args.max_lignes)
        elif step == "clean":
            from dpe.clean import run as clean, run_incremental
            if args.incremental:
                run_incremental()
            else:
                clean()
        elif step == "train":
            from dpe.train import run as train
            train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
