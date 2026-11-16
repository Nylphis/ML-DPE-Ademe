"""Éléments partagés par les pages : modèles, métadonnées, échelle DPE, formats.

Module importé par les pages, non déclaré dans st.navigation (app.py).
"""

import json
from typing import Any

import joblib
import streamlit as st

from dpe.config import CLF_PATH, METADATA_PATH, REG_PATH


# ------------------------------------------------
# Modèles (chargés une fois par processus)
# ------------------------------------------------
@st.cache_resource
def load_models() -> tuple[Any, Any, dict] | None:
    """Classifieur, régresseur et métadonnées ; None si un fichier manque."""
    if not all(p.exists() for p in (CLF_PATH, REG_PATH, METADATA_PATH)):
        return None
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return joblib.load(CLF_PATH), joblib.load(REG_PATH), metadata


MODELS = load_models()
METADATA: dict = MODELS[2] if MODELS else {}
OPTIONS: dict[str, list] = METADATA.get("options", {})


# ------------------------------------------------
# Échelle DPE et formats
# ------------------------------------------------
COULEURS_DPE = {"A": "#009c6d", "B": "#52b153", "C": "#78bd76", "D": "#f4e70f",
                "E": "#f0b40f", "F": "#eb8235", "G": "#d7221f"}


def echelle_dpe(etiquette: str | None = None) -> str:
    """Échelle DPE (A à G) en SVG.

    Parameters
    ----------
    etiquette : str, optional
        Classe à mettre en avant (« A-B » : A et B) ; autres classes estompées.

    Returns
    -------
    str
        Code SVG, affichable avec st.image.
    """
    surlignees = etiquette.split("-") if etiquette else list(COULEURS_DPE)
    barres = []
    for i, (lettre, couleur) in enumerate(COULEURS_DPE.items()):
        y, largeur = 24 + i * 34, 90 + i * 35
        opacite = 1 if lettre in surlignees else 0.25
        texte = "#333" if lettre == "D" else "#fff"
        barres.append(
            f'<g opacity="{opacite}"><rect x="0" y="{y}" width="{largeur}" height="28" rx="4" fill="{couleur}"/>'
            f'<text x="{largeur - 12}" y="{y + 20}" text-anchor="end" font-size="18" font-weight="bold" '
            f'fill="{texte}">{lettre}</text></g>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="282" font-family="sans-serif">'
        '<text x="0" y="14" font-size="13" fill="#888">Logement économe</text>'
        + "".join(barres)
        + '<text x="0" y="278" font-size="13" fill="#888">Logement énergivore</text></svg>'
    )


def fmt_int(n: Any) -> str:
    """Entier avec espace comme séparateur de milliers ; « — » si `n` n'est pas un entier."""
    return f"{n:,}".replace(",", " ") if isinstance(n, int) else "—"
