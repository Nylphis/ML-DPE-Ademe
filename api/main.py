"""API des modèles DPE (FastAPI) : étiquette et consommation d'un logement, sans interface.

Routes : POST /predict, GET /info ; documentation interactive sur /docs.
Mêmes modèles, même mise en forme (to_model_frame) et mêmes règles que la
page Prédiction : prédictions identiques à celles de l'appli.

Lancement (depuis la racine du projet) :
    python -m uvicorn api.main:app --port 8000
"""

import json
from typing import Literal, Optional

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field, create_model

from dpe.config import (BINARY, CHAMPS_APPARTEMENT, CLF_PATH, FEATURES, METADATA_PATH, REG_PATH,
                        REQUIRED)
from dpe.features import to_model_frame

# Chargement unique, au démarrage
MODELES = None
if all(p.exists() for p in (CLF_PATH, REG_PATH, METADATA_PATH)):
    MODELES = joblib.load(CLF_PATH), joblib.load(REG_PATH)
METADATA: dict = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
OPTIONS: dict[str, list] = METADATA.get("options", {})


def _champ(champ: str):
    """Type Pydantic et contraintes d'un champ du modèle d'entrée.

    Catégories : liste fermée (valeurs vues à l'entraînement) ; nombres : bornes.

    Returns
    -------
    tuple
        (type, Field) pour create_model. Champ facultatif : None par défaut
        (« inconnu », comme un champ vide dans l'appli).
    """
    contraintes = {}
    if champ in OPTIONS:
        type_ = Literal[tuple(OPTIONS[champ])]
    elif champ in BINARY:
        type_ = Literal[0, 1]
    elif champ == "surface_habitable_logement":
        type_, contraintes = float, {"gt": 0}
    elif champ == "nombre_niveau_logement":
        type_, contraintes = int, {"ge": 1}
    else:
        type_ = str
    if champ in REQUIRED:
        return type_, Field(**contraintes)
    return Optional[type_], Field(default=None, **contraintes)


# Modèle d'entrée construit depuis FEATURES et les valeurs vues à l'entraînement
Logement = create_model(
    "Logement",
    __config__=ConfigDict(extra="forbid", json_schema_extra={"examples": [
        {"code_postal_ban": "69003", "type_batiment": "appartement", "surface_habitable_logement": 65},
    ]}),
    **{champ: _champ(champ) for champ in FEATURES},
)


class Estimation(BaseModel):
    """Réponse de POST /predict."""

    etiquette_dpe: str = Field(description="Classe estimée : A-B (A et B regroupées), C, D, E, F ou G")
    conso_5_usages_ef: float = Field(description="Consommation annuelle d'énergie finale estimée, en kWh/an")
    champs_non_renseignes: list[str] = Field(description="Champs absents, traités comme « inconnus » par le modèle")


app = FastAPI(
    title="API DPE Lyon",
    description="Estime l'étiquette DPE et la consommation d'énergie d'un logement de Lyon et d'une partie "
                "de la métropole. Seuls code_postal_ban, type_batiment et surface_habitable_logement sont "
                "obligatoires ; un champ absent est traité comme « inconnu ».",
)


@app.get("/", include_in_schema=False)
def accueil():
    """Redirection vers la documentation interactive."""
    return RedirectResponse("/docs")


@app.get("/info", summary="Version, performances et valeurs acceptées")
def info() -> dict:
    """Date d'entraînement, champs obligatoires, valeurs acceptées, étiquettes et métriques."""
    return {
        "date_entrainement": METADATA.get("date_entrainement"),
        "champs_obligatoires": REQUIRED,
        "champs": FEATURES,
        "valeurs_acceptees": OPTIONS,
        "etiquettes": METADATA.get("etiquettes"),
        "metriques": METADATA.get("metriques"),
    }


@app.post("/predict", summary="Estimer l'étiquette DPE et la consommation d'un logement")
def predict(logement: Logement) -> Estimation:
    """Estimation d'un logement.

    Raises
    ------
    HTTPException
        503 si les modèles sont absents. Saisie invalide : 422 (validation Pydantic).
    """
    if MODELES is None:
        raise HTTPException(status_code=503, detail="Modèles non disponibles : lancer python run.py.")
    saisie = logement.model_dump()
    # Comme l'appli : champs ignorés pour une maison
    sans_objet = CHAMPS_APPARTEMENT if saisie["type_batiment"] == "maison" else []
    for champ in sans_objet:
        saisie[champ] = None
    clf, reg = MODELES
    X = to_model_frame(saisie)
    return Estimation(
        etiquette_dpe=str(clf.predict(X)[0]),
        conso_5_usages_ef=round(float(reg.predict(X)[0]), 1),
        champs_non_renseignes=[c for c in FEATURES if saisie[c] is None and c not in sans_objet],
    )
