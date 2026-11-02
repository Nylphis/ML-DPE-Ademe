import requests
import pandas as pd
import time

URL_BASE = "https://data.ademe.fr/data-fair/api/v1/datasets/meg-83tjwtg8dyz4vv7h1dqe/lines"

colonnes_api = [
    "numero_dpe",
    "date_etablissement_dpe",
    "version_dpe",
    "methode_application_dpe",
    "numero_dpe_immeuble_associe",
    "etiquette_dpe",
    "conso_5_usages_ef",
    "code_postal_ban",
    "type_batiment",
    "surface_habitable_logement",
    "periode_construction",
    "nombre_niveau_logement",
    "type_energie_principale_chauffage",
    "type_generateur_chauffage_principal",
    "type_installation_chauffage",
    "type_energie_principale_ecs",
    "type_installation_ecs",
    "isolation_toiture",
    "qualite_isolation_menuiseries",
    "surface_climatisee",
    "logement_traversant",
    "type_generateur_n1_installation_n1",
]

codes_postaux = [
    "69001", "69002", "69003", "69004", "69005", "69006", "69007", "69008", "69009",
    "69100", "69500", "69300", "69130", "69150", "69600", "69140", "69700",
]

# ------------------------------------------------
# Vérification : les colonnes demandées existent-elles dans l'API ?
# ------------------------------------------------
URL_DATASET = "https://data.ademe.fr/data-fair/api/v1/datasets/meg-83tjwtg8dyz4vv7h1dqe"

schema = requests.get(URL_DATASET).json()["schema"]
cles_api = {champ["key"] for champ in schema}

print("Colonnes absentes de l'API :", [c for c in colonnes_api if c not in cles_api])

# ------------------------------------------------
# Extraction paginée
# ------------------------------------------------
url = URL_BASE
params = {
    "qs": (
        f"code_postal_ban:({' OR '.join(codes_postaux)})"
        " AND version_dpe:[2.1 TO *]"
        " AND NOT type_batiment:immeuble"
    ),
    "select": ",".join(colonnes_api),
    "size": 10000,
}

toutes_les_lignes = []

# ------------------------------------------------
# While d'extraction
# ------------------------------------------------
#while url:
response = requests.get(url, params=params)
response.raise_for_status()  # stoppe proprement en cas d'erreur HTTP
data = response.json()

toutes_les_lignes.extend(data["results"])
print(f"Nombre de lignes : {len(toutes_les_lignes)} / {data.get('total')}")

# Les paramètres sont déjà présents dans next
url = data.get("next")
params = None
time.sleep(0.2)  # évite de surcharger l'API

# ------------------------------------------------
# Test du dataset et sauvegarde des données brutes (avant tout filtre local)
# ------------------------------------------------
dfExistant = pd.DataFrame(toutes_les_lignes)
print(dfExistant.shape)

dfExistant.to_csv("dpe03existant.csv", index=False)