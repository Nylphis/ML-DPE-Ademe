# Image de l'appli Streamlit : elle lit les modèles et les données déjà présents
# dans le dépôt (models/, data/processed/). Aucun téléchargement ni entraînement ici.
#
# Construction : docker build -t dpe-lyon .
# Lancement    : docker run -p 8501:8501 dpe-lyon   puis http://localhost:8501

# Même version de Python que l'environnement de développement
FROM python:3.12-slim

WORKDIR /app

# Dépendances d'abord : cette couche reste en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render fournit le port dans la variable PORT ; 8501 en local
ENV PORT=8501
EXPOSE 8501

# sh -c pour lire $PORT au lancement ; exec pour que Streamlit reçoive les signaux d'arrêt.
# fileWatcherType=none : le code ne change pas en production, inutile de surveiller les fichiers.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]
