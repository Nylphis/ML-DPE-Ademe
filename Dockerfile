# Image de l'appli Streamlit (et de l'API, avec une autre commande de démarrage).
# Modèles et données lus dans le dépôt (models/, data/processed/) : ni
# téléchargement ni entraînement dans l'image.
#
# Construction : docker build -t dpe-lyon .
# Lancement    : docker run -p 8501:8501 dpe-lyon   puis http://localhost:8501

# Même version de Python que le développement
FROM python:3.12-slim

WORKDIR /app

# Dépendances d'abord : couche en cache tant que requirements.txt ne change pas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port fourni par Render dans PORT ; 8501 en local
ENV PORT=8501
EXPOSE 8501

# sh -c : lecture de $PORT au lancement ; exec : signaux d'arrêt transmis à Streamlit.
# fileWatcherType=none : pas de surveillance des fichiers en production.
CMD ["sh", "-c", "exec streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.fileWatcherType=none --browser.gatherUsageStats=false"]
