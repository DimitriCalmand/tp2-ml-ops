FROM python:3.11-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers Python
COPY app.py /app/
COPY requirements.txt /app/
COPY mlruns /app/mlruns

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Exposer le port Flask
EXPOSE 5000

# Commande pour lancer le serveur Flask
CMD ["python", "app.py"]
