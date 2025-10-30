#!/bin/bash

# Configuration
SOURCE_FILE="$HOME/gpt_trading_bot/trades_log.csv"
DEST_DIR="$HOME/trading-dashboard"
DEST_FILE="$DEST_DIR/trades_log.csv"

# Copier le fichier
cp "$SOURCE_FILE" "$DEST_FILE"

# Aller dans le dossier du repo
cd "$DEST_DIR"

# Vérifier s'il y a des changements
if git diff --quiet trades_log.csv; then
    echo "Aucun changement détecté dans trades_log.csv"
    exit 0
fi

# Ajouter et committer
git add trades_log.csv
git commit -m "Update trades_log.csv - $(date '+%Y-%m-%d %H:%M:%S')"

# Pousser vers GitHub
git push origin main

echo "✅ Fichier synchronisé avec GitHub !"
