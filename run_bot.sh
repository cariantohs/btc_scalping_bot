#!/bin/bash
# Skrip untuk menjalankan bot secara terus-menerus
# Jika bot berhenti, akan restart setelah 5 detik

while true; do
    echo "[$(date)] Memulai bot..."
    python main.py
    echo "[$(date)] Bot berhenti. Menunggu 5 detik sebelum restart..."
    sleep 5
done
