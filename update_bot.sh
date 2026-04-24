#!/bin/bash
# Skrip untuk memperbarui bot dari GitHub dan menjalankan versi terbaru

cd ~/btc_scalping_bot

echo "========================================"
echo "[$(date)] Mengambil kode terbaru dari GitHub..."
git pull

echo "[$(date)] Menghentikan bot lama (jika berjalan)..."
# Menghentikan proses python main.py
pkill -f "python main.py"
sleep 2

echo "[$(date)] Menjalankan bot versi terbaru..."
# Mengirimkan perintah ke sesi screen yang bernama 'bot' untuk menjalankan run_bot.sh
# Jika sesi screen belum ada, buat sesi baru dan jalankan di dalamnya
if screen -list | grep -q "\.bot"; then
    screen -S bot -X stuff "./run_bot.sh\n"
    echo "[$(date)] Bot telah di restart dalam sesi screen 'bot'."
else
    echo "[$(date)] Sesi screen 'bot' tidak ditemukan. Membuat sesi baru..."
    screen -S bot -dm bash -c "./run_bot.sh"
    echo "[$(date)] Bot dijalankan dalam sesi screen baru 'bot'."
fi

echo "[$(date)] Selesai. Bot telah diperbarui."
echo "========================================"
