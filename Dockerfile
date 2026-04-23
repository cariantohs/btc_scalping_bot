# Gunakan image resmi Python
FROM python:3.10-slim

# Izinkan output Python langsung tampil di log
ENV PYTHONUNBUFFERED True

# Setel direktori kerja di dalam container
WORKDIR /app

# Salin file requirements dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua kode aplikasi
COPY . .

# Ekspos port 8080 untuk health check Back4App (WAJIB)
EXPOSE 8080

# Jalankan bot (server HTTP akan berjalan di dalam main.py)
CMD ["python", "main.py"]
