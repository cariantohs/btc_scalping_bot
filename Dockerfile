# 1. Pilih base image Python (versi 3.10)
FROM python:3.10-slim

# 2. Set working directory di dalam container
WORKDIR /app

# 3. Salin file requirements.txt dan install dependensi
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Salin seluruh kode proyek ke dalam container
COPY . .

# 5. Jalankan bot
CMD ["python", "main.py"]
