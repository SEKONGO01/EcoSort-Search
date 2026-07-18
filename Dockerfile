FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY scraping/ ./scraping/
COPY models/modele_eco_sort.h5 ./models/modele_eco_sort.h5

EXPOSE 8501

CMD ["python", "app/app.py"]