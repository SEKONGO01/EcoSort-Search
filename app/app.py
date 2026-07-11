from flask import Flask, render_template, request
import sys
import os
import requests
from io import BytesIO
from PIL import Image
import numpy as np
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scraping")))
from jumia_scraper import search_products

app = Flask(__name__)

# Chargement du modèle une seule fois, au démarrage du serveur
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modele_eco_sort.h5")
model = tf.keras.models.load_model(MODEL_PATH)

# Doit correspondre EXACTEMENT à l'ordre alphabétique utilisé lors de l'entraînement
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Mapping classe du modèle -> poubelle officielle du projet
BIN_MAPPING = {
    "cardboard": {"nom": "Poubelle JAUNE", "couleur": "#FFD700"},
    "metal":     {"nom": "Poubelle JAUNE", "couleur": "#FFD700"},
    "plastic":   {"nom": "Poubelle JAUNE", "couleur": "#FFD700"},
    "glass":     {"nom": "Poubelle VERTE", "couleur": "#228B22"},
    "paper":     {"nom": "Poubelle BLEUE", "couleur": "#1E90FF"},
    "trash":     {"nom": "Poubelle MARRON / NOIRE", "couleur": "#3B2F2F"},
}

# Mots-clés déclenchant le Bac D3E, en complément du modèle
D3E_KEYWORDS = ["smartphone", "telephone", "téléphone", "chargeur", "ecouteur",
                "écouteur", "mixeur", "montre", "batterie", "electronique", "électronique"]


def detecte_d3e(nom_produit):
    nom_lower = nom_produit.lower()
    return any(mot in nom_lower for mot in D3E_KEYWORDS)


def predict_image_from_url(image_url):
    response = requests.get(image_url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((128, 128))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # ajoute la dimension "batch"

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])

    return predicted_class, confidence


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["GET"])
def search():
    keyword = request.args.get("keyword", "")
    if not keyword:
        return render_template("index.html")

    resultats = search_products(keyword, max_results=5)
    return render_template("results.html", produits=resultats, keyword=keyword)


@app.route("/predict", methods=["GET"])
def predict():
    image_url = request.args.get("image_url", "")
    nom_produit = request.args.get("nom", "")

    if not image_url:
        return render_template("index.html")

    if detecte_d3e(nom_produit):
        bin_info = {"nom": "Bac Électronique (D3E)", "couleur": "#808080"}
        confidence = None
    else:
        predicted_class, confidence = predict_image_from_url(image_url)
        bin_info = BIN_MAPPING[predicted_class]

    return render_template(
        "predict.html",
        nom_produit=nom_produit,
        image_url=image_url,
        bin_nom=bin_info["nom"],
        bin_couleur=bin_info["couleur"],
        confidence=confidence,
    )


if __name__ == "__main__":
    app.run(debug=True)