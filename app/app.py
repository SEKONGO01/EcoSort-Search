from flask import Flask, render_template, request, jsonify
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modele_eco_sort.h5")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

BIN_MAPPING = {
    "cardboard": {"nom": "Poubelle JAUNE", "couleur": "#F5C518"},
    "metal":     {"nom": "Poubelle JAUNE", "couleur": "#F5C518"},
    "plastic":   {"nom": "Poubelle JAUNE", "couleur": "#F5C518"},
    "glass":     {"nom": "Poubelle VERTE", "couleur": "#2E9E4F"},
    "paper":     {"nom": "Poubelle BLEUE", "couleur": "#2E7FD6"},
    "trash":     {"nom": "Poubelle MARRON / NOIRE", "couleur": "#4A3B32"},
}

D3E_KEYWORDS = [
    "smartphone", "telephone", "téléphone", "chargeur", "ecouteur", "écouteur",
    "mixeur", "montre", "batterie", "electronique", "électronique",
    "electrique", "électrique", "cuiseur", "machine", "appareil",
    "ordinateur", "tablette", "console", "camera", "caméra", "rasoir",
    "seche-cheveux", "sèche-cheveux", "ventilateur", "radio", "haut-parleur",
    "enceinte", "clavier", "souris", "imprimante", "led", "lampe",
    "usb", "bluetooth", "wifi", "rechargeable", "pile"
]

SUGGESTIONS = [
    "bouteille plastique", "bouteille en verre", "canette", "boite de conserve",
    "carton", "journal", "magazine", "cahier", "smartphone", "chargeur",
    "ecouteurs", "flacon shampoing", "pot de confiture", "sachet plastique",
    "montre connectee", "mixeur", "enveloppe", "brique de lait"
]


def detecte_d3e(nom_produit):
    nom_lower = nom_produit.lower()
    return any(mot in nom_lower for mot in D3E_KEYWORDS)


def predict_image_from_url(image_url):
    response = requests.get(image_url, timeout=10)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions[0])
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(predictions[0][predicted_index])
    return predicted_class, confidence


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/suggestions")
def suggestions():
    query = request.args.get("q", "").lower()
    if not query:
        return jsonify([])
    matches = [s for s in SUGGESTIONS if query in s.lower()][:5]
    return jsonify(matches)


@app.route("/api/search")
def api_search():
    keyword = request.args.get("keyword", "")
    if not keyword:
        return jsonify([])
    resultats = search_products(keyword, max_results=5)
    return jsonify(resultats)


@app.route("/api/predict")
def api_predict():
    image_url = request.args.get("image_url", "")
    nom_produit = request.args.get("nom", "")

    if not image_url:
        return jsonify({"error": "image_url manquant"}), 400

    if detecte_d3e(nom_produit):
        bin_info = {"nom": "Bac Électronique (D3E)", "couleur": "#808080", "categorie": "d3e"}
        confidence = None
    else:
        predicted_class, confidence = predict_image_from_url(image_url)
        bin_info = dict(BIN_MAPPING[predicted_class])
        bin_info["categorie"] = predicted_class

    return jsonify({
        "bin_nom": bin_info["nom"],
        "bin_couleur": bin_info["couleur"],
        "categorie": bin_info["categorie"],
        "confidence": confidence,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501, debug=True, use_reloader=False)
