"""
Scraper Jumia CI - Recherche de produits
Projet : EcoSort-Search - branche feature/scraping
"""

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.jumia.ci"
HEADERS = {"User-Agent": "Mozilla/5.0"}


def search_products(keyword, max_results=5):
    """
    Recherche des produits sur Jumia CI à partir d'un mot-clé.

    Retourne une liste de dictionnaires :
    [{"nom": "...", "prix": "...", "image_url": "...", "lien": "..."}]
    """
    url = f"{BASE_URL}/catalog/?q={keyword}"

    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        print(f"[ERREUR] Jumia a répondu avec une erreur HTTP : {e}")
        print("-> Si c'est un 403, arrête-toi et préviens l'équipe.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[ERREUR] Problème de connexion : {e}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")

    # Chaque produit est dans une balise <article class="prd ...">
    articles = soup.find_all("article", class_="prd")

    if not articles:
        print("[ATTENTION] Aucun produit trouvé (page vide ou structure HTML différente).")
        print("-> Vérifie manuellement la page dans le navigateur avant de continuer.")

    produits = []

    for article in articles:
        if len(produits) >= max_results:
            break

        # Nom du produit
        nom_tag = article.find("h3", class_="name")
        nom = nom_tag.get_text(strip=True) if nom_tag else None

        # Prix
        prix_tag = article.find("div", class_="prc")
        prix = prix_tag.get_text(strip=True) if prix_tag else None

        # Image (souvent chargée en lazy-load -> data-src, sinon src)
        img_tag = article.find("img")
        image_url = None
        if img_tag:
            image_url = img_tag.get("data-src") or img_tag.get("src")

        # Lien vers la fiche produit
        lien_tag = article.find("a", class_="core") or article.find("a")
        lien = None
        if lien_tag and lien_tag.get("href"):
            href = lien_tag["href"]
            lien = href if href.startswith("http") else BASE_URL + href

        # On ne garde que les produits où on a au moins le nom et le prix
        if nom and prix:
            produits.append({
                "nom": nom,
                "prix": prix,
                "image_url": image_url,
                "lien": lien,
            })

    return produits


if __name__ == "__main__":
    resultats = search_products("bouteille")
    for r in resultats:
        print(r)