"""
prepare_data.py
----------------
Script de preparation du dataset garbage-classification pour EcoSort-Search.

Fonctions :
    1. count_images_per_class() : compte le nombre d'images par classe
    2. show_sample_images()     : affiche 3 images d'exemple par classe
    3. split_dataset()          : repartit les images en train/val/test (70/15/15)

Usage :
    python data/prepare_data.py
(a lancer depuis la racine du projet, avec le dossier
 data/garbage-classification/ deja present)
"""

import os
import random
import shutil

import matplotlib.pyplot as plt
from matplotlib import image as mpimg

# Chemins et constantes
DATA_DIR = os.path.join("data", "garbage-classification")
OUTPUT_DIR = "data"
CLASSES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


def _list_images(class_dir):
    """Retourne la liste des chemins d'images valides dans un dossier de classe."""
    if not os.path.isdir(class_dir):
        return []
    return [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ]


def count_images_per_class():
    """Parcourt data/garbage-classification/ et affiche le nombre d'images par classe."""
    print("Comptage des images par classe :")
    print("-" * 35)

    total = 0
    for class_name in CLASSES:
        class_dir = os.path.join(DATA_DIR, class_name)
        images = _list_images(class_dir)
        count = len(images)
        total += count

        if not os.path.isdir(class_dir):
            print("  {:<10} : dossier introuvable ({})".format(class_name, class_dir))
        else:
            print("  {:<10} : {} images".format(class_name, count))

    print("-" * 35)
    print("  TOTAL      : {} images".format(total))


def show_sample_images(n_samples=3):
    """Affiche avec matplotlib n_samples images d'exemple pour chaque classe."""
    fig, axes = plt.subplots(len(CLASSES), n_samples, figsize=(3 * n_samples, 3 * len(CLASSES)))

    for row, class_name in enumerate(CLASSES):
        class_dir = os.path.join(DATA_DIR, class_name)
        images = _list_images(class_dir)
        sample = images[:n_samples]

        for col in range(n_samples):
            ax = axes[row, col] if len(CLASSES) > 1 else axes[col]
            ax.axis("off")

            if col < len(sample):
                img_path = os.path.join(class_dir, sample[col])
                img = mpimg.imread(img_path)
                ax.imshow(img)
                if col == 0:
                    ax.set_ylabel(class_name, fontsize=12)
            else:
                ax.set_visible(False)

        if len(sample) > 0:
            axes[row, 0].set_title(class_name, fontsize=12, loc="left")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_images.png"))
    plt.show()
    print("Apercu enregistre dans data/sample_images.png")


def split_dataset():
    """
    Repartit les images de data/garbage-classification/ en :
        - data/train/ (70%)
        - data/val/   (15%)
        - data/test/  (15%)
    chacun contenant les 6 sous-dossiers de classe.
    """
    random.seed(RANDOM_SEED)

    splits = ["train", "val", "test"]

    # Creation de l'arborescence de sortie
    for split in splits:
        for class_name in CLASSES:
            os.makedirs(os.path.join(OUTPUT_DIR, split, class_name), exist_ok=True)

    summary = {split: 0 for split in splits}

    for class_name in CLASSES:
        class_dir = os.path.join(DATA_DIR, class_name)
        images = _list_images(class_dir)

        if not images:
            print("Attention : aucune image trouvee pour la classe '{}'".format(class_name))
            continue

        random.shuffle(images)

        n_total = len(images)
        n_train = int(n_total * TRAIN_RATIO)
        n_val = int(n_total * VAL_RATIO)
        # Le reste va au test pour ne perdre aucune image par arrondi
        n_test = n_total - n_train - n_val

        train_files = images[:n_train]
        val_files = images[n_train:n_train + n_val]
        test_files = images[n_train + n_val:]

        for split, files in zip(splits, [train_files, val_files, test_files]):
            dest_dir = os.path.join(OUTPUT_DIR, split, class_name)
            for filename in files:
                src = os.path.join(class_dir, filename)
                dst = os.path.join(dest_dir, filename)
                shutil.copy2(src, dst)
            summary[split] += len(files)

        print(
            "{:<10} -> train: {}, val: {}, test: {}".format(
                class_name, len(train_files), len(val_files), len(test_files)
            )
        )

    print("-" * 35)
    print("Split termine : train={}, val={}, test={}".format(
        summary["train"], summary["val"], summary["test"]
    ))


if __name__ == "__main__":
    count_images_per_class()
    show_sample_images()
    split_dataset()
