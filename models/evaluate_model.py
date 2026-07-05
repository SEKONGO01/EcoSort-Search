import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

model = tf.keras.models.load_model("models/modele_eco_sort.h5")

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    "data/split/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False  # important : on garde l'ordre pour comparer aux vraies étiquettes
)

loss, accuracy = model.evaluate(test_generator)
print(f"\nAccuracy finale sur le jeu de TEST : {accuracy*100:.2f}%")

# Prédictions détaillées
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes
class_names = list(test_generator.class_indices.keys())

print("\n--- Rapport de classification ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matrice de confusion
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap="Blues")
plt.title("Matrice de confusion")
plt.colorbar()
plt.xticks(range(len(class_names)), class_names, rotation=45)
plt.yticks(range(len(class_names)), class_names)
plt.xlabel("Prédiction")
plt.ylabel("Vraie classe")
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.savefig("models/confusion_matrix.png")
print("\nMatrice de confusion sauvegardée dans models/confusion_matrix.png")