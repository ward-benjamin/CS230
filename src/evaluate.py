import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, dev):
    # Extraire toutes les features et cibles d'un coup
    print(dev)
    dev_features, dev_targets = zip(*list(dev)) 
    print(sum(1 for _ in dev)) # Crée une liste de tuples et sépare les features et targets
    dev_features = np.array(dev_features)  # Convertir en numpy array
    dev_targets = np.array(dev_targets)  # Convertir en numpy array

    print(np.all(dev_targets == 1))

    # Générer les prédictions
    outputs = model(dev_features, training=False)
    predicted = tf.cast(outputs > 0.5, dtype=tf.float32)

    accuracy = accuracy_score(dev_targets, predicted.numpy())
    f1 = f1_score(dev_targets, predicted.numpy())
    cm = confusion_matrix(dev_targets, predicted.numpy())

    # Vérifier que la matrice de confusion a bien deux classes
    if cm.shape != (2, 2):
        # Si la matrice de confusion n'a pas deux classes, ajuster
        cm = np.array([[cm[0, 0], cm[0, 1]], [cm[1, 0], cm[1, 1]]])
        print("Ajustement de la matrice de confusion, car une seule classe a été prédite.")

    # Afficher les résultats
    print(f'Accuracy: {accuracy:.4f}')
    print(f'F1 Score: {f1:.4f}')

    # Afficher la matrice de confusion
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', cbar=False, xticklabels=['0', '1'], yticklabels=['0', '1'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'false_positive': cm[0, 1] if cm.shape == (2, 2) else 0,
        'false_negative': cm[1, 0] if cm.shape == (2, 2) else 0
    }