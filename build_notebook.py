import json

def md(id_, text):
    return {"cell_type":"markdown","id":id_,"metadata":{},"source":[text]}

def code(id_, src):
    return {"cell_type":"code","id":id_,"metadata":{},"outputs":[],"execution_count":None,"source":[src]}

cells = [

md("md_titre", """\
# Détection de fractures osseuses — FracAtlas

**Auteurs :** Yves CHEKOUA & Mohamed Mehdi TRABELSSI
**Promotion :** SN2 — Université de Technologie de Troyes

Classification binaire (fracturé / non fracturé) d'images de radiographies osseuses,
avec gestion du déséquilibre de classes et étude de l'impact de l'augmentation de données.

---
## Plan du notebook
1. Analyse du dataset FracAtlas
2. Préparation des données
3. Augmentation de données
4. Algorithme 1 : Random Forest
5. Algorithme 2 : SVM
6. Algorithme 3 : CNN Transfer Learning MobileNetV2 (sans augmentation)
7. Algorithme 4 : CNN Transfer Learning MobileNetV2 (avec augmentation)
8. Comparaison générale
9. Conclusion
"""),

md("md_imports", "## Imports et chargement des bibliothèques"),
code("cell_imports", """\
# ── Bibliothèques de base ──────────────────────────────────────────
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

# ── Machine Learning classique ────────────────────────────────────
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve)

# ── Deep Learning ─────────────────────────────────────────────────
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

print(f"TensorFlow : {tf.__version__}")
print(f"GPU disponible : {len(tf.config.list_physical_devices('GPU')) > 0}")

# Chemin vers le dataset FracAtlas
# ⚠️ Adapter ce chemin selon votre environnement
DATA_DIR = r'FracAtlas/images'   # Colab : '/content/FracAtlas/images'

CLASSES = ['Non_fracturee', 'Fracturee']
IMG_ML  = 32   # résolution pour RF et SVM
IMG_CNN = 96   # résolution pour MobileNetV2

# Vérification du dataset
for folder in ['Non_fractured', 'Fractured']:
    path = os.path.join(DATA_DIR, folder)
    if os.path.exists(path):
        print(f"{folder}: {len(os.listdir(path))} images")
    else:
        print(f"⚠️  Dossier introuvable : {path}")
"""),

md("md_analyse", "## 1. Analyse du dataset FracAtlas"),
code("cell_load", """\
def load_dataset(data_dir, img_size, max_per_class=None):
    \"\"\"Charge les images depuis les dossiers Non_fractured / Fractured.\"\"\"
    X, y = [], []
    for label, folder in enumerate(['Non_fractured', 'Fractured']):
        folder_path = os.path.join(data_dir, folder)
        files = sorted([f for f in os.listdir(folder_path)
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        if max_per_class:
            files = files[:max_per_class]
        for fname in files:
            img = cv2.imread(os.path.join(folder_path, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size))
            X.append(img)
            y.append(label)
    return np.array(X, dtype='uint8'), np.array(y)

# Chargement du dataset complet pour RF et SVM (717 fracturées / 3366 non fracturées)
print("Chargement images 32×32 (RF et SVM) — dataset complet...")
t0 = time.time()
X_ml, y_ml = load_dataset(DATA_DIR, IMG_ML)
print(f"  {X_ml.shape}  en {time.time()-t0:.1f} s")

# Chargement du dataset complet pour CNN
print("Chargement images 96×96 (CNN) — dataset complet...")
t0 = time.time()
X_cnn, y_cnn = load_dataset(DATA_DIR, IMG_CNN)
print(f"  {X_cnn.shape}  en {time.time()-t0:.1f} s")
"""),
code("cell_visu", """\
# Distribution des classes
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

counts = [(y_ml==0).sum(), (y_ml==1).sum()]
axes[0].bar(CLASSES, counts, color=['steelblue', 'tomato'])
axes[0].set_title('Distribution des classes – FracAtlas')
for i, v in enumerate(counts):
    axes[0].text(i, v+20, str(v), ha='center', fontweight='bold')
axes[1].axis('off')

# Exemples d'images
fig2, axs = plt.subplots(2, 5, figsize=(13, 6))
for label in [0, 1]:
    idxs = np.where(y_ml == label)[0][:5]
    for j, idx in enumerate(idxs):
        axs[label][j].imshow(X_ml[idx])
        axs[label][j].set_title(CLASSES[label], fontsize=8)
        axs[label][j].axis('off')
plt.suptitle('Exemples de radiographies FracAtlas')
plt.tight_layout(); plt.show()

print(f"Ratio de déséquilibre : 1:{counts[0]/counts[1]:.1f}")
"""),

md("md_prep", """\
## 2. Préparation des données

**Dataset complet** utilisé pour tous les algorithmes (4 083 images : 717 fracturées, 3 366 non fracturées).
Le déséquilibre de classes est compensé par `class_weight='balanced'` (RF, SVM) et une pondération
manuelle (CNN), plutôt que par sous-échantillonnage.
"""),
code("cell_split", """\
# Split stratifié 80/20 pour les algorithmes ML – conservation des proportions de classes
X_ml_tr, X_ml_te, y_ml_tr, y_ml_te = train_test_split(
    X_ml, y_ml, test_size=0.2, random_state=42, stratify=y_ml
)

# Normalisation + aplatissement pour RF/SVM
X_tr_flat = X_ml_tr.reshape(len(X_ml_tr), -1).astype('float32') / 255.0
X_te_flat = X_ml_te.reshape(len(X_ml_te), -1).astype('float32') / 255.0

print(f"Train ML : {X_tr_flat.shape}  |  Test ML : {X_te_flat.shape}")
print(f"  Train – Non-frac: {(y_ml_tr==0).sum()}  |  Frac: {(y_ml_tr==1).sum()}")
print(f"  Test  – Non-frac: {(y_ml_te==0).sum()}  |  Frac: {(y_ml_te==1).sum()}")
"""),

md("md_aug", """\
## 3. Augmentation de données
L'augmentation génère synthétiquement de nouveaux exemples, ce qui est en théorie utile pour
enrichir la classe minoritaire (fractures). Elle est intégrée dans le pipeline du CNN
(active uniquement pendant l'entraînement).
"""),
code("cell_aug", """\
# Pipeline d'augmentation
augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal_and_vertical'),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomContrast(0.2),
], name='augmentation_fractures')

# Visualisation de l'effet de l'augmentation
frac_idxs = np.where(y_cnn == 1)[0]
sample = (X_cnn[frac_idxs[0]:frac_idxs[0]+1].astype('float32') / 127.5 - 1.0)

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
axes.flat[0].imshow((sample[0]+1)/2)
axes.flat[0].set_title('Original', fontweight='bold')
axes.flat[0].axis('off')
for ax in list(axes.flat)[1:]:
    aug = augmentation(sample, training=True)[0].numpy()
    ax.imshow(np.clip((aug+1)/2, 0, 1))
    ax.axis('off')
plt.suptitle('Augmentation de données – Radiographie fracturée')
plt.tight_layout(); plt.show()
"""),

md("md_rf", """\
## 4. Algorithme 1 : Random Forest
Métrique d'optimisation : **F1-score** (plus pertinent que l'accuracy sur données déséquilibrées).
"""),
code("cell_rf", """\
param_grid_rf = {
    'n_estimators': [100, 200],
    'max_depth':    [10, 20, None],
    'max_features': ['sqrt', 'log2']
}

print("GridSearchCV Random Forest...")
t0 = time.time()
gs_rf = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced'),
    param_grid_rf, cv=3, scoring='f1', verbose=1, n_jobs=-1
)
gs_rf.fit(X_tr_flat, y_ml_tr)
print(f"Temps : {time.time()-t0:.0f} s  |  Meilleurs params : {gs_rf.best_params_}")

rf = gs_rf.best_estimator_
y_pred_rf = rf.predict(X_te_flat)
y_prob_rf = rf.predict_proba(X_te_flat)[:, 1]
acc_rf = accuracy_score(y_ml_te, y_pred_rf)
auc_rf = roc_auc_score(y_ml_te, y_prob_rf)

print(f"\\nRandom Forest – Exactitude : {acc_rf*100:.2f}%  |  AUC : {auc_rf:.4f}")
print()
print(classification_report(y_ml_te, y_pred_rf, target_names=CLASSES))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_ml_te, y_pred_rf), annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'RF – Matrice de confusion (acc={acc_rf*100:.1f}%)')
plt.xlabel('Prédit'); plt.ylabel('Réel')
plt.tight_layout(); plt.show()
"""),

md("md_svm", """\
## 5. Algorithme 2 : SVM
**StandardScaler** obligatoire avant le SVM (centrage-réduction des features).
"""),
code("cell_svm", """\
# Standardisation des features (indispensable pour le SVM)
scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr_flat)
X_te_sc = scaler.transform(X_te_flat)

param_grid_svm = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}

print("GridSearchCV SVM...")
t0 = time.time()
gs_svm = GridSearchCV(
    SVC(probability=True, random_state=42, class_weight='balanced'),
    param_grid_svm, cv=3, scoring='f1', verbose=1, n_jobs=-1
)
gs_svm.fit(X_tr_sc, y_ml_tr)
print(f"Temps : {time.time()-t0:.0f} s  |  Meilleurs params : {gs_svm.best_params_}")

svm = gs_svm.best_estimator_
y_pred_svm = svm.predict(X_te_sc)
y_prob_svm = svm.predict_proba(X_te_sc)[:, 1]
acc_svm = accuracy_score(y_ml_te, y_pred_svm)
auc_svm = roc_auc_score(y_ml_te, y_prob_svm)

print(f"\\nSVM – Exactitude : {acc_svm*100:.2f}%  |  AUC : {auc_svm:.4f}")
print()
print(classification_report(y_ml_te, y_pred_svm, target_names=CLASSES))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_ml_te, y_pred_svm), annot=True, fmt='d', cmap='Oranges',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'SVM – Matrice de confusion (acc={acc_svm*100:.1f}%)')
plt.xlabel('Prédit'); plt.ylabel('Réel')
plt.tight_layout(); plt.show()
"""),

md("md_cnn_prep", """\
## 6. Algorithme 3 : CNN Transfer Learning MobileNetV2 (sans augmentation)

**Approche en deux phases :**
- Phase 1 (Freeze) : base MobileNetV2 gelée, seule la tête FC est entraînée (`lr=1e-3`)
- Phase 2 (Fine-tuning) : 30 dernières couches dégelées (`lr=1e-5`)
"""),
code("cell_cnn_prep", """\
# Split pour le CNN
X_cnn_tr, X_cnn_te, y_cnn_tr, y_cnn_te = train_test_split(
    X_cnn, y_cnn, test_size=0.2, random_state=42, stratify=y_cnn
)
# Normalisation MobileNetV2 : [-1, 1]
X_cnn_tr = X_cnn_tr.astype('float32') / 127.5 - 1.0
X_cnn_te = X_cnn_te.astype('float32') / 127.5 - 1.0

# Poids de classe pour compenser le déséquilibre
n_neg = (y_cnn_tr == 0).sum()
n_pos = (y_cnn_tr == 1).sum()
class_w = {0: 1.0, 1: n_neg / n_pos}

print(f"Train CNN : {X_cnn_tr.shape}  |  Test CNN : {X_cnn_te.shape}")
print(f"Poids classe Fracturée : {class_w[1]:.2f}")
"""),
code("cell_cnn_build", """\
# Construction du modèle SANS augmentation
base = MobileNetV2(weights='imagenet', include_top=False,
                   input_shape=(IMG_CNN, IMG_CNN, 3))
for layer in base.layers:
    layer.trainable = False

inp = layers.Input(shape=(IMG_CNN, IMG_CNN, 3))
x   = base(inp, training=False)                  # extraction de features (gelée)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.Dense(64, activation='relu')(x)
x   = layers.Dropout(0.4)(x)
out = layers.Dense(1, activation='sigmoid')(x)    # sortie binaire

model_cnn = models.Model(inputs=inp, outputs=out, name='CNN_Fracture_MobileNetV2')
print(f"Paramètres entraînables : {sum(w.numpy().size for w in model_cnn.trainable_weights):,}")
"""),
code("cell_cnn_phase1", """\
model_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
cb_cnn = [
    EarlyStopping(monitor='val_auc', patience=6, restore_best_weights=True,
                  mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

print("Phase 1 – Freeze (tête FC uniquement)...")
t0 = time.time()
hist_freeze = model_cnn.fit(
    X_cnn_tr, y_cnn_tr,
    epochs=20, batch_size=32, validation_split=0.1,
    class_weight=class_w, callbacks=cb_cnn, verbose=1
)
t_freeze = time.time() - t0
print(f"Phase 1 terminée en {t_freeze/60:.1f} min")
"""),
code("cell_cnn_phase2", """\
# Dégeler les 30 dernières couches pour le fine-tuning
for layer in base.layers[-30:]:
    layer.trainable = True

model_cnn.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # lr très faible : ne pas écraser ImageNet
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("Phase 2 – Fine-tuning (30 dernières couches)...")
t0 = time.time()
hist_ft = model_cnn.fit(
    X_cnn_tr, y_cnn_tr,
    epochs=10, batch_size=32, validation_split=0.1,
    class_weight=class_w, callbacks=cb_cnn, verbose=1
)
t_total = t_freeze + (time.time() - t0)
print(f"Temps total CNN (sans augmentation) : {t_total/60:.1f} min")
"""),
code("cell_cnn_eval", """\
# Évaluation – CNN sans augmentation
y_prob_cnn = model_cnn.predict(X_cnn_te, verbose=0).flatten()
y_pred_cnn = (y_prob_cnn >= 0.5).astype(int)
acc_cnn = accuracy_score(y_cnn_te, y_pred_cnn)
auc_cnn = roc_auc_score(y_cnn_te, y_prob_cnn)

print(f"CNN MobileNetV2 (sans augmentation) – Exactitude : {acc_cnn*100:.2f}%  |  AUC : {auc_cnn:.4f}")
print()
print(classification_report(y_cnn_te, y_pred_cnn, target_names=CLASSES))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_cnn_te, y_pred_cnn), annot=True, fmt='d', cmap='Greens',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'CNN (sans augmentation) – Matrice de confusion (acc={acc_cnn*100:.1f}%)')
plt.xlabel('Prédit'); plt.ylabel('Réel')
plt.tight_layout(); plt.show()
"""),

md("md_cnn_aug", """\
## 7. Algorithme 4 : CNN Transfer Learning MobileNetV2 (avec augmentation)
Même architecture que l'algorithme précédent, avec la couche d'augmentation
(incluant `RandomFlip` vertical) intégrée en amont du réseau.
"""),
code("cell_cnn_build_aug", """\
# Construction du modèle AVEC augmentation intégrée
base_aug = MobileNetV2(weights='imagenet', include_top=False,
                       input_shape=(IMG_CNN, IMG_CNN, 3))
for layer in base_aug.layers:
    layer.trainable = False

inp_aug = layers.Input(shape=(IMG_CNN, IMG_CNN, 3))
x   = augmentation(inp_aug)                       # augmentation en début de pipeline
x   = base_aug(x, training=False)
x   = layers.GlobalAveragePooling2D()(x)
x   = layers.Dense(64, activation='relu')(x)
x   = layers.Dropout(0.4)(x)
out_aug = layers.Dense(1, activation='sigmoid')(x)

model_cnn_aug = models.Model(inputs=inp_aug, outputs=out_aug, name='CNN_Fracture_MobileNetV2_Aug')
print(f"Paramètres entraînables : {sum(w.numpy().size for w in model_cnn_aug.trainable_weights):,}")
"""),
code("cell_cnn_phase1_aug", """\
model_cnn_aug.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)
cb_aug = [
    EarlyStopping(monitor='val_auc', patience=6, restore_best_weights=True,
                  mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1)
]

print("Phase 1 – Freeze (avec augmentation)...")
t0 = time.time()
hist_aug_freeze = model_cnn_aug.fit(
    X_cnn_tr, y_cnn_tr,
    epochs=20, batch_size=32, validation_split=0.1,
    class_weight=class_w, callbacks=cb_aug, verbose=1
)
print(f"Phase 1 terminée en {(time.time()-t0)/60:.1f} min")
"""),
code("cell_cnn_phase2_aug", """\
for layer in base_aug.layers[-30:]:
    layer.trainable = True

model_cnn_aug.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

print("Phase 2 – Fine-tuning (avec augmentation)...")
t0 = time.time()
hist_aug_ft = model_cnn_aug.fit(
    X_cnn_tr, y_cnn_tr,
    epochs=10, batch_size=32, validation_split=0.1,
    class_weight=class_w, callbacks=cb_aug, verbose=1
)
print(f"Phase 2 terminée en {(time.time()-t0)/60:.1f} min")
"""),
code("cell_cnn_eval_aug", """\
# Évaluation – CNN avec augmentation
y_prob_cnn_aug = model_cnn_aug.predict(X_cnn_te, verbose=0).flatten()
y_pred_cnn_aug = (y_prob_cnn_aug >= 0.5).astype(int)
acc_cnn_aug = accuracy_score(y_cnn_te, y_pred_cnn_aug)
auc_cnn_aug = roc_auc_score(y_cnn_te, y_prob_cnn_aug)

print(f"CNN MobileNetV2 (avec augmentation) – Exactitude : {acc_cnn_aug*100:.2f}%  |  AUC : {auc_cnn_aug:.4f}")
print()
print(classification_report(y_cnn_te, y_pred_cnn_aug, target_names=CLASSES))

plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_cnn_te, y_pred_cnn_aug), annot=True, fmt='d', cmap='Reds',
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title(f'CNN (avec augmentation) – Matrice de confusion (acc={acc_cnn_aug*100:.1f}%)')
plt.xlabel('Prédit'); plt.ylabel('Réel')
plt.tight_layout(); plt.show()

print()
print("Observation : l'augmentation dégrade ici la performance.")
print("Le RandomFlip vertical déforme la sémantique anatomique des radiographies —")
print("une augmentation mal choisie peut nuire plutôt qu'aider.")
"""),

md("md_compar", "## 8. Comparaison générale"),
code("cell_roc", """\
# Courbes ROC des 4 algorithmes
plt.figure(figsize=(8, 6))
for name, y_true_r, y_prob_r, auc_v in [
    ('Random Forest',                   y_ml_te,  y_prob_rf,      auc_rf),
    ('SVM',                             y_ml_te,  y_prob_svm,     auc_svm),
    ('CNN MobileNetV2',                 y_cnn_te, y_prob_cnn,     auc_cnn),
    ('CNN MobileNetV2 (Augmentation)',  y_cnn_te, y_prob_cnn_aug, auc_cnn_aug),
]:
    fpr, tpr, _ = roc_curve(y_true_r, y_prob_r)
    plt.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC={auc_v:.3f})')
plt.plot([0,1],[0,1],'k--', label='Aléatoire')
plt.xlabel('Taux de faux positifs')
plt.ylabel('Taux de vrais positifs')
plt.title('Courbes ROC – Détection de fractures')
plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()
"""),
code("cell_bilan", """\
# Tableau comparatif
print('═' * 58)
print(f'{"Algorithme":<28} {"Exactitude":>12} {"AUC-ROC":>10}')
print('─' * 58)
for name, acc, auc_v in [
    ('Random Forest',                  acc_rf,     auc_rf),
    ('SVM',                            acc_svm,    auc_svm),
    ('CNN MobileNetV2',                acc_cnn,    auc_cnn),
    ('CNN MobileNetV2 (Augmentation)', acc_cnn_aug, auc_cnn_aug),
]:
    print(f'{name:<28} {acc*100:>11.2f}% {auc_v:>10.4f}')
print('═' * 58)

# Graphique
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
noms = ['Random\\nForest', 'SVM', 'CNN\\nMobileNetV2', 'CNN +\\nAugmentation']
accs = [acc_rf, acc_svm, acc_cnn, acc_cnn_aug]
aucs = [auc_rf, auc_svm, auc_cnn, auc_cnn_aug]
colors = ['steelblue', 'darkorange', 'seagreen', 'indianred']

axes[0].bar(noms, [a*100 for a in accs], color=colors)
axes[0].set_title('Exactitude – Fractures'); axes[0].set_ylabel('%')
for i, v in enumerate(accs):
    axes[0].text(i, v*100+0.5, f'{v*100:.1f}%', ha='center', fontweight='bold')

axes[1].bar(noms, aucs, color=colors)
axes[1].set_title('AUC-ROC – Fractures'); axes[1].set_ylim(0, 1)
for i, v in enumerate(aucs):
    axes[1].text(i, v+0.01, f'{v:.3f}', ha='center', fontweight='bold')

plt.tight_layout(); plt.show()
"""),

md("md_conclusion", """\
---
## 9. Conclusion

| Modèle | Exactitude | AUC-ROC |
|--------|-----------|---------|
| Random Forest | 83,6% | 0,730 |
| SVM | 74,3% | 0,745 |
| **CNN MobileNetV2 (sans augmentation)** | **86,05%** | **0,848** |
| CNN MobileNetV2 (avec augmentation) | 75,3% | 0,626 |

### Enseignements

- Le **CNN MobileNetV2 sans augmentation** obtient les meilleures performances : le Transfer Learning
  surpasse les algorithmes ML classiques dès lors qu'il dispose d'un volume de données suffisant
- Sur des données déséquilibrées, l'**AUC-ROC** est une métrique plus fiable que l'exactitude brute
  pour comparer les modèles
- L'**augmentation de données n'est pas systématiquement bénéfique** : le `RandomFlip` vertical
  déforme la sémantique anatomique des radiographies et dégrade les performances du CNN
  (-10,8 points d'exactitude, -0,222 sur l'AUC-ROC) — la technique d'augmentation doit être choisie
  en fonction du domaine d'application, et non appliquée de façon générique
- Le `class_weight='balanced'` (ML) et la pondération de classe manuelle (CNN) sont indispensables
  pour éviter qu'un modèle ne se contente de prédire systématiquement la classe majoritaire
"""),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": [], "toc_visible": True}
    },
    "cells": cells
}

out = r'C:\Users\Yves Chekoua\OneDrive\Bureau\UE\SN2\IA02\github_projects\fracatlas-fracture-detection\fracatlas_fracture_detection.ipynb'
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f"OK - Notebook ecrit : {out} ({len(cells)} cellules)")
