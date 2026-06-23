# Détection de fractures osseuses — FracAtlas

Projet réalisé dans le cadre du cours **IA02 – Intelligence Artificielle** (Université de Technologie de Troyes, promotion SN2).

Classification binaire (fracturé / non fracturé) d'images de radiographies osseuses, avec gestion du **déséquilibre de classes** et étude de l'impact de l'**augmentation de données**.

## Auteurs

- Yves CHEKOUA
- Mohamed Mehdi TRABELSSI

## Le dataset

[FracAtlas](https://www.kaggle.com/datasets/akshayramakrishnan28/fracture-classification-dataset) contient **4 083 images** de radiographies osseuses, dont **717 fracturées (17,6 %)** et **3 366 non fracturées** — un déséquilibre de classes de l'ordre de **1 pour 4,7**, représentatif des situations médicales réelles où les cas pathologiques sont minoritaires.

![Répartition des classes](images/repartition_donnee.png)

![Exemples de radiographies](images/exemple.png)

## Algorithmes comparés

| Modèle | Exactitude (test) | AUC-ROC |
|---|---|---|
| Random Forest | 83,6 % | 0,730 |
| SVM | 74,3 % | 0,745 |
| **CNN MobileNetV2 (sans augmentation)** | **86,05 %** | **0,848** |
| CNN MobileNetV2 (avec augmentation) | 75,3 % | 0,626 |

> Sur ce contexte médical à classes déséquilibrées, l'**AUC-ROC** est la métrique de référence : elle reflète la capacité du modèle à séparer les deux classes indépendamment du seuil de décision, contrairement à l'exactitude brute qui peut être trompeuse.
>
> Fait notable : l'augmentation de données **dégrade** ici les performances du CNN (-10,8 points d'exactitude). Voir l'analyse dans le [RAPPORT.md](RAPPORT.md).

Détails complets, méthodologie et analyse : voir [RAPPORT.md](RAPPORT.md).

## Structure du dépôt

```
fracatlas-fracture-detection/
├── fracatlas_fracture_detection.ipynb   # Notebook complet (RF, SVM, CNN ± augmentation)
├── images/                              # Visualisations (matrices, courbes ROC...)
├── README.md
└── RAPPORT.md                           # Rapport détaillé (méthodologie, résultats, analyse)
```

## Exécution

Le notebook est conçu pour être exécuté sur **Google Colab**.

1. Télécharger le dataset [FracAtlas sur Kaggle](https://www.kaggle.com/datasets/akshayramakrishnan28/fracture-classification-dataset) et l'organiser comme suit :
   ```
   FracAtlas/images/
   ├── Fractured/
   └── Non_fractured/
   ```
2. Téléverser ce dossier sur Google Drive et monter le Drive dans Colab, ou téléverser directement dans l'environnement Colab
3. Adapter la variable `DATA_DIR` dans la cellule d'import (ex : `'/content/drive/MyDrive/FracAtlas/images'`)
4. Exécuter les cellules dans l'ordre

### Dépendances

```bash
pip install tensorflow scikit-learn numpy matplotlib seaborn opencv-python
```

## Principaux enseignements

- Sur données déséquilibrées, l'**AUC-ROC** est plus informative que l'exactitude pour comparer des modèles
- Le **Transfer Learning** (CNN MobileNetV2 sans augmentation) est le modèle le plus performant sur ce dataset
- L'**augmentation de données n'est pas systématiquement bénéfique** : un `RandomFlip` vertical déforme la sémantique anatomique des radiographies et dégrade les performances du CNN — la technique d'augmentation doit être choisie en fonction du domaine
- Le `class_weight='balanced'` (ML) et la pondération de classe manuelle (CNN) sont indispensables pour éviter qu'un modèle ne se contente de prédire systématiquement la classe majoritaire

## Licence

Projet académique — UTT, IA02, 2026.
