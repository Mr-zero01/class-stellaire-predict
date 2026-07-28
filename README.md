# 🌌 Predicting Stellar Class

Classification automatique d'objets célestes (**Galaxy**, **QSO**, **Star**) à partir de leurs caractéristiques photométriques et astrométriques, dans le cadre de la compétition [Kaggle Playground Series S6E6](https://www.kaggle.com/competitions/playground-series-s6e6).

**Accuracy finale : 0.96** (jeu de test) — 0.97 en entraînement, sans surapprentissage significatif.

---

## 📌 Contexte

Les relevés astronomiques modernes (SDSS, Gaia...) génèrent des millions d'observations. Automatiser la classification des objets célestes permet de traiter ce volume à grande échelle. Ce projet propose un pipeline complet, du prétraitement des données jusqu'au déploiement d'une interface utilisable.

## 🎯 Objectif

Prédire la classe d'un objet céleste (Galaxy, QSO, Star) à partir de mesures telles que les magnitudes photométriques (u, g, r, i, z), le redshift et les coordonnées célestes (alpha, delta).

## 🔍 Démarche

1. **Analyse exploratoire** — étude des relations entre variables via des tests statistiques (corrélation de Spearman, tests d'association) pour identifier les features les plus discriminantes.
2. **Prétraitement** — nettoyage, gestion des valeurs aberrantes, encodage des variables catégorielles. Étape la plus exigeante du projet.
3. **Modélisation** — comparaison de 5 algorithmes de classification :
   - Decision Tree
   - Random Forest
   - AdaBoost
   - XGBoost
   - LightGBM
4. **Évaluation** — matrices de confusion et classification report (precision, recall, F1-score) sur les jeux d'entraînement et de test.
5. **Déploiement** — interface web permettant d'utiliser le modèle facilement, y compris pour un public non-technique.

## 📊 Résultats

| Jeu | Accuracy |
|---|---|
| Entraînement | 0.97 |
| Test | 0.96 |

Le **redshift** est de loin la variable la plus discriminante, suivi des bandes photométriques **z** et **i**. Les coordonnées célestes (alpha, delta) apportent peu d'information au modèle.

La classe **Star** reste la plus délicate à prédire avec précision (precision 0.88 en test), mais avec un excellent recall (0.96). Les classes **Galaxy** et **QSO** sont très bien discriminées (F1-score ≥ 0.96).

## 🛠️ Stack technique

- **Langage** : Python
- **Manipulation de données** : pandas, numpy
- **Visualisation** : matplotlib, seaborn
- **Machine Learning** : scikit-learn, XGBoost, LightGBM
- **Interface web** : *Streamlit

## 🚀 Installation

```bash
git clone https://github.com/Mr-zero01/class-stellaire-predict.git
cd [ton-repo]
pip install -r requirements.txt
```

## ▶️ Utilisation

```bash
# Lancer le notebook d'analyse et d'entraînement
jupyter notebook classe-stelaire.ipynb

# Lancer l'interface web
[commande de lancement : streamlit run exploi_model.py]
```

## 📎 Source des données

[Kaggle Playground Series — Season 6, Episode 6](https://www.kaggle.com/competitions/playground-series-s6e6)
