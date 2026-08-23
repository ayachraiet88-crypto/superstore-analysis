# SmartStore AI — Analyse des ventes & prédiction de rentabilité (Superstore)

Projet complet d'analyse de données business, combinant EDA approfondie, statistiques inférentielles, Machine Learning (classification de rentabilité), segmentation client RFM, prévision de ventes (Prophet), et une application Streamlit multi-pages complète. Basé sur le dataset Superstore (Kaggle), l'un des datasets de référence en business analytics.

##  Objectif du projet

Fournir une analyse business de bout en bout — de l'exploration des données brutes jusqu'à un outil interactif d'aide à la décision — permettant d'identifier les facteurs de rentabilité, de segmenter la clientèle, et de prédire si une commande sera profitable avant même sa validation.

##  Dataset

- **Source** : Superstore Dataset (Kaggle)
- **9 994 commandes**, 21 colonnes (dates, catégories produits, région, remise, ventes, profit...)
- Mélange de types de données (dates, texte catégoriel, numérique) — idéal pour couvrir nettoyage, visualisation, stats et ML sur un même dataset

##  Stack technique

- **Langage** : Python
- **Analyse** : Pandas, NumPy, SciPy (tests statistiques)
- **Visualisation** : Matplotlib, Seaborn, Plotly
- **ML** : scikit-learn (LogisticRegression, RandomForestClassifier)
- **Prévision de séries temporelles** : Prophet (Meta/Facebook)
- **Déploiement** : Streamlit
- **Sérialisation** : joblib
- **Complément BI** : exports pour Power BI (dashboards Sales/Profit, Customer Intelligence RFM, Forecasting)

##  Pipeline du projet

1. **Nettoyage** : conversion des dates, vérification des doublons et incohérences logiques (ventes/quantités négatives, délais de livraison négatifs), détection des outliers par méthode IQR
2. **Feature engineering temporel** : extraction année/mois de commande, calcul du délai de livraison
3. **EDA approfondie** : distribution des ventes (échelle log), ventes/profit par catégorie, relation remise-profit, tendance mensuelle, matrice de corrélation
4. **Tests statistiques** : test t de Student comparant le profit entre faibles et fortes remises
5. **Classification de rentabilité** (`Is_Profitable`) : Logistic Regression vs Random Forest, avec matrices de confusion, courbes ROC/AUC et importance des variables
6. **Segmentation client RFM** (Recency, Frequency, Monetary) : 5 segments (Champions, Loyal Customers, Potential Loyalists, At Risk, Lost Customers)
7. **Export des données transformées** pour un dashboard Power BI complémentaire
8. **Application Streamlit** : dashboard exécutif, analyse produit, segmentation client, prévision Prophet, simulateur de rentabilité de commande

##  Résultats

### Test statistique — impact de la remise sur le profit

t-statistic : 15.82, p-value ≈ 0.00000 → différence de profit entre faibles et fortes remises **statistiquement très significative**.

### Classification de rentabilité (`Is_Profitable`)

| Modèle | Accuracy | AUC |
|---|---|---|
| Logistic Regression | 0.942 | 0.982 |
| **Random Forest** | 0.934 | **0.984** |

### Variable la plus importante (Random Forest)

**`Discount`** — de très loin la variable la plus déterminante pour prédire si une commande sera rentable, cohérent avec le test statistique et l'analyse de corrélation (Discount/Profit : -0.22).

### Segmentation RFM

| Segment | Nombre de clients |
|---|---|
| Loyal Customers | 222 |
| Champions | 193 |
| Potential Loyalists | 182 |
| At Risk | 141 |
| Lost Customers | 55 |

##  Application Streamlit (SmartStore AI)

5 pages :

1. **Dashboard** — KPIs exécutifs (ventes, profit, marge), ventes par région, profit par catégorie, évolution mensuelle, top produits
2. **Product Analysis** — relation ventes/profit par produit, top/bottom produits, impact de la remise
3. **Customer Segmentation** — visualisation des segments RFM, identification des clients à risque
4. **Sales Forecast** — prévision des ventes futures avec Prophet, intervalle de confiance, composantes tendance/saisonnalité
5. **Product Simulator** — simule la rentabilité d'une commande avant validation, analyse "what-if" de l'impact de la remise sur la probabilité de rentabilité

### Lancer l'application

```bash
git clone <repo-url>
cd smartstore-ai
pip install -r requirements.txt
streamlit run app.py
```

##  Structure du projet

```
smartstore-ai/
├── app.py                          # Application Streamlit
├── superstore_analysis.ipynb       # Notebook d'analyse complet
├── model_rf.pkl                    # Modèle Random Forest entraîné
├── scaler.pkl                      # StandardScaler fitté
├── model_columns.pkl               # Colonnes attendues par le modèle (alignement one-hot)
├── powerbi_exports/                # Fichiers exportés pour Power BI
│   ├── superstore_clean.csv
│   ├── customer_rfm.csv
│   ├── product_summary.csv
│   └── monthly_summary.csv
└── README.md
```

> Le dataset original (`Sample - Superstore.csv`) n'est pas inclus dans ce repo — disponible sur [Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final).

##  Points clés méthodologiques

- **Détection d'outliers par méthode IQR** : sans suppression automatique — les valeurs extrêmes ne sont écartées qu'après vérification qu'elles reflètent une vraie erreur, pas un gros client légitime
- **Test t de Student** : valide statistiquement (pas juste visuellement) que la remise impacte significativement le profit
- **Alignement des colonnes one-hot en production** (`reindex(columns=model_columns, fill_value=False)`) : garantit que les données saisies dans le simulateur correspondent exactement à la structure vue à l'entraînement, même si l'utilisateur sélectionne une combinaison de catégories différente de celle du jeu d'entraînement
- **Segmentation RFM** : approche marketing classique, complémentaire au Machine Learning, pour prioriser les actions de rétention client
- **Prophet pour la prévision** : modèle de séries temporelles gérant nativement la saisonnalité annuelle, avec intervalles de confiance visualisés
- **Double sortie du projet** : à la fois une application Streamlit interactive et des exports structurés pour un dashboard Power BI complémentaire

##  Auteur

Aya Chraiet
