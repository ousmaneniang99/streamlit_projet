#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
#from ydata_profiling import ProfileReport
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib


# In[2]:


data = pd.read_csv('Classeur1.csv')


# In[3]:


data.head()


# In[4]:


data.columns


# In[5]:


data.info()


# In[ ]:





# In[6]:


data.describe().T


# In[7]:


#copions le dataset dans df
df=data.copy()


# In[8]:


# Vérifier les valeurs manquantes
print(df.isnull().sum())


# In[9]:


# Vérifier les doublons
print(df.duplicated().sum())

# Supprimer les doublons
df = df.drop_duplicates()


# **Gestion des valeurs manquantes:**
# 
# #### Impute les valeurs manquantes:
# - Remplace les valeurs manquantes dans REGION par 'UNKNOWN'.
# - Remplace les valeurs manquantes dans toutes les colonnes numériques par 0, sauf pour MONTANT et REVENUE.
# - Encode les variables catégorielles:
# - Convertit les catégories de TENURE, TOP_PACK et MRG en valeurs numériques.
# - Impute les valeurs manquantes par la médiane dans MONTANT et REVENUE
# #### Supprime les colonnes ZONE1 et ZONE2.
# 

# In[10]:


# Remplacer les valeurs manquantes dans `REGION` par 'UNKNOWN'.
df['REGION'] = df['REGION'].fillna('UNKNOWN')

# Remplacer les valeurs manquantes dans les colonnes numériques par 0, sauf pour `MONTANT` et `REVENUE`.
df[['FREQUENCE_RECH','ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'FREQ_TOP_PACK']] = df[['FREQUENCE_RECH', 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO', 'FREQ_TOP_PACK']].fillna(0)

# Encoder les colonnes catégorielles.
le = LabelEncoder()
df['TENURE'] = le.fit_transform(df['TENURE'])
df['TOP_PACK'] = le.fit_transform(df['TOP_PACK'])
df['MRG'] = le.fit_transform(df['MRG'])

# Calculer la médiane de `MONTANT` et `REVENUE`.
median_montant = df['MONTANT'].median()
median_revenue = df['REVENUE'].median()

# Remplacer les valeurs 0 dans `MONTANT` et `REVENUE` par leurs médianes respectives.
df['MONTANT'] = df['MONTANT'].fillna(median_montant)
df['REVENUE'] = df['REVENUE'].fillna(median_revenue)

# Supprimer les colonnes `ZONE1` et `ZONE2`
df.drop(columns=['ZONE1', 'ZONE2'], inplace=True)

# Afficher les cinq premières lignes du DataFrame après l'imputation
# Afficher les 5 premières lignes de X_train




# In[ ]:





# In[11]:


# Identifier les colonnes catégorielles
colonnes_categorielles = df.select_dtypes(include=['object']).columns

# Appliquer LabelEncoder à chaque colonne catégorielle
label_encoders = {}
for col in colonnes_categorielles:
    le = LabelEncoder()
    df[col] = df[col].astype(str)  # Convertir les valeurs NaN en string pour éviter les erreurs
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Enregistrer le label encoder pour chaque colonne


# In[12]:


# choix du features  et de la variable cible 
X = df.drop('CHURN', axis=1)
y = df['CHURN']

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer un modèle RandomForestClassifier
model = RandomForestClassifier(random_state=42)

# Entraînement le modèle
model.fit(X_train, y_train)

# prédictions sur l'ensemble de test
y_pred = model.predict(X_test)

# Évaluation du modèle
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))
print("\nPremières lignes de X_train :")
print(X_train.head().to_markdown(index=False, numalign="left", stralign="left"))


# In[13]:


# Charger le modèle entraîné
# Sauvegarder le modèle entraîné
joblib.dump(model, 'model.pkl')
model = joblib.load('model.pkl')

# Titre de l'application
st.title("Prédiction de l'attrition client")

# Créer des champs de saisie pour les features
region = st.selectbox("Région", df['REGION'].unique())
tenure = st.selectbox("Ancienneté", df['TENURE'].unique())
montant = st.number_input("Montant", min_value=0)
frequence_rech = st.number_input("Fréquence de recharge", min_value=0)
revenue = st.number_input("Revenue", min_value=0)
arpu_segment = st.number_input("ARPU Segment", min_value=0)
frequence = st.number_input("Fréquence", min_value=0)
data_volume = st.number_input("Volume de données", min_value=0)
on_net = st.number_input("On-net", min_value=0)
orange = st.number_input("Orange", min_value=0)
tigo = st.number_input("Tigo", min_value=0)
mrg = st.selectbox("MRG", df['MRG'].unique())
regularity = st.number_input("Régularité", min_value=0)
top_pack = st.selectbox("Top Pack", df['TOP_PACK'].unique())
freq_top_pack = st.number_input("Fréquence Top Pack", min_value=0)

# Bouton pour déclencher la prédiction
if st.button("Prédire"):
    # Créer un DataFrame avec les valeurs saisies
    input_data = pd.DataFrame({
        'REGION': [region],
        'TENURE': [tenure],
        'MONTANT': [montant],
        'FREQUENCE_RECH': [frequence_rech],
        'REVENUE': [revenue],
        'ARPU_SEGMENT': [arpu_segment],
        'FREQUENCE': [frequence],
        'DATA_VOLUME': [data_volume],
        'ON_NET': [on_net],
        'ORANGE': [orange],
        'TIGO': [tigo],
        'MRG': [mrg],
        'REGULARITY': [regularity],
        'TOP_PACK': [top_pack],
        'FREQ_TOP_PACK': [freq_top_pack]
    })

    # Faire la prédiction
    prediction = model.predict(input_data)[0]

    # Afficher le résultat
    if prediction == 0:
        st.success("Le client ne va probablement pas churn.")
    else:
        st.warning("Le client va probablement churn.")

