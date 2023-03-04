# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 09:20:23 2023

@author: sarah
"""
import streamlit as st
import pandas as pd
import pickle
import seaborn as sns
import warnings
import json
warnings.filterwarnings('ignore')

st.set_option('deprecation.showPyplotGlobalUse', False)
# Définir l'URL de l'API REST
API_URL = 'http://sarahkhomsi.pythonanywhere.com/'



#st.set_page_config(layout="wide")   
# Chargement des données de clients
data = pd.read_csv('C:/Users/sarah/Desktop/scoring/df_test_final.csv')

data_visu = pd.read_csv('C:/Users/sarah/Desktop/scoring/df_test_final_visu.csv')
data = data.rename(columns={'Unnamed: 0':'SK_ID_CURR'})
data_visu = data_visu.drop('Unnamed: 0',axis=1)
liste_clients = list(data['SK_ID_CURR'].unique())
#data = data.set_index('SK_ID_CURR')
df_dash = pd.read_csv('C:/Users/sarah/Desktop/scoring/df_dash.csv')
df_dash = df_dash.drop('Unnamed: 0',axis=1)

# Chargement du modèle de prédiction
with open('best_lgb_model.pkl', 'rb') as file:
    modele = pickle.load(file)



st.sidebar.markdown("## Pamarètres")
#st.sidebar.multiselect('Sélectionnez un client :', liste_clients)
client_id = st.sidebar.selectbox('Sélectionnez un client :', liste_clients)


# Filtrage des données pour ne conserver que les informations du client sélectionné
client_data = data.loc[data['SK_ID_CURR'] == client_id,:]#.drop('SK_ID_CURR',axis=1)

#client_data = client_data.set_index('SK_ID_CURR')
#client_data = client_data.drop('SK_ID_CURR',axis=1)
client_data_col = client_data[1:].columns


client_data_visu = data_visu.loc[data_visu['SK_ID_CURR'] == client_id,:].reset_index()
client_data_visu.drop('index',axis=1,inplace=True)

client_data_visu['DAYS_BIRTH'] = round(abs(client_data_visu['DAYS_BIRTH'])/365,0)
client_data_visu = client_data_visu.rename(columns={'DAYS_BIRTH':'AGE'})


# Prédiction du score du client sélectionné
#prediction = modele.predict(data.iloc[:,1:])



import requests

def get_prediction(client_data):
    # Convertir les données du client en un dictionnaire JSON
    client_data_dict = client_data.to_dict(orient='records')[0]
    client_data_json = json.dumps(client_data_dict)
    
    # Envoyer la requête POST à l'API
    response = requests.post(API_URL + 'predict', json=client_data_json)

    # Extraire la prédiction de la réponse de l'API
    prediction = response.json()['prediction']

    return prediction

# Modification de df_dash pour les graphiques
#df_dash['prediction'] = get_prediction(data)
#df_dash['prediction'] = df_dash['prediction'].astype('int')

df_dash['DAYS_BIRTH'] = round(abs(df_dash['DAYS_BIRTH'])/365,0)
df_dash['DAYS_EMPLOYED'] = round(abs(df_dash['DAYS_EMPLOYED'])/365,0)
df_dash = df_dash.rename(columns={'DAYS_BIRTH':'AGE',
                        'DAYS_EMPLOYED':'YEAR_EMPLOYED'})
#df_dash['prediction'] = df_dash['prediction'].astype('int')
df_dash['NAME_INCOME_TYPE_Working'] = df_dash['NAME_INCOME_TYPE_Working'] .replace({0: 'sans activité', 1: 'en activité'})
df_dash['NAME_EDUCATION_TYPE_Highereducation'] = df_dash['NAME_EDUCATION_TYPE_Highereducation'] .replace({0: 'Non', 1: 'Oui'});
df_dash['NAME_EDUCATION_TYPE_Secondarysecondaryspecial'] = df_dash['NAME_EDUCATION_TYPE_Secondarysecondaryspecial'] .replace({0: 'Non', 1: 'Oui'});

# Définir les intervalles
intervals = [0, 5, 10, 15, 25, 45]
# Définir les labels correspondants
labels = ['0-5', '6-10', '11-15', '20-25', '26+']
# Diviser les valeurs en intervalles et assigner les labels correspondants
df_dash['YEAR_EMPLOYED_concat'] = pd.cut(df_dash['YEAR_EMPLOYED'], bins=intervals, labels=labels)

df_dash['BURO_DAYS_CREDIT_MEAN_date'] = round((abs(df_dash['BURO_DAYS_CREDIT_MEAN'])/365),3)
#df_dash['prediction_str']= df_dash['prediction'] .replace({0: 'Refus', 1: 'Accord'})
df_dash['SK_ID_CURR'] = df_dash['SK_ID_CURR'].astype('object') 
## Affichage des informations du client
st.write("Informations du client :")
st.write(client_data_visu[['NAME_CONTRACT_TYPE','AGE','NAME_INCOME_TYPE',
                          'NAME_FAMILY_STATUS','CNT_CHILDREN']])
#création d'une première ligne pour afficher la prédiction
col_texte, col_kpi_1 = st.columns(2)

# Affichage du résultat de la prédiction
prediction = get_prediction(client_data)
if prediction == 1.0:
    col_texte.write('<p style="font-size:32px; color:green;">Accord</p>', unsafe_allow_html=True)
else:
    col_texte.write('<p style="font-size:32px; color:red;">Refus</p>', unsafe_allow_html=True)

# Afficher les KPI dans la deuxième colonne
#col_kpi_1.metric("Résultat", value=modele.predict(client_data.iloc[:,1:])[0])
col_kpi_1.metric("Pourcentage", value=round(modele.predict_proba(client_data.iloc[:,1:])[0][1]*100,2))



#création d'une deuxième ligne pour afficher la prédiction
col_kpi_client, col_graph = st.columns(2)

group_select = st.sidebar.selectbox('Choisissez le groupe à afficher', ['Tous'] + list(df_dash['prediction'].unique()))
# Menu déroulant pour sélectionner les colonnes
hue_variable = 'prediction'
df_kpi = df_dash.copy()
# Filtrage des données en fonction du groupe choisi
if group_select != 'Tous':
    df_dash = df_dash[df_dash[hue_variable] == group_select]

# Création des menus déroulants pour choisir le type de graphique
chart_type = st.sidebar.selectbox('Choisissez le type de graphique à afficher', ['Scatter', 'Line', 'Bar', 'Histogram', 'Pie'])

# Variables numériques et catégorielles pour chaque type de graphique
numeric_vars = [col for col in df_dash.columns if df_dash[col].dtype != 'object']
num_col_drop = ['prediction', 'YEAR_EMPLOYED_concat','BURO_DAYS_CREDIT_MEAN_date']
for item in num_col_drop:
    if item in numeric_vars:
        numeric_vars.remove(item)

categorical_vars = [col for col in df_dash.columns if df_dash[col].dtype == 'object']


cat_col_drop = ['prediction_str']

for item in cat_col_drop:
    if item in categorical_vars:
        categorical_vars.remove(item)

with col_graph:
        
    if chart_type in ['Scatter', 'Line']:
        # Variables numériques pour les graphiques Scatter et Line
        x_variable = st.sidebar.selectbox('Choisissez la variable x', numeric_vars)
        y_variable = st.sidebar.selectbox('Choisissez la variable y', numeric_vars)
    elif chart_type == 'Bar':
        # Variables catégorielles et numériques pour les graphiques Bar
        x_variable = st.sidebar.selectbox('Choisissez la variable x', categorical_vars + numeric_vars)
        y_variable = st.sidebar.selectbox('Choisissez la variable y', numeric_vars)
    elif chart_type in ['Pie']:
        # Variables catégorielles pour les graphiques Histogram et Pie
        x_variable = st.sidebar.selectbox('Choisissez la variable', categorical_vars)
    elif chart_type in ['Histogram']:
        # Variables catégorielles pour les graphiques Histogram et Pie
        x_variable = st.sidebar.selectbox('Choisissez la variable', numeric_vars)
    
    # Création du graphique en fonction du type choisi
    if chart_type == 'Scatter':
        sns.scatterplot(data=df_dash, x=x_variable, y=y_variable, hue=hue_variable)
    elif chart_type == 'Line':
        sns.lineplot(data=df_dash, x=x_variable, y=y_variable, hue=hue_variable)
    elif chart_type == 'Bar':
        sns.barplot(data=df_dash, x=x_variable, y=y_variable, hue=hue_variable)
    elif chart_type == 'Histogram':
        sns.histplot(data=df_dash, x=x_variable, hue=hue_variable)
    elif chart_type == 'Box':
        sns.boxplot(data=df_dash, x=x_variable, y=y_variable, hue=hue_variable)
    elif chart_type == 'Violin':
        sns.violinplot(data=df_dash, x=x_variable, y=y_variable, hue=hue_variable)
    
    # Affichage du graphique
    st.pyplot()
        
with col_kpi_client:
    if chart_type == 'Scatter':
        col_kpi_client.metric(label=f"{x_variable}",value=round(client_data.iloc[:, df_dash.columns.get_loc(x_variable)],2))
        col_kpi_client.metric(label=f"{y_variable}",value=round(client_data.iloc[:,df_dash.columns.get_loc(y_variable)],2))
    elif chart_type == 'Line':
        col_kpi_client.metric(label=f"{x_variable}",value=round(client_data.iloc[:, df_dash.columns.get_loc(x_variable)],2))
        col_kpi_client.metric(label=f"{y_variable}",value=round(client_data.iloc[:,df_dash.columns.get_loc(y_variable)],2))
    elif chart_type == 'Bar':
        col_kpi_client.metric(label=f"{x_variable}",value=round(client_data.iloc[:, df_dash.columns.get_loc(x_variable)],2))
        col_kpi_client.metric(label=f"{y_variable}",value=round(client_data.iloc[:,df_dash.columns.get_loc(y_variable)],2))
    elif chart_type == 'Histogram':
        col_kpi_client.metric(label=f"{x_variable}",value=round(client_data.iloc[:, df_dash.columns.get_loc(x_variable)],2))
    elif chart_type == 'Box': 
        col_kpi_client.metric(label=f"{x_variable}",value=round(client_data.iloc[:, df_dash.columns.get_loc(x_variable)],2))
    elif chart_type == 'Violin':
        col_kpi_client.metric(label=f"{x_variable}",value=round(client_data.iloc[:, df_dash.columns.get_loc(x_variable)],2))

















