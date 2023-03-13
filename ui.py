# Import required librairies
import streamlit as st
import json
import pandas as pd
import seaborn as sns
import requests
import warnings 


warnings.filterwarnings('ignore')
#Paramètres d'accesibilité pour les personnes en situation de handicap:
    
# Définir une palette de couleurs à contraste élevé
palette = ["#0072B2","#F0E442", "#009E73", "#D55E00", "#CC79A7",  "#56B4E9"]
# Appliquer la palette
sns.set_palette(palette)
# Définir le style par défaut pour les graphiques Seaborn
sns.set_style('darkgrid', {'axes.labelsize': 'large', 'axes.titlesize': 'x-large',"axes.facecolor": ".9"})
# Streamlit settings
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title = "Projet 7 - Implémentez un modèle de scoring", layout="wide")

df_test = pd.read_csv('https://github.com/saraharouni/scoring/blob/main/df_test_final.csv')
df_test= df_test.rename(columns={'Unnamed: 0':'SK_ID_CURR'})
df_test = df_test.iloc[:100,:]

# Ouverture des fichiers
# Pour afficher les KPI du client
df_dash = pd.read_csv('https://github.com/saraharouni/scoring/blob/main/df_dash.csv')
df_dash = df_dash.drop('Unnamed: 0',axis=1)

#Pour afficher les informations civiles du client
data_visu = pd.read_csv('https://github.com/saraharouni/scoring/blob/main/df_test_final_visu.csv')
data_visu = data_visu.drop('Unnamed: 0',axis=1)

# Pour réaliser les graphiques de comparaison avec les autres clients
df_train = pd.read_csv('https://github.com/saraharouni/scoring/blob/main/df_train_final.csv')
df_train= df_train.rename(columns={'Unnamed: 0':'SK_ID_CURR'})


# Modification des dataframes:
# data_visu

data_visu['DAYS_BIRTH'] = round(abs(data_visu['DAYS_BIRTH'])/365,0)
data_visu = data_visu.rename(columns={'DAYS_BIRTH':'AGE',
                        'DAYS_EMPLOYED':'YEAR_EMPLOYED'})
data_visu['AGE'] = data_visu['AGE'].astype('int')
# df_dash:
df_dash['DAYS_BIRTH'] = round(abs(df_dash['DAYS_BIRTH'])/365,0)
df_dash['DAYS_EMPLOYED'] = round(abs(df_dash['DAYS_EMPLOYED'])/365,0)
df_dash = df_dash.rename(columns={'DAYS_BIRTH':'AGE',
                        'DAYS_EMPLOYED':'YEAR_EMPLOYED'})
df_dash['AGE'] = df_dash['AGE'].astype('int')
df_dash['YEAR_EMPLOYED'] = df_dash['YEAR_EMPLOYED'].astype('int')
df_dash = df_dash.drop('NAME_INCOME_TYPE_Working',axis=1 )
df_dash['NAME_EDUCATION_TYPE_Highereducation'] = df_dash['NAME_EDUCATION_TYPE_Highereducation'] .replace({0: 'Niveau non Universitaire', 1: 'Niveau Universitaire'});
df_dash['NAME_EDUCATION_TYPE_Secondarysecondaryspecial'] = df_dash['NAME_EDUCATION_TYPE_Secondarysecondaryspecial'] .replace({0: 'Niveau non secondaire', 1: 'Niveau secondaire'});

# Définir les intervalles
intervals = [0, 5, 10, 15, 25, 45]
# Définir les labels correspondants
labels = ['0-5', '6-10', '11-15', '20-25', '26+']
# Diviser les valeurs en intervalles et assigner les labels correspondants
df_dash['YEAR_EMPLOYED_concat'] = pd.cut(df_dash['YEAR_EMPLOYED'], bins=intervals, labels=labels)
df_dash['BURO_DAYS_CREDIT_MEAN_date'] = round((abs(df_dash['BURO_DAYS_CREDIT_MEAN'])/365),3)
#df_dash['prediction_str']= df_dash['prediction'] .replace({0: 'Refus', 1: 'Accord'})
df_dash['SK_ID_CURR'] = df_dash['SK_ID_CURR'].astype('object') 
# Sélectionner les colonnes numériques
colonnes_numeriques = df_dash.select_dtypes(include=[float, int]).columns
# Arrondir les valeurs des colonnes numériques à 2 décimales
df_dash[colonnes_numeriques] = df_dash[colonnes_numeriques].round(2)
# df_train:
df_train['DAYS_BIRTH'] = round(abs(df_train['DAYS_BIRTH'])/365,0)
df_train['DAYS_EMPLOYED'] = round(abs(df_train['DAYS_EMPLOYED'])/365,0)
df_train = df_train.rename(columns={'DAYS_BIRTH':'AGE',
                         'DAYS_EMPLOYED':'YEAR_EMPLOYED'})
df_train = df_train.drop('NAME_INCOME_TYPE_Working',axis=1 )
df_train['NAME_EDUCATION_TYPE_Highereducation'] = df_train['NAME_EDUCATION_TYPE_Highereducation'] .replace({0: 'Non', 1: 'Oui'});
df_train['NAME_EDUCATION_TYPE_Secondarysecondaryspecial'] = df_train['NAME_EDUCATION_TYPE_Secondarysecondaryspecial'] .replace({0: 'Non', 1: 'Oui'});
# Diviser les valeurs en intervalles et assigner les labels correspondants
df_train['YEAR_EMPLOYED_concat'] = pd.cut(df_train['YEAR_EMPLOYED'], bins=intervals, labels=labels)
df_train['BURO_DAYS_CREDIT_MEAN_date'] = round((abs(df_train['BURO_DAYS_CREDIT_MEAN'])/365),3)
#df_dash['prediction_str']= df_dash['prediction'] .replace({0: 'Refus', 1: 'Accord'})
df_train['SK_ID_CURR'] = df_train['SK_ID_CURR'].astype('object') 

# Enregistrement de l'url de l'API dans une variable:
api_url = "http://sarahkhomsi.pythonanywhere.com/api"

# couleur de fond : #D7D8D7
image = 'logo_streamlit.png'
st.sidebar.image(image, caption='Prêt à dépenser', use_column_width=True)
# Layout (Sidebar)
st.sidebar.markdown("## Paramètres")
# Obtenir le client ID sélectionné à partir de la liste déroulante

client_id = st.sidebar.selectbox("Sélectionnez un client ID :",df_test["SK_ID_CURR"].tolist())
# Filtrage des données pour ne conserver que les informations du client sélectionné
client_data = df_dash.loc[df_dash['SK_ID_CURR'] == client_id,:]

client_data_col = client_data[1:].columns

# Préparation de la requête et affichage de la réponse
idx = client_id
j_idx = json.dumps(idx)

headers = {'content-type': 'application/json', 'Accept-Charset': 'UTF-8'}
request = requests.post(api_url, data = j_idx, headers = headers)
req = request.json()
proba_api = req['proba']
rep_api = req['rep']
rep_api_1 = req['importance']
features_importance = pd.read_json(rep_api_1)

## Dashboard title
st.write('<h1 style="text-align: center;"">Dashboard de scoring de Prêt à dépenser</h1>',unsafe_allow_html=True)


## Affichage des informations du client
st.table(data_visu.loc[data_visu['SK_ID_CURR']==client_id][['NAME_CONTRACT_TYPE','AGE','NAME_INCOME_TYPE',
                          'NAME_FAMILY_STATUS','CNT_CHILDREN']])


#création d'une première ligne pour afficher la prédiction
col_texte, col_kpi_1 = st.columns(2)
# Ajouter un titre

#col_kpi_1.metric("Pourcentage (%)", value=round(proba_api*100,2))
#☺col1.metric(metric_label, metric_value)
metric_value = round(proba_api*100,2)
metric_label = "Probabilité"


with col_texte:
    # Affichage du résultat de la prédiction
    if rep_api == 1:
        st.write('<p style="font-size:46px; color:green;text-align: center">Accord</p>', unsafe_allow_html=True)
    else:
        st.write('<p style="font-size:46px; color:red;text-align: center">Refus</p>', unsafe_allow_html=True)
if rep_api == 1:       
    col_kpi_1.markdown(f"<p style='font-size: 42px; color:green;text-align: center;'>{metric_label}: {metric_value} %</p>",unsafe_allow_html=True)
else:
    col_kpi_1.markdown(f"<p style='font-size: 42px; color:red;text-align: center;'>{metric_label}: {metric_value} %</p>",unsafe_allow_html=True)
#création d'une deuxième ligne pour afficher la prédiction
col_kpi_client,col_graph = st.columns(2)
df_train['prédiction'] = df_train['TARGET'].replace({0: 'Accord', 1: 'Refus'})
group_select = st.sidebar.selectbox('Choisissez le groupe à afficher', ['Tous'] + list(df_train['prédiction'].unique()))
features_graph = st.sidebar.checkbox('Afficher les features importances')

    
# Menu déroulant pour sélectionner les colonnes
hue_variable = 'prédiction'
df_kpi = df_dash.copy()
# Filtrage des données en fonction du groupe choisi
if group_select != 'Tous':
    df_train = df_train[df_train[hue_variable] == group_select]

graphique_liste = ['Bar', 'Scatter', 'Line', 'Histogram','Violin']
# Création des menus déroulants pour choisir le type de graphique
chart_type = st.sidebar.selectbox('Choisissez le type de graphique à afficher', graphique_liste)


# Variables numériques et catégorielles pour chaque type de graphique
numeric_vars = [col for col in df_train.columns[2:] if df_train[col].dtype != 'object']
num_col_drop = ['prediction', 'YEAR_EMPLOYED_concat','BURO_DAYS_CREDIT_MEAN_date']

for item in num_col_drop:
    if item in numeric_vars:
        numeric_vars.remove(item)
categorical_vars = [col for col in df_train.columns[2:] if df_train[col].dtype == 'object']
cat_col_drop = ['prediction_str']

df_dash[numeric_vars] = df_dash[numeric_vars].round(2)
#df_dash[categorical_vars] = df_dash[categorical_vars].astype('str')
for item in cat_col_drop:
    if item in categorical_vars:
        categorical_vars.remove(item)

with col_graph:
    st.write('<h4 style="text-align: center;"">Graphique de comparaison</h4>',unsafe_allow_html=True)
    #st.subheader("Graphiques de comparaison ")
    if chart_type in ['Scatter', 'Line']:
        # Variables numériques pour les graphiques Scatter et Line
        x_variable = st.sidebar.selectbox('Choisissez la variable X', numeric_vars)
        y_variable = st.sidebar.selectbox('Choisissez la variable Y', numeric_vars)
        
    elif chart_type == 'Bar':
        # Variables catégorielles et numériques pour les graphiques Bar
        x_variable = st.sidebar.selectbox('Choisissez la variable X', categorical_vars + numeric_vars)
        y_variable = st.sidebar.selectbox('Choisissez la variable Y', numeric_vars)
    
    elif chart_type in ['Histogram']:
        # Variables catégorielles pour les graphiques Histogram et Pie
        x_variable = st.sidebar.selectbox('Choisissez la variable X', numeric_vars)
    elif chart_type in ['Violin']:
        # Variables catégorielles pour les graphiques Histogram et Pie
        x_variable = st.sidebar.selectbox('Choisissez la variable X', categorical_vars)
        y_variable = st.sidebar.selectbox('Choisissez la variable Y', numeric_vars)
    # Expander qui s'ouvre lorsqu'on sélectionne une valeur dans la barre de sélection
    
    # Création du graphique en fonction du type choisi
    if chart_type == 'Scatter':
        ax = sns.scatterplot(data=df_train, x=x_variable, y=y_variable, hue=hue_variable)
        sns.move_legend(ax,loc='upper left', bbox_to_anchor=(1.1, 0.5))
        
    elif chart_type == 'Line':
        ax =sns.lineplot(data=df_train, x=x_variable, y=y_variable, hue=hue_variable)
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.1, 0.5))
    elif chart_type == 'Bar':
        ax= sns.barplot(data=df_train, x=x_variable, y=y_variable, hue=hue_variable)
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.1, 0.5))
        
    elif chart_type == 'Histogram':
        ax = sns.histplot(data=df_train, x=x_variable, hue=hue_variable,multiple="stack",)
        sns.move_legend(ax, loc="upper left", bbox_to_anchor=(1.1, 0.5))
    
    elif chart_type == 'Box':
        ax = sns.boxplot(data=df_train, x=x_variable, y=y_variable, hue=hue_variable)
        sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.1, 0.5))
    elif chart_type == 'Violin':
        ax = sns.violinplot(data=df_train, x=x_variable, y=y_variable, hue=hue_variable)
        sns.move_legend(ax, loc="center left", bbox_to_anchor=(1.1, 0.5))
    # Affichage du graphique
    st.pyplot()
with col_kpi_client:   
    #st.write('<h4 style="text-align: center;"">Données du client</h4>',unsafe_allow_html=True)
    st.write('<h4>Données du client</h4>',unsafe_allow_html=True)
    st.write("Variable X")
    if chart_type != 'Histogram':
        # Déterminer si x_variable est numérique
        if isinstance(x_variable, (int, float)):
            col_kpi_client.metric(label=f"{x_variable}",value=round(client_data.iloc[:, df_dash.columns.get_loc(x_variable)],2))
            
        else:
            #st.write(f"**KPI x:** {x_variable}")
            col_kpi_client.metric(label=f"{x_variable}",value=client_data[x_variable].astype(str).iloc[0])
            #st.write(client_data.iloc[:, df_dash.columns.get_loc(x_variable)])
            
            st.write("Variable Y")  
        # Vérifier si y_variable est un objet (chaîne de caractères)
        if isinstance(y_variable, str):
            # Déterminer si y_variable est numérique
            if isinstance(client_data.iloc[:, df_dash.columns.get_loc(y_variable)], (int, float)):
                col_kpi_client.metric(label=f"{y_variable}",value=round(client_data.iloc[:, df_dash.columns.get_loc(y_variable)],2))
            else:
                col_kpi_client.metric(label=f"{y_variable}",value=client_data[y_variable].astype(str).iloc[0])
    else :
         # Déterminer si x_variable est numérique
         if isinstance(x_variable, (int, float)):
             col_kpi_client.metric(label=f"{x_variable}",value=round(client_data.iloc[:, df_dash.columns.get_loc(x_variable)],2))
         else:
             #st.write(f"**KPI x:** {x_variable}")
             col_kpi_client.metric(label=f"{x_variable}",value=client_data[x_variable].astype(str).iloc[0])


col_desc_var, col_desc_graph = st.columns(2)
with col_desc_var:
    with st.expander('Description des variables'):
       st.write("'Age' : Age du client (années),\n \
          'Days Employed' : Durée d\'activité (années),\n \
          'Region Rating Client': Note attribuée à la région de résidence,\n \
          'Region Rating Client W City' : Note attribuée à la ville de résidence,\n \
          'Ext_Source_1 /_2 / _3' : Score normalisé attribué par une source extérieure,\n \
          'NAME_EDUCATION_TYPE_Highereducation' : Indique si le client a un niveau universitaire,\n \
          'NAME_EDUCATION_TYPE_Secondarysecondaryspecial' : Indique si le client a un niveau d'études secondaire,\n \
          'Days_Perc' : Corresond au pourcentage de la durée d\'activité,\n \
          'Buro_Days_Credit_min / _mean' : Nombre de jours min et en moyenne depuis la demande de prêt,\n \
          'Buro_Days_Credi_Update_Mean' : Nombre de jours en moyenne depuis la mise à jour de la demande de prêt,\n \
          'Buro_credit_active_active_mean' : Moyenne du score attribué pour l\'état des prêts obtenus,\n \
          'Buro_Credit_Active_Closed_Mean' : Moyenne du score attribué pour l\'état des prêts remboursés,\n \
          'Prev_Name_contract_status_approved_mean' : Note attribuée en fonction des anciennes demandes de prêt acceptées,\n \
          'Prev_Name_contract_status_refused_mean' : Note attribuée en fonction des anciennes demandes de prêt refusées,\n \
          'Prev_Code_Reject_Reason_Hc_mean' : Code du motif de refus des demandes de type Hc,\n \
          'Prev_Code_Reject_Reason_XAP_mean' : Code du motif de refus des demandes de type XAP")

with col_desc_graph:
    # Expander qui s'ouvre lorsqu'on sélectionne une valeur dans la barre de sélection
    with st.expander('Description du graphique sélectionné'):
        if chart_type == 'Bar':
            st.write("Un diagramme à barres, également appelé diagramme à bâtons, est un graphique qui présente des variables catégorielles avec des barres rectangulaires avec des hauteurs ou des longueurs proportionnelles aux valeurs qu'elles représentent. Les barres peuvent être tracées verticalement ou horizontalement.")
        elif chart_type == 'Scatter':
            st.write('les scatter plots sont un type de graphique sous forme d’un nuage de points montrant ainsi comment une variable  est affectée par une autre.')
        elif chart_type == 'Line':
            st.write('Un graphique linéaire est un graphique qui utilise des lignes pour relier des points de données individuels qui affichent des valeurs quantitatives sur un intervalle de temps spécifié. ')
        elif chart_type == 'Histogram':
            st.write("Un histogramme est un graphe permettant de représenter la répartition d'une variable")
        elif chart_type == 'Violin':
            st.write("Les violin plots sont similaires au boxplots. L’avantage de ces derniers par rapport aux boxplots est qu’ils nous permette de visualiser la distribution des données et et leur densité de probabilité.")
if features_graph:
    ax = sns.barplot(data=features_importance[:15], y='Feature',x='Value')
    st.pyplot()
    
    




