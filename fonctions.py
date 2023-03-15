import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import gc
import time
import joblib
import imblearn
import sys
from contextlib import contextmanager
from datetime import datetime
    
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, balanced_accuracy_score, roc_auc_score, roc_curve,make_scorer, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier

import lightgbm as lgb

import mlflow
import mlflow.sklearn

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

path = 'https://raw.githubusercontent.com/saraharouni/scoring/main/dashboard/'

x_col = ['DAYS_BIRTH', 'DAYS_EMPLOYED', 'REGION_RATING_CLIENT',
       'REGION_RATING_CLIENT_W_CITY', 'EXT_SOURCE_1', 'EXT_SOURCE_2',
       'EXT_SOURCE_3', 'NAME_INCOME_TYPE_Working',
       'NAME_EDUCATION_TYPE_Highereducation',
       'NAME_EDUCATION_TYPE_Secondarysecondaryspecial', 'DAYS_EMPLOYED_PERC',
       'BURO_DAYS_CREDIT_MIN', 'BURO_DAYS_CREDIT_MEAN',
       'BURO_DAYS_CREDIT_UPDATE_MEAN', 'BURO_CREDIT_ACTIVE_Active_MEAN',
       'BURO_CREDIT_ACTIVE_Closed_MEAN',
       'PREV_NAME_CONTRACT_STATUS_Approved_MEAN',
       'PREV_NAME_CONTRACT_STATUS_Refused_MEAN',
       'PREV_CODE_REJECT_REASON_HC_MEAN', 'PREV_CODE_REJECT_REASON_XAP_MEAN']

### Cette fonction est utilisée pour chronométrer des opérations 
### dans le code afin de mesurer les performances et d'optimiser le temps d'exécution.

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.0f}s".format(title, time.time() - t0))
    
    
### Fonction qui permet d'encoder les variables catégorielles avec One Hot Encoder ###
    
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


### Définition de la fonction coût pour la validation croisée

def cost_function(y_test, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cost = 10 * fp + fn
    return cost

### Fonction qui supprime les colonnes qui ne contiennent qu'une valeur

def col_drop(df):
    for col in df:
        if(len(df.loc[:,col].unique()) == 1):
            df.pop(col)
    return df

### Fonction d'entraînement d'un modèle dummy 

def modele_dummy(train_x,train_y,test_x):
    dummy_clf = DummyClassifier(strategy="most_frequent",random_state=0)
    dummy_clf.fit(train_x, train_y)
    y_pred_dummy = dummy_clf.predict(test_x)
    y_pred_proba_dummy = dummy_clf.predict_proba(test_x)
    dummy_score = dummy_clf.score(train_x,train_y)
    
    return dummy_clf, y_pred_dummy,y_pred_proba_dummy

### Fonction de traitement des valeurs manquantes

def nan_value(df):
    print('Valeurs manquantes avant traitement : ',df.isna().mean())
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.fillna(df.median())
    print('df shape :', df.shape)
    print('Valeurs manquantes après traitement : ',df.isna().mean())
    print(df.shape)
    return df

### Fonction d'évaluation de la performance d'un modele de classification avec 
### plusieurs métriques

def evaluate_classification_model(y_test, y_pred_optimal, y_proba):
       
    # Compute the accuracy score
    accuracy = accuracy_score(y_test, y_pred_optimal)
    
    # Compute the F1 score
    f1 = f1_score(y_test, y_pred_optimal)
    
    # Compute the precision score
    precision = precision_score(y_test, y_pred_optimal)
    
    # Compute the recall score
    recall = recall_score(y_test, y_pred_optimal)
    
    # Compute the AUC-ROC score
    auc = roc_auc_score(y_test, y_proba)
    
     # Compute le score de la fonction coût
    cost = cost_function(y_test, y_pred_optimal)
    
       
    # Store the results in a dictionary
    results = {
        "accuracy": accuracy,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "auc": auc,
        "cost": cost
       
    }
    
    return results

### Fonction de prétraitement des données pour le modele de classification 

def preprocessing(df,standardiser= StandardScaler(),mode='undersampling'):
    print('suppression des colonnes à valeur unique')
    df = col_drop(df)
    print('Suppression des colonnes qui contiennent plus de 75% de valeurs manquantes')
    df= df.loc[:, df.isnull().mean() < 0.75]
            
    print('Séparation df_train / df_test')
    
    df_train = df[df['TARGET'].notnull()]
    df_test = df[df['TARGET'].isnull()]
    
    print(df_train['TARGET'].value_counts())
    
    
    print('Imputation des valeurs manquantes')
    df_train = nan_value(df_train)
    df_test = nan_value(df_test)
    
    print('Sampling de df_train')
    df_train = df_train.sample(n=50000, replace=False, random_state=1)
    df_test = df_test.sample(n=1500, replace=False, random_state=1)
    df_test_list = df_test['SK_ID_CURR'].to_list()
    df_test = df_test.drop(['TARGET','index','SK_ID_CURR'],axis=1)
    
    X = df_train.drop(['TARGET', 'index', 'SK_ID_CURR'], axis=1)
    
    y = df_train['TARGET']
    
    # Application de SelectKBest pour sélectionner les 20 meilleures features
    selector = SelectKBest(k=20)
    X = selector.fit_transform(X, y)

    # Enregistrement des features sélectionnées
    selected_features = df_train.drop(columns=['TARGET','index','SK_ID_CURR']).columns[selector.get_support()]
    df_train = pd.concat([df_train['TARGET'], df_train[selected_features]], axis=1)
    df_train.to_csv('df_train_final.csv')
    df_test = df_test[selected_features]
    test_col = df_test.columns
    X = df_train.drop(['TARGET'], axis=1)
    x_col = X.columns
    print('Train_test_split')
    X_train, X_test, y_train, y_test = train_test_split(
         X, y, test_size=0.2, random_state=42)

    print('Normalisation des données avec',standardiser)
    scaler = standardiser
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    df_test = scaler.transform(df_test)
    df_test = pd.DataFrame(df_test,index=df_test_list,columns=test_col)
    df_test.to_csv(path+"df_test_final.csv")  
    
    print('Application de la technique de sampling choisie :',mode)
    
    if mode== 'undersampling':
        rus = RandomUnderSampler(sampling_strategy='auto')
        X_train, y_train = rus.fit_resample(X_train, y_train)
        ax = y_train.value_counts().plot.pie(autopct='%.2f')
        _ = ax.set_title("Under-sampling")
    elif mode =='oversampling':
        ros = RandomOverSampler(sampling_strategy='auto')
        X_train, y_train = ros.fit_resample(X_train, y_train)
        ax = y_train.value_counts().plot.pie(autopct='%.2f')
        _ = ax.set_title("Over-sampling")
    else:
        print('Pas de méthode sélectionnée')
        ax = y_train.value_counts().plot.pie(autopct='%.2f')
    return X_train, X_test, y_train, y_test, x_col 


### Fonction qui affiche la courbe ROC et l'enregistre au format png

def create_roc_auc_plot(model, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn import metrics
    metrics.plot_roc_curve(model, X_test, y_test) 
    plt.savefig('roc_auc_curve.png')
    plt.show()
    
### Fonction qui affiche la matrice de confusion et l'enregistre au format txt

def create_confusion_matrix_plot(clf, X_test, y_test):
    import matplotlib.pyplot as plt
    from sklearn.metrics import plot_confusion_matrix
    
    plot_confusion_matrix(clf, X_test,y_test)
    plt.savefig('confusion_matrix.png')
    plt.show()
    
### Fonction qui affiche le rapport de classification et l'enregistre au format png

def create_classification_report(clf,test_y,y_pred_optimal):
    from sklearn.metrics import classification_report
    name = list(set(test_y))
    clsf_report = pd.DataFrame(classification_report(test_y, y_pred_optimal,target_names=name, output_dict=True)).round(2).transpose()
    print(clsf_report)
    clsf_report.to_string('classification_report.txt',index=True)
    plt.savefig('classification_report.png')
    
### Fonction qui permet d'afficher les features importances sur un modèle random forest
def feature_importance_rf(model,x_col):
    
    importance_data = pd.DataFrame()
    importance_data['features'] = x_col
    importance_data['importance']= model.feature_importances_
    # plot feature importance
    sns.barplot(data=importance_data[:15], y='features',x='importance')
    plt.savefig('feature_importance.png')
    plt.show()

### Fonction qui permet d'afficher les features importances   

def feature_importance_reg(model, X_train,y_train):
    model.fit(X_train, y_train)
    indices = np.argsort(model.feature_importances_)[::-1]
    features = []
    for i in range(20):
        features.append(X_train.columns[indices[i]])

    sns.barplot(x=features, y=model.feature_importances_[indices[range(20)]], color=("orange"))
    plt.xlabel('Features importance')
    plt.xticks(rotation=90)
    plt.savefig('feature_importance.png')
    plt.show()
    
### Fonction qui permet d'afficher les features importances du modèle dummy    
    
def feature_importance_dummy(model,x_col):
    
    importance_data = pd.DataFrame()
    importance_data['features'] = x_col
    importance_data['coef']= model.coef_
    # plot feature importance
    sns.barplot(data=importance_data[:15], y='features',x='coef')
    plt.show()
    
### Fonction qui permet d'entrainer le modèle random forest, calcul les métriques avant et après
### optimisation avec GridSearch CV

def random_forest(X_train,y_train,X_test,y_test):
    print('Création dun modèle ')
    rf_model = RandomForestClassifier(random_state=42)
    print('Entrainement du modèle sur X_train / y_train')
    rf_model.fit(X_train, y_train)
   
    y_pred = rf_model.predict(X_test)
    y_proba = rf_model.predict_proba(X_test)[:, 1]
        
    print('Optimisation du seuil de classification pour maximiser l\'AUC')      
    # Optimiser le seuil de classification pour maximiser l'AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    print('Scores du modèle avant optimisation')
    print(evaluate_classification_model(y_test, y_pred_optimal, y_proba))
    
    print('Affichage des features importances avant optimisation')
    print(feature_importance_rf(rf_model,x_col))
    
    print('Affichage de la courbe Roc avant optimisation')
    print(create_roc_auc_plot(rf_model,X_test, y_test))
    
    print('Affichage du rapport de classification avant optimisation')
    print(create_classification_report(rf_model,y_test,y_pred_optimal))
    
    print('Affichage de la matrice de confusion avant optimisation')
    print(create_confusion_matrix_plot(rf_model, X_test, y_test))   
    
    print('Création d\'un dictionnaire pour optimiser les paramètres')
    
    class_counts = np.bincount(y_train)
    n_classes = len(class_counts)
    class_weights = {i: sum(class_counts)/class_counts[i] for i in range(n_classes)}
    
    # Optimisation du modèle 
    param_grid = {
    'n_estimators': [255,260],
    'max_depth': [13,14],
    'class_weight':['balanced',class_weights]   }
    
    print('Création d\'un dictionnaire de scoring')
      
    scoring = {
     'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score),
    'cost': make_scorer(cost_function),
    'precision' : make_scorer(precision_score)
    }
   
    # Création de l'objet GridSearchCV pour optimiser les hyperparamètres avec la fonction coût
    grid_search = GridSearchCV(rf_model, param_grid, scoring=scoring, cv=5,verbose=2, refit='precision')
    print('Execution de GridSearchCV sur le modèle')
    # Entraînement du modèle avec la validation croisée
    grid_search.fit(X_train, y_train)

    # affichage des résultats
    print("Best parameters: ", grid_search.best_params_)
    print("Best accuracy: ", grid_search.best_score_)
    print("Best recall: ", grid_search.cv_results_['mean_test_recall'][grid_search.best_index_])
    print("Best cost: ", grid_search.cv_results_['mean_test_cost'][grid_search.best_index_])
    print("Best precision score: ", grid_search.cv_results_['mean_test_precision'][grid_search.best_index_])
    
    # Entraînement du modèle sur les données d'entraînement avec les meilleurs hyperparamètres trouvés
    print('Ajustement du modèle avec les paramètres optimisés')
    best_rf_model = grid_search.best_estimator_
    best_rf_model.fit(X_train, y_train)

    
    y_pred_best = best_rf_model.predict(X_test)
    y_proba_best = best_rf_model.predict_proba(X_test)[:, 1]
    
    print('Optimisation du seil de classification pour maximiser l\'AUC')      
    # Optimiser le seuil de classification pour maximiser l'AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Prédire les classes pour l'ensemble de test avec le seuil de classification optimal
    y_pred_optimal_best = (y_proba_best >= optimal_threshold).astype(int)
    print('Scores du modèle après optimisation')
    
    results = evaluate_classification_model(y_test, y_pred_optimal_best, y_proba_best)
     # Enregistrement des metrics dans une dataframe
    results_df = pd.DataFrame(columns=['Modèle', 'Best accuracy', 'Best recall', 'Best cost','Best precision Score'])
    # Ajouter les résultats de la recherche de grille à la dataframe
    results_df.loc[0] = ['Random forest',grid_search.best_score_, 
                         grid_search.cv_results_['mean_test_recall'][grid_search.best_index_], 
                         grid_search.cv_results_['mean_test_cost'][grid_search.best_index_], 
                         grid_search.cv_results_['mean_test_precision'][grid_search.best_index_]]
    
    
    print(results_df)
    print('Affichage des features importances après optimisation')
    print(feature_importance_rf(best_rf_model,x_col))
    
    print('Affichage de la courbe Roc après optimisation')
    print(create_roc_auc_plot(best_rf_model,X_test, y_test))
    
    print('Affichage du rapport de classification après optimisation')
    print(create_classification_report(best_rf_model,y_test,y_pred_optimal_best))
    
    print('Affichage de la matrice de confusion après optimisation')
    print(create_confusion_matrix_plot(best_rf_model, X_test, y_test))  
    
    return y_pred_optimal, y_proba, rf_model, y_pred_optimal_best, y_proba_best, best_rf_model, results, results_df

### Fonction équivalente à random forest pour le dummy Classifier

def modele_dummy(X_train,y_train,X_test,y_test):
    dummy_clf = DummyClassifier(strategy="most_frequent",random_state=0)
    dummy_clf.fit(X_train, y_train)
    y_pred_dummy = dummy_clf.predict(X_test)
    y_pred_proba_dummy = dummy_clf.predict_proba(X_test)[:, 1]
    dummy_score = dummy_clf.score(X_train,y_train)
    
    print('Optimisation du seuil de classification pour maximiser l\'AUC')      
    # Optimiser le seuil de classification pour maximiser l'AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_dummy)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_optimal_dummy = (y_pred_proba_dummy>= optimal_threshold).astype(int)
    
    print('Scores du modèle ')
    print(evaluate_classification_model(y_test, y_pred_optimal_dummy, y_pred_proba_dummy))
    results = evaluate_classification_model(y_test, y_pred_optimal_dummy, y_pred_proba_dummy)    
    print('Affichage de la courbe Roc ')
    print(create_roc_auc_plot(dummy_clf,X_test, y_test))
    
    print('Affichage du rapport de classification ')
    print(create_classification_report(dummy_clf,y_test,y_pred_optimal_dummy))
    
    print('Affichage de la matrice de confusion')
    print(create_confusion_matrix_plot(dummy_clf, X_test, y_test))
    
    
    return dummy_clf, y_pred_optimal_dummy,y_pred_proba_dummy, results

# Fonction équivalente à random forest adaptée à lightgbm

def lightgbm(X_train,y_train,X_test,y_test):
    
    print('Création d\'un modèle LGBM ')
    
    lgb_model = lgb.LGBMClassifier(random_state=42)
    
    
    print('Entrainement du modèle sur X_train / y_train')
    
    lgb_model.fit(X_train, y_train)
   
    y_pred = lgb_model.predict(X_test)
    y_proba = lgb_model.predict_proba(X_test)[:, 1]
        
    print('Optimisation du seuil de classification pour maximiser l\'AUC')      
    # Optimiser le seuil de classification pour maximiser l'AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    print('Scores du modèle avant optimisation')
    print(evaluate_classification_model(y_test, y_pred_optimal, y_proba))
    
    print('Affichage des features importances avant optimisation')
    print(feature_importance_rf(lgb_model,x_col))
    
    print('Affichage de la courbe Roc avant optimisation')
    print(create_roc_auc_plot(lgb_model,X_test,y_test))
    
    print('Affichage du rapport de classification avant optimisation')
    print(create_classification_report(lgb_model,y_test,y_pred_optimal))
    
    print('Affichage de la matrice de confusion avant optimisation')
    print(create_confusion_matrix_plot(lgb_model, X_test, y_test))
    
        
    print('Création d\'un dictionnaire pour optimiser les paramètres')
    # Optimisation du modèle 
    param_grid = {
      'max_depth': [8,9,10],
    'learning_rate': [0.3,0.4],
    'n_estimators': [260,275,280],
    'num_leaves': [40,60,80],
    'scale_pos_weight': [None, 10]   
    }
    
    print('Création d\'un dictionnaire de scoring')
      
    scoring = {
     'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score),
    'cost': make_scorer(cost_function),
    'precision' : make_scorer(precision_score)
    }
   
    # Création de l'objet GridSearchCV pour optimiser les hyperparamètres avec la fonction coût
    grid_search = GridSearchCV(lgb_model, param_grid, scoring=scoring, cv=5,verbose=2, refit='precision')
    print('Execution de GridSearchCV sur le modèle')
    # Entraînement du modèle avec la validation croisée
    grid_search.fit(X_train, y_train)

    # affichage des résultats
    print("Best parameters: ", grid_search.best_params_)
    print("Best accuracy: ", grid_search.best_score_)
    print("Best recall: ", grid_search.cv_results_['mean_test_recall'][grid_search.best_index_])
    print("Best cost: ", grid_search.cv_results_['mean_test_cost'][grid_search.best_index_])
    print("Best precision score : ",grid_search.cv_results_['mean_test_precision'][grid_search.best_index_])
    
    
    # Entraînement du modèle sur les données d'entraînement avec les meilleurs hyperparamètres trouvés
    print('Ajustement du modèle avec les paramètres optimisés')
    best_lgb_model = grid_search.best_estimator_
    best_lgb_model.fit(X_train, y_train)

    
    y_pred_best = best_lgb_model.predict(X_test)
    y_proba_best = best_lgb_model.predict_proba(X_test)[:, 1]
    
    print('Optimisation du seil de classification pour maximiser l\'AUC')      
    # Optimiser le seuil de classification pour maximiser l'AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Prédire les classes pour l'ensemble de test avec le seuil de classification optimal
    y_pred_optimal_best = (y_proba_best >= optimal_threshold).astype(int)
    print('Scores du modèle après optimisation')
    
    #print(evaluate_classification_model(y_test, y_pred_optimal_best, y_proba_best))
    
    results = evaluate_classification_model(y_test, y_pred_optimal_best, y_proba_best)
    
    # Enregistrement des metrics dans une dataframe
    results_df = pd.DataFrame(columns=['Modèle', 'Best accuracy','Best recall', 'Best cost', 'Best precision score'])
    # Ajouter les résultats de la recherche de grille à la dataframe
    results_df.loc[0] = ['Lightgbm', grid_search.best_score_, 
                         grid_search.cv_results_['mean_test_recall'][grid_search.best_index_], 
                         grid_search.cv_results_['mean_test_cost'][grid_search.best_index_], 
                         grid_search.cv_results_['mean_test_precision'][grid_search.best_index_]]
    print(results_df)
    
    print('Affichage des features importances après optimisation')
    print(feature_importance_rf(best_lgb_model,x_col))
    
    print('Affichage de la courbe Roc après optimisation')
    print(create_roc_auc_plot(best_lgb_model,X_test, y_test))
    
    print('Affichage du rapport de classification après optimisation')
    print(create_classification_report(best_lgb_model,y_test,y_pred_optimal_best))
    
    print('Affichage de la matrice de confusion après optimisation')
    print(create_confusion_matrix_plot(best_lgb_model, X_test, y_test))
       
    return y_pred_optimal, y_proba, lgb_model, y_pred_optimal_best, y_proba_best, best_lgb_model, results, results_df


### Fonction équivalente à random forest adaptée à une regression logistique
def logistic_regression(X_train, y_train, X_test, y_test):
    print('Création d\'un modèle de régression logistique')
    logreg_model = LogisticRegression(random_state=42, max_iter=1000)
    
    print('Entrainement du modèle sur X_train / y_train')
    logreg_model.fit(X_train, y_train)

    y_pred = logreg_model.predict(X_test)
    y_proba = logreg_model.predict_proba(X_test)[:, 1]
        
    print('Optimisation du seuil de classification pour maximiser l\'AUC')      
    # Optimiser le seuil de classification pour maximiser l'AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    
    print('Scores du modèle avant optimisation')
    print(evaluate_classification_model(y_test, y_pred_optimal, y_proba))
    
    print('Affichage de la courbe Roc avant optimisation')
    print(create_roc_auc_plot(logreg_model,X_test, y_test))
    
    print('Affichage du rapport de classification avant optimisation')
    print(create_classification_report(logreg_model,y_test,y_pred_optimal))
    
    print('Affichage de la matrice de confusion avant optimisation')
    print(create_confusion_matrix_plot(logreg_model, X_test, y_test))
    
    print('Création d\'un dictionnaire pour optimiser les paramètres')
    
    class_counts = np.bincount(y_train)
    n_classes = len(class_counts)
    class_weights = {i: sum(class_counts)/class_counts[i] for i in range(n_classes)}
    
    # Optimisation du modèle 
    param_grid = {
    'penalty': ['l2'],
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'class_weight':['balanced', class_weights]   
    }
    
    print('Création d\'un dictionnaire de scoring')
    
    scoring = {
     'accuracy': make_scorer(accuracy_score),
    'recall': make_scorer(recall_score),
    'cost': make_scorer(cost_function),
    'precision' : make_scorer(precision_score)
    }
    
    # Création de l'objet GridSearchCV pour optimiser les hyperparamètres avec la fonction coût
    grid_search = GridSearchCV(logreg_model, param_grid, scoring=scoring, cv=5,verbose=2, refit='precision')
    print('Execution de GridSearchCV sur le modèle')
    # Entraînement du modèle avec la validation croisée
    grid_search.fit(X_train, y_train)

    # affichage des résultats
    print("Best parameters: ", grid_search.best_params_)
    print("Best accuracy: ", grid_search.best_score_)
    print("Best recall: ", grid_search.cv_results_['mean_test_recall'][grid_search.best_index_])
    print("Best cost: ", grid_search.cv_results_['mean_test_cost'][grid_search.best_index_])
    print("Best precision score: ", grid_search.cv_results_['mean_test_precision'][grid_search.best_index_])
    
    # Entraînement du modèle sur les données d'entraînement avec les meilleurs hyperparamètres trouvés
    print('Ajustement du modèle avec les paramètres optimisés')
    best_lr_model = grid_search.best_estimator_
    best_lr_model.fit(X_train, y_train)

    y_pred_best = best_lr_model.predict(X_test)
    y_proba_best = best_lr_model.predict_proba(X_test)[:, 1]
    
    print('Optimisation du seuil de classification pour maximiser l\'AUC')      
    # Optimiser le seuil de classification pour maximiser l'AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_proba_best)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Prédire les classes pour l'ensemble de test avec le seuil de classification optimal
    y_pred_optimal_best = (y_proba_best >= optimal_threshold).astype(int)
    print('Scores du modèle après optimisation')
    
    results = evaluate_classification_model(y_test, y_pred_optimal_best, y_proba_best)
    # Enregistrement des metrics dans une dataframe
    results_df = pd.DataFrame(columns=['Modèle', 'Best accuracy', 'Best recall', 'Best cost','Best precision Score'])
    # Ajouter les résultats de la recherche de grille à la dataframe
    results_df.loc[0] = ['Logistic Regression', grid_search.best_score_, 
                         grid_search.cv_results_['mean_test_recall'][grid_search.best_index_], 
                         grid_search.cv_results_['mean_test_cost'][grid_search.best_index_], 
                         grid_search.cv_results_['mean_test_precision'][grid_search.best_index_]]
    
    print(results_df)
    print('Affichage de la courbe Roc après optimisation')
    print(create_roc_auc_plot(best_lr_model,X_test, y_test))
    
    print('Affichage du rapport de classification après optimisation')
    print(create_classification_report(best_lr_model,y_test,y_pred_optimal_best))
    
    print('Affichage de la matrice de confusion après optimisation')
    print(create_confusion_matrix_plot(best_lr_model, X_test, y_test))  
    
    return y_pred_optimal, y_proba, logreg_model, y_pred_optimal_best, y_proba_best, best_lr_model, results, results_df

### Fonction qui permet de créer une expérience sur MlFlow

def create_experiment(experiment_name, run_metrics,model, 
                      confusion_matrix_path = None, roc_auc_plot_path = None, 
                      classification_report_plot_path =   None,feature_importance_path=None,
                      run_params=None, mode='undersampling'):
        
    mlflow.set_tracking_uri("http://localhost:5000") 
    mlflow.set_experiment(experiment_name)
    
    ## Define the name of our run
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    name = experiment_name +'_'+ mode +'_'+ dt_string
    
    with mlflow.start_run(run_name=experiment_name+'_'+ mode ):
        
        if not run_params == None:
            for param in run_params:
                mlflow.log_param(param, run_params[param])
            
        for metric in run_metrics:
            mlflow.log_metric(metric, run_metrics[metric])
        
        mlflow.sklearn.log_model(model, "model")
        
        if not confusion_matrix_path == None:
            mlflow.log_artifact(confusion_matrix_path, 'confusion_matrix.png')
            
        if not classification_report_plot_path == None:
            mlflow.log_artifact(classification_report_plot_path, 'classification report.txt')
            
        if not roc_auc_plot_path == None:
            mlflow.log_artifact(roc_auc_plot_path, "roc_auc_plot.png")
        
        if not feature_importance_path == None:
            mlflow.log_artifact(feature_importance_path, "feature_importance.png")
        
        mlflow.set_tag("tag1", experiment_name)
        
            
    print('Run - %s is logged to Experiment - %s' %(name +'_'+ mode, experiment_name))