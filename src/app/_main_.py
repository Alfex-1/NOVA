# Imports
# from tools.utils import *
import numpy as np
import pandas as pd
import modin.pandas as mpd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PowerTransformer
from scipy import stats
from scipy.stats.mstats import winsorize
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split, cross_validate, learning_curve
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.metrics import confusion_matrix
import io
from io import BytesIO
import os
import streamlit as st
from PIL import Image


def correlation_missing_values(df: pd.DataFrame):
    """
    Analyse la corrélation entre les valeurs manquantes dans un DataFrame.

    Cette fonction identifie les colonnes contenant des valeurs manquantes, 
    calcule la proportion de NaN par colonne et retourne la matrice de corrélation 
    des valeurs manquantes.

    Args:
        df (pd.DataFrame): Le DataFrame à analyser.

    Returns:
        tuple: 
            - pd.DataFrame : Matrice de corrélation (%) des valeurs manquantes entre les colonnes.
            - pd.DataFrame : Proportion des valeurs manquantes par colonne avec le type de variable.
    """
    # Filtrer les colonnes avec des valeurs manquantes
    df_missing = df.iloc[:, [i for i, n in enumerate(np.var(df.isnull(), axis=0)) if n > 0]]
    
    # Calculer la proportion de valeurs manquantes et ajouter le type de variable
    prop_nan = pd.DataFrame({
        "NaN proportion": round(df.isnull().sum() / len(df) * 100,2),
        "Type": df.dtypes.astype(str)
    })

    # Calculer la matrice de corrélation des valeurs manquantes
    corr_mat = round(df_missing.isnull().corr() * 100, 2)

    return corr_mat, prop_nan

def encode_data(df: pd.DataFrame,
                list_binary: list[str] = None,
                list_ordinal: list[str] = None,
                list_nominal: list[str] = None,
                ordinal_mapping: dict[str, dict[str, int]] = None) -> pd.DataFrame:
    """
    Encode les variables catégorielles du DataFrame selon leur nature.
    - Binaire : OneHotEncoder avec drop='if_binary' (une seule variable conservée)
    - Ordinale : mapping dict ou OrdinalEncoder
    - Nominale : OneHotEncoder classique (toutes les modalités)
    
    Returns:
        DataFrame encodé
    """
    df = df.copy()

    # Binaire
    if list_binary:
        encoder = OneHotEncoder(drop='if_binary', sparse_output=False)
        encoded_array = encoder.fit_transform(df[list_binary])
        encoded_cols = encoder.get_feature_names_out(list_binary)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
        df.drop(columns=list_binary, inplace=True)
        df = mpd.concat([df, encoded_df], axis=1)

    # Ordinal
    if list_ordinal:
        for col in list_ordinal:
            if ordinal_mapping and col in ordinal_mapping:
                df[col] = df[col].map(ordinal_mapping[col])
            else:
                encoder = OrdinalEncoder()
                df[col] = encoder.fit_transform(df[[col]])

    # Nominal
    if list_nominal:
        encoder = OneHotEncoder(drop=None, sparse_output=False)
        encoded_array = encoder.fit_transform(df[list_nominal])
        encoded_cols = encoder.get_feature_names_out(list_nominal)
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df.index)
        df.drop(columns=list_nominal, inplace=True)
        df = mpd.concat([df, encoded_df], axis=1)

    return df

def encode_data1(df: pd.DataFrame, list_binary: list[str] = None, list_ordinal: list[str]=None, list_nominal: list[str]=None, ordinal_mapping: dict[str, int]=None):
    """
    Encode les variables catégorielles d'un DataFrame selon leur nature (binaire, ordinale, nominale).

    - **Binaire** : One-Hot Encoding
    - **Ordinal** : Encodage en respectant un ordre défini (via un mapping)
    - **Nominal** : One-Hot Encoding

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données à encoder.
        list_binary (list, optional): Liste des colonnes binaires à encoder en One-Hot. Defaults to None.
        list_ordinal (list, optional): Liste des colonnes ordinales à encoder. Defaults to None.
        list_nominal (list, optional): Liste des colonnes nominales à encoder en One-Hot. Defaults to None.
        ordinal_mapping (dict, optional): Dictionnaire contenant le mapping des valeurs ordinales sous la forme 
            {'colonne': {'valeur1': 0, 'valeur2': 1, ...}}. Defaults to None.

    Returns:
        pd.DataFrame: Le DataFrame encodé avec les transformations appliquées.
    """    
    # Encodage binaire (OneHot) pour les variables binaires
    if list_binary is not None and len(list_binary) > 0:
        onehot = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), list_binary)], 
                                  remainder='passthrough')
        df = onehot.fit_transform(df)
        df = pd.DataFrame(df, columns=onehot.get_feature_names_out(list_binary))
    
    # Encodage ordinal pour les variables ordinales
    if list_ordinal is not None and len(list_ordinal) > 0:
        for col in list_ordinal:
            if ordinal_mapping is not None and col in ordinal_mapping:
                # Appliquer le mapping d'ordinal
                df[col] = df[col].map(ordinal_mapping[col])
            else:
                # Si le mapping n'est pas fourni, utiliser OrdinalEncoder
                encoder = OrdinalEncoder(categories=[list(ordinal_mapping[col].keys())])
                df[col] = encoder.fit_transform(df[[col]])
    
    # Encodage non-ordinal (OneHot) pour les variables nominales
    if list_nominal is not None and len(list_nominal) > 0:
        onehot = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), list_nominal)], 
                                remainder='passthrough')
        df = onehot.fit_transform(df)
        
        # Obtenir les nouveaux noms de colonnes et supprimer le préfixe 'onehot__'
        new_columns = onehot.get_feature_names_out()
        new_columns = [col.replace('onehot__', '') for col in new_columns]
        
        df = pd.DataFrame(df, columns=new_columns)
        df.index = range(len(df))
            
    return df

def impute_missing_values(df: pd.DataFrame, corr_info: pd.DataFrame, prop_nan: pd.DataFrame):
    """
    Impute les valeurs manquantes dans un DataFrame en fonction des proportions et des corrélations.

    Cette fonction impute automatiquement les valeurs manquantes pour chaque variable du DataFrame
    en analysant la proportion de valeurs manquantes et la force moyenne de corrélation (en valeur absolue)
    entre les variables. Selon ces indicateurs, différentes stratégies d'imputation sont appliquées :
      - Pour les variables avec moins de 10% de valeurs manquantes :
          • Si la force de corrélation est très faible (< 2), on utilise SimpleImputer avec la stratégie
            'most_frequent' pour les variables de type 'object' ou 'median' pour les variables numériques.
          • Si la force de corrélation est modérée (entre 2 et 5), on applique KNNImputer.
          • Si la force de corrélation est élevée (entre 5 et 7), on utilise IterativeImputer avec des paramètres par défaut.
          • Si la force de corrélation est très élevée (≥ 7), on emploie IterativeImputer avec un RandomForestRegressor comme estimateur.
      - Pour les variables dont la proportion de valeurs manquantes est comprise entre 10% et 65%, des stratégies similaires
        sont utilisées avec des paramètres ajustés.
      - Les variables ayant plus de 65% de valeurs manquantes sont supprimées du DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée contenant des valeurs manquantes.
        corr_info (pd.DataFrame): Une matrice de corrélation des patterns de valeurs manquantes (exprimée en pourcentage),
                                  telle que générée par une fonction comme `correlation_missing_values`.
        prop_nan (pd.DataFrame): Un DataFrame indiquant le pourcentage de valeurs manquantes et le type de données pour chaque variable,
                                 généralement obtenu via `correlation_missing_values`.

    Returns:
        pd.DataFrame: Un nouveau DataFrame avec les valeurs manquantes imputées selon les stratégies définies.
    """   
    df_imputed = df.copy()
    
    # Détection automatique des variables à imputer
    variables_a_imputer = prop_nan[prop_nan["NaN proportion"] > 0].index.tolist()

    for var in variables_a_imputer:
        taux_nan = prop_nan.loc[var, "NaN proportion"]
        type_ = prop_nan.loc[var, "Type"]
        corr_strength = corr_info[var].abs().mean()  # Moyenne des corrélations absolues avec les autres variables
        
        # Paramètres pour IterativeImputer
        max_iter, tol = 500, 1e-3

        if taux_nan < 10:
            if corr_strength < 2:
                if type_ == 'object':
                    imputer = SimpleImputer(strategy='most_frequent')
                else:
                    imputer = SimpleImputer(strategy='median')
            
            elif corr_strength < 5 and corr_strength >= 2:
                imputer = KNNImputer(n_neighbors=4, weights='distance')
            
            elif corr_strength < 7 and corr_strength >= 5:
                imputer = IterativeImputer(max_iter=max_iter, tol=tol, random_state=42)
            
            else:
                estimator = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                imputer = IterativeImputer(estimator=estimator,max_iter=max_iter, tol=tol, random_state=42)
            
            df_imputed[[var]] = imputer.fit_transform(df_imputed[[var]])
            if type_ == 'object' and taux_nan >= 10 and corr_strength >= 2:
                    df_imputed[[var]] = df_imputed[[var]].round(0).astype(int)
        
        elif taux_nan >= 10 and taux_nan < 65:
            if corr_strength < 5:
                imputer = IterativeImputer(max_iter=max_iter, tol=tol, random_state=42)
            
            elif corr_strength < 7 and corr_strength >= 5:
                estimator = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
                imputer = IterativeImputer(estimator=estimator,max_iter=max_iter, tol=tol, random_state=42)
                    
            else:
                estimator = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
                imputer = IterativeImputer(estimator=estimator,max_iter=max_iter, tol=tol, random_state=42)
            
            df_imputed[[var]] = imputer.fit_transform(df_imputed[[var]])
            if type_ == 'object' and taux_nan >= 10 and corr_strength >= 2:
                        df_imputed[[var]] = df_imputed[[var]].round(0).astype(int)
        
        else:
            df = df.drop(var, axis=1)        

    return df_imputed

def detect_outliers_iforest_lof(df: pd.DataFrame, target: str):
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    
    # Index des lignes valides (sans NaN dans les colonnes numériques)
    valid_rows = df_numeric.dropna().index  
    df_numeric = df_numeric.loc[valid_rows]  

    if df_numeric.shape[1] == 0:
        raise ValueError("Aucune colonne numérique dans le DataFrame.")

    # Détection des outliers avec Isolation Forest
    iforest = IsolationForest(n_estimators=500, contamination='auto', random_state=42, n_jobs=-1)
    df.loc[valid_rows, 'outlier_iforest'] = iforest.fit_predict(df_numeric)
    df['outlier_iforest'] = np.where(df['outlier_iforest'] == -1, 1, 0)

    # Détection avec Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    outliers_lof = lof.fit_predict(df_numeric)  # Génère uniquement pour les lignes valides

    # Assignation correcte des valeurs LOF
    df.loc[valid_rows, 'outlier_lof'] = np.where(outliers_lof == -1, 1, 0)

    # Fusion des résultats
    df['outlier'] = ((df['outlier_iforest'] == 1) & (df['outlier_lof'] == 1)).astype(int)
    df = df.drop(columns=['outlier_lof', 'outlier_iforest'])

    # Suppression ou remplacement des outliers
    seuil = max(0.05 * len(df), 1)  # Toujours au moins 1
    if df['outlier'].sum() > seuil:
        for col in df_numeric.columns:
            mask_outliers = df['outlier'] == 1
            mask_non_outliers = df['outlier'] == 0

            mean = df.loc[mask_non_outliers, col].mean()
            std = df.loc[mask_non_outliers, col].std()
            if std == 0:
                std = 1e-6  # Évite les valeurs identiques

            df.loc[mask_outliers, col] = np.random.normal(mean, std, mask_outliers.sum())
    else:
        df = df[df['outlier'] == 0].drop(columns='outlier')
    
    return df

def detect_outliers_iforest_lof(df: pd.DataFrame, target: str):
    """
    Detects outliers in a dataset using Isolation Forest and Local Outlier Factor (LOF) methods.
    The outliers are identified based on the combination of results from both methods, and can be either replaced using winsorization or removed based on a predefined threshold.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data, including both numerical features and the target variable.
        target (str): The name of the target variable column in the DataFrame, which will be excluded from outlier detection.

    Raises:
        ValueError: If the DataFrame does not contain any numeric columns after removing the target variable or has missing data.

    Returns:
        pd.DataFrame: The input DataFrame with outliers flagged and handled (either winsorized or removed). A new column 'outlier' will indicate the final outliers after combining the results of both methods.
    """
    
    # Sélectionner uniquement les colonnes numériques
    df_numeric = df.select_dtypes(include=[np.number]).drop(columns=[target], errors='ignore')
    
    # Index des lignes valides (sans NaN dans les colonnes numériques)
    valid_rows = df_numeric.dropna().index  
    df_numeric = df_numeric.loc[valid_rows]  

    if df_numeric.shape[1] == 0:
        raise ValueError("Aucune colonne numérique dans le DataFrame.")

    # Détection des outliers avec Isolation Forest
    iforest = IsolationForest(n_estimators=500, contamination='auto', random_state=42, n_jobs=-1)
    df.loc[valid_rows, 'outlier_iforest'] = iforest.fit_predict(df_numeric)
    df['outlier_iforest'] = np.where(df['outlier_iforest'] == -1, 1, 0)
    
    # Détection avec Local Outlier Factor
    lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    outliers_lof = lof.fit_predict(df_numeric)  # Génère uniquement pour les lignes valides
    df.loc[valid_rows, 'outlier_lof'] = np.where(outliers_lof == -1, 1, 0)

    # Fusion des résultats
    df['outlier'] = ((df['outlier_iforest'] == 1) & (df['outlier_lof'] == 1)).astype(int)
    df = df.drop(columns=['outlier_lof', 'outlier_iforest'])

    # Calcul de la proportion d'outliers détectés par les deux algos
    proportion_outliers = df['outlier'].sum() / len(df)

    # Suppression ou remplacement des outliers (winsorisation)
    seuil = max(0.01 * len(df), 1)
    if df['outlier'].sum() > seuil:
        for col in df_numeric.columns:
            # Appliquer la winsorisation aux outliers avec la proportion calculée
            df[col] = winsorize(df[col], limits=[proportion_outliers, proportion_outliers])  # Limites ajustées selon proportion_outliers
    else:
        df = df[df['outlier'] == 0]
    
    df = df.drop(columns='outlier')
        
    return df

def transform_data(split_data: bool, df: pd.DataFrame = None, df_train: pd.DataFrame = None, df_test: pd.DataFrame = None, list_boxcox: list[str] = None, list_yeo: list[str] = None, list_log: list[str] = None, list_sqrt: list[str] = None):
    """
    Applique des transformations statistiques (Box-Cox, Yeo-Johnson, Logarithme, Racine carrée) sur les colonnes spécifiées,
    avec gestion optionnelle des ensembles d'entraînement et de test pour éviter toute fuite de données.

    Args:
        split_data (bool): 
            Indique si les données sont séparées en df_train et df_test. Si False, 'df' est utilisé pour transformation globale.
        df (pd.DataFrame, optional): 
            DataFrame complet à transformer si split_data=False. Ignoré sinon.
        df_train (pd.DataFrame, optional): 
            DataFrame d'entraînement à transformer si split_data=True.
        df_test (pd.DataFrame, optional): 
            DataFrame de test à transformer si split_data=True.
        list_boxcox (list[str], optional): 
            Liste des colonnes sur lesquelles appliquer la transformation de Box-Cox (valeurs strictement positives).
        list_yeo (list[str], optional): 
            Liste des colonnes sur lesquelles appliquer la transformation de Yeo-Johnson (valeurs quelconques).
        list_log (list[str], optional): 
            Liste des colonnes à transformer via logarithme naturel (valeurs strictement positives).
        list_sqrt (list[str], optional): 
            Liste des colonnes à transformer via racine carrée (valeurs ≥ 0).

    Returns:
        pd.DataFrame or tuple(pd.DataFrame, pd.DataFrame): 
            - Si split_data=False : retourne le DataFrame transformé (df).
            - Si split_data=True : retourne un tuple (df_train, df_test) transformés.
    """
    # Box-Cox Transformation
    if list_boxcox and len(list_boxcox) > 0:
        transformer_bc = PowerTransformer(method='box-cox')
        
        if split_data:
            transformer_bc.fit(df_train[list_boxcox])
            df_train[list_boxcox] = transformer_bc.transform(df_train[list_boxcox])
            df_test[list_boxcox] = transformer_bc.transform(df_test[list_boxcox])
        else:
            df[list_boxcox] = transformer_bc.fit_transform(df[list_boxcox])

    # Yeo-Johnson Transformation
    if list_yeo and len(list_yeo) > 0:
        transformer_yeo = PowerTransformer(method='yeo-johnson')
        
        if split_data:
            transformer_yeo.fit(df_train[list_yeo])
            df_train[list_yeo] = transformer_yeo.transform(df_train[list_yeo])
            df_test[list_yeo] = transformer_yeo.transform(df_test[list_yeo])
        else:
            df[list_yeo] = transformer_yeo.fit_transform(df[list_yeo])
    
    # Logarithmic Transformation
    if list_log and len(list_log) > 0:
        if split_data:
            for col in list_log:
                df_train[col] = np.log(df_train[col])
                df_test[col] = np.log(df_test[col])
        else:        
            for col in list_log:
                df[col] = np.log(df[col])

    # Square Root Transformation
    if list_sqrt and len(list_sqrt) > 0:
        if split_data:
            for col in list_sqrt:
                df_train[col] = np.sqrt(df_train[col])
                df_test[col] = np.sqrt(df_test[col])
        else:
            for col in list_sqrt:
                df[col] = np.sqrt(df[col])
    
    if split_data:
        return df_train, df_test
    else:
        return df

def calculate_inertia(X):
    """
    Calcule l'inertie (variance expliquée) de la dernière composante principale ajoutée à chaque étape
    de l'ACP, en augmentant progressivement le nombre de composantes.

    Args:
        X (np.ndarray or pd.DataFrame): 
            Matrice des données (features uniquement), à transformer via ACP.

    Returns:
        list[float]: 
            Liste des pourcentages de variance expliquée par la dernière composante ajoutée à chaque itération 
            (de 1 à n_features).
    """
    inertias = []
    for i in range(1, X.shape[1] + 1):
        pca = PCA(n_components=i)
        pca.fit(X)
        inertias.append(pca.explained_variance_ratio_[-1]*100)  # La dernière composante expliquée à chaque étape
    return inertias

def objective(trial, task="regression", model_type="Random Forest", multi_class=False, X=None, y=None, cv=5, scoring_comp='neg_root_mean_squared_error'):
    # Paramètres d'optimisation selon le type de modèle
    if model_type == "LightGBM":
        param = {
            'objective': 'regression' if task == 'Regression' else 'binary' if task == 'Classification' else 'multiclass',
            'metric': 'rmse' if task == 'Regression' else 'binary_logloss' if task == 'Classification' else 'multi_logloss',
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.01),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.01),
        }
        model = lgb.LGBMRegressor(**param, verbose=-1, n_jobs=-1) if task == 'Regression' else lgb.LGBMClassifier(**param, verbose=-1, n_jobs=-1)

    elif model_type == "XGBoost":
        # Déterminer l'objectif et les métriques selon la tâche
        if task == 'Regression':
            objective = 'reg:squarederror'
            eval_metric = 'rmse'
        elif task == 'Classification':
            objective = 'binary:logistic'  # pour la classification binaire
            eval_metric = 'logloss'
            num_class = 1  # binaire, pas nécessaire de définir num_class ici
        else:  # pour la classification multiclasse
            objective = 'multi:softmax'
            eval_metric = 'mlogloss'
            num_class = len(y_train.unique())  # Nombre de classes unique dans la variable cible

        # Hyperparamètres à optimiser
        param = {
            'objective': objective,
            'eval_metric': eval_metric,
            'max_depth': trial.suggest_int('max_depth', 2, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.01),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0, step=0.01),
        }

        # Ajouter le paramètre num_class pour la classification multiclasse
        if task == 'Classification' and objective == 'multi:softmax':
            param['num_class'] = num_class

        # Créer le modèle avec les meilleurs paramètres
        if task == 'Regression':
            model = xgb.XGBRegressor(**param, n_jobs=-1)
        else:
            model = xgb.XGBClassifier(**param, n_jobs=-1)

    elif model_type == "Random Forest":
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
        }
        model = RandomForestRegressor(**param, n_jobs=-1) if task == 'Regression' else RandomForestClassifier(**param, n_jobs=-1)

    elif model_type == "SVM":
        param = {
            'C': trial.suggest_float('C', 0.1, 10.0, log=True),
            'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
            'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
            'degree': trial.suggest_int('degree', 2, 5),
        }
        model = SVC(**param) if task == 'Classification' else SVR(**param)

    elif model_type == "Linear Regression":
        # Définition des hyperparamètres pour la régression linéaire et les modèles régularisés
        model_linreg = trial.suggest_categorical("model", ["linear", "ridge", "lasso", "elasticnet"])
    
        if model_linreg == "linear":
            model = LinearRegression()
        
        elif model_linreg == "ridge":
            ridge_alpha = trial.suggest_float("ridge_alpha", 0.01, 10.01, log=True)
            ridge_alpha = round(ridge_alpha, 2)
            model = Ridge(alpha=ridge_alpha)

        elif model_linreg == "lasso":
            lasso_alpha = trial.suggest_float("lasso_alpha", 0.01, 10.01, log=True)
            lasso_alpha = round(lasso_alpha, 2)
            model = Lasso(alpha=lasso_alpha)

        elif model_linreg == "elasticnet":
            enet_alpha = trial.suggest_float("enet_alpha", 0.01, 10.01, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0, 1.0, step=0.01)
            enet_alpha = round(enet_alpha, 2)
            l1_ratio = round(l1_ratio, 2)
            model = ElasticNet(alpha=enet_alpha, l1_ratio=l1_ratio)

    elif model_type == "Logistic Regression":
        # Paramètres pour la régression logistique
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet", None])
        C = trial.suggest_float("C", 1e-3, 10.001, step=0.01)
        C = round(C, 3)
        
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0, 1, step=0.01)
            l1_ratio = round(l1_ratio, 2)
            model = LogisticRegression(penalty=penalty, C=C, solver='saga', l1_ratio=l1_ratio, max_iter=10000, n_jobs=-1)
        elif penalty == "l1":
            solver = "saga" if multi_class else "liblinear"
            model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10000, n_jobs=-1)
        elif penalty == "l2":
            solver = "saga" if multi_class else "lbfgs"
            model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10000, n_jobs=-1)
        else:
            solver = "saga" if multi_class else "lbfgs"
            model = LogisticRegression(penalty=penalty, solver=solver, max_iter=10000, n_jobs=-1)


    elif model_type == "KNN":
        param = {
            'n_neighbors': trial.suggest_int('n_neighbors', 3, 50),
            'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
            'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute']),
            'leaf_size': trial.suggest_int('leaf_size', 10, 50),
        }
        model = KNeighborsRegressor(**param, n_jobs=-1) if task == 'Regression' else KNeighborsClassifier(**param, n_jobs=-1)

    # Validation croisée pour évaluer le modèle
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring_comp, return_train_score=False)

    # Retourner la performance (ici on maximise la précision, mais à ajuster selon le modèle)
    return np.mean(cv_results['test_score'])

def optimize_model(model_choosen, task: str, X_train: pd.DataFrame, y_train: pd.Series, cv: int = 10, scoring: str = "neg_root_mean_quared_error", multi_class: bool = False, n_trials: int = 70, n_jobs: int = -1):
    study = optuna.create_study(direction='maximize', sampler=TPESampler(n_startup_trials=15), pruner=HyperbandPruner())
    study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen, multi_class=multi_class, X=X_train, y=y_train, cv=cv, scoring_comp=scoring), n_trials=n_trials, n_jobs=n_jobs)
    
    # Créer le modèle avec les meilleurs hyperparamètres
    if model_choosen == "LightGBM":
        best_model = lgb.LGBMRegressor(**study.best_params, verbose=-1) if task == 'Regression' else lgb.LGBMClassifier(**study.best_params, verbose=-1)
    elif model_choosen == "XGBoost":
        best_model = xgb.XGBRegressor(**study.best_params) if task == 'Regression' else xgb.XGBClassifier(**study.best_params)
    elif model_choosen == "Random Forest":
        best_model = RandomForestRegressor(**study.best_params) if task == 'Regression' else RandomForestClassifier(**study.best_params)
    elif model_choosen == "SVM":
        best_model = SVR(**study.best_params) if task == 'Regression' else SVC(**study.best_params)
    elif model_choosen == "Linear Regression":
        # Gestion des régressions linéaires et régularisées
        if "model" in study.best_params and study.best_params["model"] == "linear":
            best_model = LinearRegression()
        elif "model" in study.best_params and study.best_params["model"] == "ridge":
            best_model = Ridge(alpha=study.best_params["ridge_alpha"])
        elif "model" in study.best_params and study.best_params["model"] == "lasso":
            best_model = Lasso(alpha=study.best_params["lasso_alpha"])
        elif "model" in study.best_params and study.best_params["model"] == "elasticnet":
            best_model = ElasticNet(alpha=study.best_params["enet_alpha"], l1_ratio=study.best_params["l1_ratio"])
    elif model_choosen == "Logistic Regression":
        best_model = LogisticRegression(**study.best_params, max_iter=10000, n_jobs=-1)
    elif model_choosen == "KNN":
        best_model = KNeighborsRegressor(**study.best_params) if task == 'Regression' else KNeighborsClassifier(**study.best_params)
    
    # Retourner le modèle avec les meilleurs hyperparamètres et les résultats
    best_params = study.best_params
    best_value = study.best_value
    
    return best_model, best_params, best_value

def bias_variance_decomp(estimator, X, y, task, cv=5, random_seed=None):
    """Calcule le biais et la variance d'un estimateur via une décomposition par validation croisée.

    Cette fonction effectue une décomposition du biais et de la variance d'un modèle d'estimation en utilisant 
    la validation croisée (KFold). Elle permet d'évaluer la performance de l'estimateur en termes de biais et 
    de variance en fonction de la tâche (régression ou classification).

    Args:
        estimator (sklearn.base.BaseEstimator): L'estimateur (modèle) à évaluer, par exemple, une régression 
                                                linéaire, un classifieur SVM, etc.
        X (array-like, shape (n_samples, n_features)): Matrices de caractéristiques, où chaque ligne est une 
                                                      observation et chaque colonne est une caractéristique.
        y (array-like, shape (n_samples,)): Vecteur ou matrice des valeurs cibles (vérités terrain), qui 
                                            varient en fonction de la tâche (régression ou classification).
        task (str): Type de tâche, soit "Classification", soit "Regression". Cela détermine le calcul du biais 
                    et de la variance.
        cv (int, optional): Nombre de divisions (splits) pour la validation croisée. Par défaut à 5.
        random_seed (int, optional): Seed pour le générateur aléatoire, utile pour reproduire les résultats. 
                                     Par défaut à None.

    Returns:
        tuple: Un tuple contenant les valeurs suivantes :
            - avg_expected_loss (float): Perte moyenne (erreur quadratique moyenne pour la régression, erreur 
                                          de classification pour la classification).
            - avg_bias (float): Biais moyen (écart moyen entre les prédictions et les valeurs réelles).
            - avg_var (float): Variance moyenne des prédictions.
            - bias_relative (float): Biais relatif, normalisé par rapport à l'écart-type de la cible (régression) 
                                     ou au nombre de classes (classification).
            - var_relative (float): Variance relative des prédictions par rapport à la variance de la cible 
                                     (régression) ou au nombre de classes (classification).
    """
    # Vérifier si 'loss' est un DataFrame ou une série et en extraire la valeur
    rng = np.random.RandomState(random_seed)
    kf = KFold(n_splits=cv, shuffle=True, random_state=rng)

    all_pred = []
    y_tests = []

    for train_idx, test_idx in kf.split(X):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        model = estimator.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_test_fold)
        
        all_pred.append(preds)
        y_tests.append(y_test_fold)

    all_pred = np.concatenate(all_pred)
    y_tests = np.concatenate(y_tests)

    if task == "Classification":
        # Classification : calcul de la majorité des prédictions (mode)
        main_predictions = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=0, arr=all_pred.astype(int))
        avg_expected_loss = np.mean(all_pred != y_tests)
        avg_bias = np.mean(main_predictions != y_tests)
        avg_var = np.mean((all_pred != main_predictions).astype(int))

        # Calcul du biais et de la variance relatifs
        bias_relative = np.mean(main_predictions != y_tests) / len(np.unique(y_tests))  # Par rapport au nombre de classes
        var_relative = np.mean((all_pred != main_predictions).astype(int)) / len(np.unique(y_tests))  # Par rapport au nombre de classes
        
    else:
        # Régression : calcul de la moyenne des prédictions
        main_predictions = np.mean(all_pred, axis=0)
        avg_expected_loss = np.mean((all_pred - y_tests) ** 2)
        avg_bias = np.mean(main_predictions - y_tests)
        avg_var = np.mean((all_pred - main_predictions) ** 2)

        # Calcul du biais et de la variance relatifs
        bias_relative = np.mean(main_predictions - y_tests) / np.std(y_tests)  # Par rapport à l'écart-type de y
        var_relative = np.mean((all_pred - main_predictions) ** 2) / np.var(y_tests)  # Par rapport à la variance de y

    return avg_expected_loss, avg_bias, avg_var, bias_relative, var_relative

def instance_model(index, df, task):
    # Récupérer le nom du modèle depuis df_train
    model_name = index
    
    # Récupérer les hyperparamètres pour ce modèle
    best_params = df.loc[index, 'Best Params']
    
    # Déterminer l'instance du modèle selon le nom et la tâche (Classification ou Regression)
    if model_name == 'KNN':
        if task == 'Classification':
            model = KNeighborsClassifier(**best_params)  # KNNClassifier
        else:
            model = KNeighborsRegressor(**best_params)  # KNNRegressor
            
    elif model_name == 'LightGBM':
        if task == 'Classification':
            model = lgb.LGBMClassifier(**best_params)  # LGBMClassifier
        else:
            model = lgb.LGBMRegressor(**best_params)  # LGBMRegressor
            
    elif model_name == 'XGBoost':
        if task == 'Classification':
            model = xgb.XGBClassifier(**best_params)  # XGBClassifier
        else:
            model = xgb.XGBRegressor(**best_params)  # XGBRegressor
            
    elif model_name == 'Random Forest':
        if task == 'Classification':
            model = RandomForestClassifier(**best_params)  # RandomForestClassifier
        else:
            model = RandomForestRegressor(**best_params)  # RandomForestRegressor
            
    elif model_name == 'Linear Regression':
        type_model_reg = best_params.get('model')
        
        if type_model_reg == 'linear':
            model = LinearRegression()
        elif type_model_reg == 'ridge': 
            alpha = best_params.get('ridge_alpha')
            model = Ridge(alpha=alpha)
        elif type_model_reg == 'lasso':
            alpha = best_params.get('lasso_alpha')
            model = Lasso(alpha=alpha)
        elif type_model_reg == 'elasticnet':
            alpha = list(best_params.values())[1]
            l1_ratio = list(best_params.values())[2]
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(*best_params)
           
    elif model_name == 'SVM':
        if task == 'Classification':
            model = SVC(**best_params)  # SVC pour classification
        else:
            model = SVR(**best_params)  # SVR pour régression
    
    return model

# python -m streamlit run src/app/_main_.py
st.set_page_config(page_title="NOVA", layout="wide")

st.title("✨ NOVA : Numerical Optimization & Validation Assistant")
st.subheader("Votre assistant flexible pour le traitement des données et la modélisation.")

st.write(
    """
    **NOVA** vous accompagne dans la préparation et l’optimisation de vos modèles de machine learning.  
    Conçue pour les professionnels qui savent que chaque projet est unique, **NOVA** offre des outils puissants
    pour la gestion des données et l'ajustement des modèles, tout en laissant l'exploration et la personnalisation à votre charge.

    **Fonctionnalités principales :**
    - 🔄 **Prétraitement des données** : mise à l’échelle, encodage, gestion des valeurs manquantes, outliers, et transformations adaptées.
    - 🔍 **Optimisation des hyperparamètres** : recherche des meilleurs réglages pour 7 modèles populaires (régression linéaire/logistique, KNN, SVM, Random Forest, LightGBM, XGboost).
    - 🏆 **Évaluation des modèles** : validation croisée, analyse biais-variance, importance par permutation, analyse de drift et matrice de confusion pour les tâches de classification.
    
    **NOVA** permet à chaque utilisateur de bénéficier d’une infrastructure robuste, tout en maintenant une flexibilité totale sur le traitement fondamental des données.
    Vous contrôlez les choix, nous optimisons les outils.
    """
)

# Chargement des données
df = None
uploaded_file = st.file_uploader("Choississez un fichier (csv, xlsx et  txt acceptés seulement)", type=["csv", "xlsx", "txt"])
wrang = st.checkbox("La base de données nécessite un traitement")
valid_mod=False
valid_wrang=False

if uploaded_file is not None:
    byte_data = uploaded_file.read()
    separators = [";", ",", "\t"]
    df = None
    detected_sep = None

    for sep in separators:
        try:
            tmp_df = pd.read_csv(BytesIO(byte_data), sep=sep, engine="python", nrows=20)
            if tmp_df.shape[1] > 1:
                detected_sep = sep
                break
        except Exception:
            continue

    if detected_sep is not None:
        df = pd.read_csv(BytesIO(byte_data), sep=detected_sep)  # Chargement complet proprement
    else:
        st.warning("Échec de la détection du séparateur. Vérifiez le format du fichier.")

# Sidebar pour la configuration de l'utilisateur    
if df is not None:
    st.sidebar.image(Image.open("logo_nova.png"), width=200)
    
    if wrang is True:            
        st.sidebar.title("Paramètres du traitement des données")
        
        # Demander s'il faut demander de diviser la base
        split_data = st.sidebar.checkbox("Diviser la base de données en apprentissage/validation ?", value=True, help="La division des données durant leur traitement est fondamentale pour éviter la fuite de données lors de votre modélisation.")
        
        if split_data:
            test_size = st.sidebar.slider("Proportion des données utilisées pour l'apprentissage des modèles (en %)", min_value=50, max_value=90, value=75)
            test_size=test_size/100
            
            df_train, df_test = train_test_split(df, test_size=test_size, shuffle=True, random_state=42)
        
        # Demander la variable cible pour la modélisation
        target = st.sidebar.selectbox("Choisissez la variable cible de votre modélisation", df.columns.to_list(), help="Si vous n'avez pas de variable cible, choisissez une variable au harsard.")
        pb = False
        wrang_finished = False
        
        st.sidebar.subheader("Mise à l'échelle des variables numériques")
        
        # Déterminer si la variable cible doit être incluse dans la mise à l'échelle
        use_target = st.sidebar.checkbox("Inclure la variable cible dans la mise à l'échelle", value=False, help="Cochez cette case si vous ne souhaotez pas inclure une variable dans le traitement.")
        drop_dupli = st.sidebar.checkbox("Supprimer toutes les observations dupliquées", value=False)
        
        if drop_dupli:
            df.drop_duplicates()
        
        if not use_target:
            df_copy=df.copy().drop(columns=target)
        
        # Tout mettre à l'échelle directement
        scale_all_data = st.sidebar.checkbox("Voulez-vous mettre à l'échelle vos données ?")
        
        if scale_all_data:
            scale_method = st.sidebar.selectbox("Méthode de mise à l'échelle à appliquer",
                                                ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "Quantile Transformer (Uniform)"])
        
        # Obtenir des dataframes distinctes selon les types des données
        if not use_target:
            df_num = df_copy.select_dtypes(include=['number'])
            df_cat = df_copy.select_dtypes(exclude=['number'])
        else:
            df_num = df.select_dtypes(include=['number'])
            df_cat = df.select_dtypes(exclude=['number'])
        
        # Sélection des variables à encoder
        if df_cat.shape[1] > 0:
            st.sidebar.subheader("Encodage des variables catégorielles")
            list_binary = None
            list_nominal = None
            list_ordinal = None
            list_binary = st.sidebar.multiselect("Variables binaires", df_cat.columns.to_list())
            list_nominal = st.sidebar.multiselect("Variables nominales (non-ordinales)", df_cat.columns.to_list())
            list_ordinal = st.sidebar.multiselect("Variables ordinales", df_cat.columns.to_list())
            
            # Vérification dans les listes de mise à l'échelle
            encoding_vars = list_binary + list_nominal + list_ordinal

            # Vérifier les doublons dans les listes de mise à l'échelle
            duplicates_in_encoding = set([var for var in encoding_vars if encoding_vars.count(var) > 1])
            if duplicates_in_encoding:
                pb = True
                st.sidebar.warning(f"⚠️ Les variables suivantes sont présentes plusieurs fois dans les listes de variables à encoder : {', '.join(duplicates_in_encoding)}")
            
            # Création du mapping ordinal avec UI améliorée
            ordinal_mapping = {}

            if list_ordinal:
                st.sidebar.subheader("Mapping des variables ordinales")

                for var in list_ordinal:
                    unique_values = sorted(df_cat[var].dropna().unique().tolist())  # Trier les valeurs uniques
                    ordered_values = st.sidebar.multiselect(f"Ordre pour {var} (ordre croissant à spécifié)", unique_values)

                    # Vérification stricte : s'assurer que toutes les valeurs sont bien prises en compte
                    if set(ordered_values) == set(unique_values):
                        ordinal_mapping[var] = {val: idx for idx, val in enumerate(ordered_values)}
                    else:
                        st.sidebar.warning(f"⚠️ L'ordre défini pour {var} est incomplet ou contient des erreurs.")
                        
        st.sidebar.subheader("Transformation des données")
        
        # Transformation des variables (Box-Cox, Yeo-Johnson, Log, Sqrt)
        if not scale_all_data:
            # Déterminer les variables strcitement positives
            strictly_positive_vars = df_num.columns[(df_num > 0).all()].to_list()
            # Déterminer les variables positives ou nulles
            positive_or_zero_vars = df_num.columns[(df_num >= 0).all()].to_list()
            
            list_boxcox = None
            list_yeo = None
            list_log = None
            list_sqrt = None            
            list_boxcox = st.sidebar.multiselect("Variables à transformer avec Box-Cox", strictly_positive_vars)
            list_yeo = st.sidebar.multiselect("Variables à transformer avec Yeo-Johnson", df_num.columns.to_list())
            list_log = st.sidebar.multiselect("Variables à transformer avec le logarithme", strictly_positive_vars)
            list_sqrt = st.sidebar.multiselect("Variables à transformer avec la racine carrée", positive_or_zero_vars)
            
            # Vérification dans les listes de transformation
            transform_vars = list_boxcox + list_yeo + list_log + list_sqrt

            # Vérifier les doublons dans les listes de transformation
            duplicates_in_transform = set([var for var in transform_vars if transform_vars.count(var) > 1])
            if duplicates_in_transform:
                pb = True
                st.sidebar.warning(f"⚠️ Les variables suivantes sont présentes plusieurs fois dans les listes de variables à transformer : {', '.join(duplicates_in_transform)}")
                
        
        # Transformation de variables (ACP)
        use_pca = st.sidebar.checkbox("Utiliser l'Analyse en Composantes Principales (ACP)", value=False, help="⚠️ Il est fortement recommandé de mettre à l'échelle toutes les variables avec la même méthode avant d'appliquer l'ACP, au risque de la biaiser.")
        
        if use_pca:
            # Option pour spécifier le nombre de composantes ou la variance expliquée
            pca_option = st.sidebar.radio("Choisissez la méthode de sélection", ("Nombre de composantes", "Variance expliquée"))

            if pca_option == "Nombre de composantes":
                n_components = st.sidebar.slider("Nombre de composantes principales", min_value=1, max_value=15, value=3)
            elif pca_option == "Variance expliquée":
                explained_variance = st.sidebar.slider("Variance expliquée à conserver (%)", min_value=00, max_value=100, value=95)
        
        # Valider les choix
        valid_wrang = st.sidebar.button("Valider les choix de modélisation")
    
    else:
        # Modélisation
        st.sidebar.title("Paramètres de Modélisation")

        # Définition de la variable cible
        target = st.sidebar.selectbox("Choisissez la variable cible", df.columns.to_list())
        
        # Division des données
        test_size = st.sidebar.slider("Proportion des données utilisées pour l'apprentissage des modèles (en %)", min_value=50, max_value=90, value=75)
        test_size=test_size/100
        
        st.sidebar.subheader("Choix des modèles")

        # Sélection de la tâche (Classification ou Régression)
        task = st.sidebar.radio("Type de tâche", ["Classification", "Regression"])

        # Déterminer si la tâche est de classification multigroupe ou binaire
        if task == "Classification" and len(df[target].unique()) > 2:
            multi_class = True
        else:
            multi_class = False

        # Sélection des modèles
        if task == "Regression":
            models = st.sidebar.multiselect("Modèle(s) à tester", ["Linear Regression", "KNN", "SVM", "Random Forest", "XGBoost", "LightGBM"], default=["Linear Regression"])
        else:
            models = st.sidebar.multiselect("Modèle(s) à tester", ["Logistic Regression", "KNN", "SVM", "Random Forest", "XGBoost", "LightGBM"], default=["Logistic Regression"])
            
        # Sélection du critère de scoring
        metrics_regression = {
            "R² Score": "r2",
            "Mean Squared Error": "neg_mean_squared_error",
            "Root Mean Squared Error": "neg_root_mean_squared_error",
            "Mean Absolute Error": "neg_mean_absolute_error",
            "Mean Absolute Percentage Error": "neg_mean_absolute_percentage_error"
        }

        metrics_classification = {
            "Accuracy": "accuracy",
            "F1 Score (Weighted)": "f1_weighted",
            "Precision (Weighted)": "precision_weighted",
            "Recall (Weighted)": "recall_weighted"
        }

        if task == "Regression":
            scoring_comp = st.sidebar.selectbox(
                "Métrique pour la comparaison des modèles",
                list(metrics_regression.keys()))
            
            # Conversion en valeurs sklearn
            scoring_comp = metrics_regression[scoring_comp]

        else:
            scoring_comp = st.sidebar.selectbox(
                "Métrique pour la comparaison des modèles",
                list(metrics_classification.keys()))
            
            scoring_comp = metrics_classification[scoring_comp]
        
        st.sidebar.subheader("Critères d'évaluation")

        # Sélection des métriques selon la tâche
        if task == "Regression":
            scoring_eval = st.sidebar.multiselect(
                "Métrique(s) pour l'évaluation des modèles",
                list(metrics_regression.keys())
            )
            
            # Conversion en valeurs sklearn
            scoring_eval = [metrics_regression[m] for m in scoring_eval]

        else:
            scoring_eval = st.sidebar.multiselect(
                "Métrique(s) pour l'évaluation des modèles",
                list(metrics_classification.keys())
            )
            
            scoring_eval = [metrics_classification[m] for m in scoring_eval]
            
        # Saisie du nombre de folds pour la validation croisée
        # Checkbox pour activer le Leave-One-Out CV
        use_loocv = st.sidebar.checkbox("Utiliser une seule observation par évaluation (recommandé pour les petits ensembles de données uniquement)")

        # Si LOO-CV est coché, le champ des folds est désactivé
        if not use_loocv:
            cv = st.sidebar.number_input(
                "Nombre de folds (CV)",
                min_value=2, max_value=20,
                value=7, step=1,
                disabled=use_loocv)            
            
        # Demander le temps disponible selon les choix de l'utilisateur
        time_sup= st.sidebar.checkbox("Voulez vous prendre plus de temps pour améliorer les résultats ?")
        
        # Pondération de complexité selon les modèles
        complexity_weights = { "Linear Regression": 1, "Logistic Regression": 2,
                              "KNN": 4, "SVM": 6, "Random Forest": 8,
                              "LightGBM": 7, "XGBoost": 7}
        
        # Paramètres de base
        max_global_trials = 50
        max_trials_hard_limit = 150
        
        # Taille des données
        n_rows, n_cols = df.shape
        data_penalty = (n_rows / 1e6) * (n_cols / 5)

        # Complexité totale des modèles
        model_penalty = sum(complexity_weights.get(m, 3) for m in models)
        
        # Calcul du budget brut
        raw_score = data_penalty * model_penalty
        # Inversion du score pour définir le nombre d'essais
        trial = max_global_trials / (1 + raw_score)

        # Ajout de la préférence utilisateur
        if time_sup:
            trial = min(150, int(round(trial * 1.5)))
        trial = int(min(max_trials_hard_limit, max(5, round(trial))))
        
        repeats = trial
            
        st.sidebar.subheader("Enregistrement des modèles")
        # Demander à l'utilisateur où il souhaite enregistrer les modèles
        base_dir = st.sidebar.text_input("Entrez le chemin du dossier qui contiendra les modèles enregistrés", help="Exemple : C:\\Users\\Documents")
        
        # Valider les choix
        valid_mod = st.sidebar.button("Valider les choix de modélisation")

if valid_wrang:
    # 1. Analyser la corrélation des valeurs manquantes
    if not use_target:
        df = df.dropna(subset=[target])
    corr_mat, prop_nan = correlation_missing_values(df)

    # 2. Détecter les outliers
    df_outliers = detect_outliers_iforest_lof(df, target)

    # 3. Appliquer l'encodage des variables (binaire, ordinal, nominal)
    if df_cat.shape[1] > 0:
        df_encoded = encode_data(df_outliers, list_binary=list_binary, list_ordinal=list_ordinal, list_nominal=list_nominal, ordinal_mapping=ordinal_mapping)
    else:
        df_encoded = df_outliers.copy()
        
    # 4. Imputer les valeurs manquantes
    df_encoded = df_encoded.dropna(subset=[target])
    df_imputed = impute_missing_values(df_encoded, corr_mat, prop_nan)

    # 5. Mettre à l'échelle les données
    if scale_all_data:
        if scale_method:
            if scale_method == "Standard Scaler":
                scaler = StandardScaler()
            elif scale_method == "MinMax Scaler":
                scaler = MinMaxScaler()
            elif scale_method == "Robust Scaler":
                scaler = RobustScaler()
            else:
                scaler = QuantileTransformer(output_distribution='uniform')
        
            
            if not use_target:
                df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed.drop(columns=target)),
                                        columns=df_imputed.drop(columns=target).columns,
                                        index=df_imputed.index)
                df_scaled = pd.concat([df_scaled, df_imputed[target]], axis=1)
            else:
                df_scaled = pd.DataFrame(scaler.fit_transform(df_imputed), columns=df_imputed.columns)

        else:
            st.warning("⚠️ Veuillez sélectionner une méthode de mise à l'échelle.")
        
    # Appliquer les transformations individuelles
    if not scale_all_data:
        df_scaled = transform_data(df_scaled, list_boxcox=list_boxcox, list_yeo=list_yeo, list_log=list_log, list_sqrt=list_sqrt)
    
    # Application de l'ACP en fonction du choix de l'utilisateur
    if use_pca:
        pca = PCA()
        if not use_target:
            df_explicatives = df_scaled.drop(columns=[target])
            df_target = df_scaled[target]
        else:
            df_explicatives = df_scaled.copy()
            df_target = None
        
        # Si l'utilisateur choisit le nombre de composantes
        if pca_option == "Nombre de composantes":
            # Ajuster n_components en fonction du nombre de features disponibles
            n_components = min(n_components, df_explicatives.shape[1])
            pca = PCA(n_components=n_components)
            df_pca = pca.fit_transform(df_explicatives)
        
        # Si l'utilisateur choisit la variance expliquée
        elif pca_option == "Variance expliquée":
            if explained_variance == 100:
                # Utilisation de la méthode PCA avec "None" pour récupérer toutes les composantes qui expliquent 100% de la variance
                pca = PCA(n_components=None)  
            else:
                pca = PCA(n_components=explained_variance / 100)  # Conversion du % en proportion
            df_pca = pca.fit_transform(df_explicatives)
        
        df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])], index=df_explicatives.index)
        if df_target is not None:
            df_scaled = mpd.concat([df_pca, df_target], axis=1)
        else:
            df_scaled = df_pca.copy()
            
    if use_pca:   
        pca_inertias = calculate_inertia(df_explicatives)
        pca_cumulative_inertias = [sum(pca_inertias[:i+1]) for i in range(len(pca_inertias))]
        pca_infos=pd.DataFrame({'Variance expliquée': pca_inertias, 'Variance expliquée cumulée': pca_cumulative_inertias}).round(2)
        pca_infos=pca_infos.reset_index().rename(columns={'index':'Nombre de composantes'})
        pca_infos['Nombre de composantes'] += 1
        
        # Visualisation avec Seaborn
        fig = px.line(pca_infos, x='Nombre de composantes', y=['Variance expliquée', 'Variance expliquée cumulée'], 
                markers=True, title="Evolution de la variance expliquée par les composantes principales",
                labels={'value': 'Variance (%)', 'variable': 'Type de variance'},
                color_discrete_map={'Variance expliquée': 'red', 'Variance expliquée cumulée': 'blue'})
        fig.update_layout(
            xaxis_title='Nombre de composantes principales',
            yaxis_title='Variance (%)',
            legend_title='Type de variance',
            width=900, height=600
        )
    
    # Finir le traitement
    wrang_finished = True
    # Afficher le descriptif de la base de données
    st.write("### Descriptif de la base de données :")
    st.write("**Nombre d'observations :**", df_scaled.shape[0])
    st.write("**Nombre de variables :**", df_scaled.shape[1])
    if df is not None:
        description = []
        for col in df_scaled.columns:
            if pd.api.types.is_numeric_dtype(df_scaled[col]):
                var_type = 'Numérique'
                n_modalites = np.nan
            else:
                var_type = 'Catégorielle'
                n_modalites = df_scaled[col].nunique()
            
            description.append({
                'Variable': col,
                'Type': var_type,
                'Nb modalités': n_modalites
            })
        st.dataframe(pd.DataFrame(description), use_container_width=True, hide_index=True)
    
        if use_pca:
            st.plotly_chart(fig)
    
    # Téléchargement du fichier encodé
    if df is not None and wrang_finished and not pb:
        df = df_scaled.copy()
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Afficher l'aperçu des données traitées
        st.write("### Aperçu des données traitées :")
        st.dataframe(df_scaled)

        # Afficher le bouton pour télécharger le fichier
        st.download_button(
            label="📥 Télécharger les données traitées",
            data=csv_data,
            file_name="data.csv",
            mime="text/csv"
        )
        
if valid_mod:
    # Effectuer la modélisation

    # 1. Division des données
    X = df.drop(columns=target)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    if use_loocv:
        cv = X_train.shape[0]
    # 2. Choisir le meilleur modèle
    results = []
    for model in models:  
        # Déterminer chaque modèle à optimiser
        best_model, best_params, best_value = optimize_model(model_choosen=model, task=task,
                                                            X_train=X_train, y_train=y_train,
                                                            cv=cv, scoring=scoring_comp,
                                                            multi_class=multi_class,
                                                            n_trials=100, n_jobs=-1)
        
        # Ajouter les résultats à la liste
        results.append({
            'Model': model,
            'Best Model': best_model,
            'Best Params': best_params})

    # Créer un DataFrame à partir des résultats
    df_train = pd.DataFrame(results)

    df_train2=df_train.copy()        
    df_train2.set_index('Model', inplace=True)
    df_train2["Best Model"] = df_train2["Best Model"].astype(str)
      
    st.dataframe(df_train2.drop(columns='Best Params'), use_container_width=True)
    
    # 7. Evaluer les meilleurs modèles
    list_models = df_train['Best Model'].tolist()

    list_score = []
    for model in list_models:  # Utilise les vrais objets modèles
        scores = cross_validate(model, X_test, y_test, cv=cv, scoring=scoring_eval, n_jobs=-1)
        mean_scores = {metric: scores[f'test_{metric}'].mean() for metric in scoring_eval}
        std_scores = {metric: scores[f'test_{metric}'].std().round(5) for metric in scoring_eval}

        list_score.append({
            'Best Model': str(model),  # Affichage du nom seulement
            'Mean Scores': {metric: (val * 100).round(2) if task == "Classification" else -val.round(3) for metric, val in mean_scores.items()},
            'Std Scores': std_scores
        })

    df_score = pd.DataFrame(list_score)  

    # Dictionnaires des métriques
    metrics_regression = {
        "R² Score": "r2",
        "Mean Squared Error": "neg_mean_squared_error",
        "Root Mean Squared Error": "neg_root_mean_squared_error",
        "Mean Absolute Error": "neg_mean_absolute_error",
        "Mean Absolute Percentage Error": "neg_mean_absolute_percentage_error"
    }

    metrics_classification = {
        "Accuracy": "accuracy",
        "F1 Score (Weighted)": "f1_weighted",
        "Precision (Weighted)": "precision_weighted",
        "Recall (Weighted)": "recall_weighted"
    }

    # Inverser les dictionnaires des métriques
    inv_metrics_regression = {v: k for k, v in metrics_regression.items()}
    inv_metrics_classification = {v: k for k, v in metrics_classification.items()}
    inv_metrics = inv_metrics_classification if task == "Classification" else inv_metrics_regression

    for metric in scoring_eval:
        df_score[f'Mean {metric}'] = df_score['Mean Scores'].apply(lambda x: x[metric])
        df_score[f'Std {metric}'] = df_score['Std Scores'].apply(lambda x: x[metric])
        
    # Renommage des colonnes pour des noms plus lisibles
    for metric in scoring_eval:
        clean_metric = inv_metrics.get(metric, metric)  # Fallback si absent
        df_score.rename(columns={
            f"Mean {metric}": f"Mean - {clean_metric}",
            f"Std {metric}": f"Std - {clean_metric}"
        }, inplace=True)

    # Derniers traitement
    df_score = df_score.drop(columns=['Mean Scores', 'Std Scores'])
    df_score.index = df_train2.index
    df_score2 = df_score.drop(columns='Best Model')
    st.subheader("Validation des modèles")
    st.dataframe(df_score2, use_container_width=True)
    
    # 8. Appliquer le modèle : calcul-biais-variance et matrice de confusion    
    bias_variance_results = []
    for idx, best_model in df_score['Best Model'].items():
        model = instance_model(idx, df_train2, task)
        expected_loss, bias, var, bias_relative, var_relative = bias_variance_decomp(
            model, task=task,
            X=X_train.values, y=y_train.values,
            cv=cv)

        # Création d'un dictionnaire pour stocker les résultats
        result = {
            "Model Index": idx,  # L'index du modèle
            "Bias": round(bias, 3),  # Biais moyen, arrondi à 3 décimales
            "Variance": round(var, 3),  # Variance moyenne, arrondi à 3 décimales
            "Bias Relative": round(bias_relative, 3),  # Biais relatif, arrondi à 3 décimales
            "Variance Relative": round(var_relative, 3),  # Variance relative, arrondi à 3 décimales
        }
        
        # Logique de conclusion en fonction du biais relatif et de la variance relative
        if abs(bias_relative) > 0.2 and var_relative > 0.2:
            result["Conclusion"] = "Problème majeur : le modèle est vraisembbalement pas adapté"
        elif abs(bias_relative) > 0.2:
            result["Conclusion"] = "Biais élevé : suspition de sous-ajustement"
        elif var_relative > 0.2:
            result["Conclusion"] = "Variance élevée : suspition de sur-ajustement"
        else:
            result["Conclusion"] = "Bien équilibré"
        
        # Ajout du dictionnaire à la liste des résultats
        bias_variance_results.append(result)            
        
    # Création du DataFrame
    df_bias_variance = pd.DataFrame(bias_variance_results)
    df_bias_variance.index = df_train2.index

    # Affichage dans Streamlit
    st.subheader("Etude Bias-Variance")
    st.dataframe(df_bias_variance)

    # Matrices de confusion
    if task == 'Classification':
        st.subheader(f"Bilan des Erreurs de Classification")
        for index, model in df_score['Best Model'].items():
            model = instance_model(idx, df_train2, task)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculer la matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            
            fig, ax = plt.subplots(figsize=(3, 2))  # Taille du graphique
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                        xticklabels=[f"Classe {i}" for i in range(cm.shape[1])] if multi_class else ["Classe 0", "Classe 1"],
                        yticklabels=[f"Classe {i}" for i in range(cm.shape[0])] if multi_class else ["Classe 0", "Classe 1"],
                        annot_kws={"size": 5},
                        cbar=False)  # Désactiver la barre de couleur si tu veux

            # Ajuster la taille des labels
            ax.set_xticklabels(ax.get_xticklabels(), fontsize=5)  # Taille des labels sur l'axe X
            ax.set_yticklabels(ax.get_yticklabels(), fontsize=5)  # Taille des labels sur l'axe Y
            
            plt.xlabel("Prédictions", fontsize=5)  # Taille de l'étiquette X
            plt.ylabel("Réalité", fontsize=5)  # Taille de l'étiquette Y
            plt.title(f"Confusion Matrix - {index}", fontsize=7)  # Taille du titre
            
            st.pyplot(fig)
            plt.close(fig)
    
    # Feature importance
    st.subheader(f"Importance des variables")
    for index, mdl in df_score['Best Model'].items():
        model = instance_model(idx, df_train2, task)
        model.fit(X_train, y_train)
        
        # Calculer l'importance des features par permutation
        result = permutation_importance(model, X_test, y_test, n_repeats=repeats, random_state=42)

        # Extraire l'importance moyenne des features
        importances = result.importances_mean
        std_importances = result.importances_std

        # Trier les importances par ordre décroissant
        sorted_idx = np.argsort(importances)[::-1]  # Tri décroissant

        # Trier les valeurs d'importance et les noms des features
        sorted_importances = importances[sorted_idx]
        sorted_std_importances = std_importances[sorted_idx]
        sorted_features = X_train.columns[sorted_idx]

        # Créer le graphique
        plt.figure(figsize=(5, 3))
        plt.barh(range(len(sorted_features)), sorted_importances, xerr=sorted_std_importances, align="center")
        plt.yticks(range(len(sorted_features)), sorted_features, fontsize=6)
        plt.xticks(fontsize=6)
        plt.xlabel("Importance", fontsize=7)
        plt.title(f"Importance des variables par permutation - {index}", fontsize=8)
        plt.gca().invert_yaxis()
        st.pyplot(plt)
        plt.close()
            
    # Courbes d'apprentissage
    st.subheader(f"Courbes d'apprentissage")
    
    for index, mdl in df_score['Best Model'].items(): 
        model = instance_model(idx, df_train2, task)       
        train_sizes, train_scores, test_scores = learning_curve(
            model, X, y, cv=cv, scoring=scoring_eval[0],  # On prend la première métrique comme référence
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(5, 3))
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score entraînement")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score validation")
        plt.title(f"Learning Curve - {index}", fontsize=8)
        plt.xlabel("Taille de l'échantillon d'entraînement", fontsize=7)
        plt.ylabel("Score", fontsize=7)
        plt.legend(loc="best", fontsize=6)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        st.pyplot(plt)
        plt.close()
        
    # Analyse de drift
    st.subheader("Analyse de drift : comparaison des distributions entre apprentissage et validation")
    
    drift_results = []

    for col in X.columns:
        stat, p_value = ks_2samp(X_train[col], X_test[col])
        drift_results.append({
            "Feature": col,
            "KS Statistic": round(stat, 4),
            "p-value": round(p_value, 4),
            "Drift détecté": "Oui" if p_value < 0.05 else "Non"
        })

    df_drift = pd.DataFrame(drift_results).sort_values("p-value", ascending=False)
    df_drift.set_index("Feature", inplace=True)
    df_drift = df_drift[df_drift["Drift détecté"] == "Oui"].drop(columns="Drift détecté")
    
    if not df_drift.empty:
        st.dataframe(df_drift)
    else:
        st.info("Aucun drift détecté entre les distributions de la base d'apprentissage et la base de validation.")
    
    # Vérifier si le chemin existe
    if os.path.exists(base_dir):
        save_dir = os.path.join(base_dir, "saved_models")
        
        # Créer le dossier s'il n'existe pas
        os.makedirs(save_dir, exist_ok=True)
        
        # Sauvegarde des modèles
        for index, model in df_score['Best Model'].items():
            file_path = os.path.join(save_dir, f"{model_name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(best_model, f)

        # Message de succès global
        st.success(f"✅ Tous les modèles ont été enregistrés dans `{save_dir}`.")
    else:
        st.error("❌ Le chemin spécifié n'existe pas ou n'est pas valide.")
 