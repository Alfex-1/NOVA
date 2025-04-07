# Imports
# from tools.utils import *
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PowerTransformer
from scipy.stats.mstats import winsorize
from scipy import stats
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.metrics import confusion_matrix
from io import BytesIO
import io
import os
import streamlit as st
from PIL import Image
# from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet


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
    - Binaire : OneHotEncoder avec drop='if_binary' (une seule variable)
    - Ordinale : mapping dict ou OrdinalEncoder
    - Nominale : OneHotEncoder classique

    Returns:
        DataFrame encodé
    """
    df = df.copy()

    # Binaire : OneHotEncoder avec drop='if_binary'
    if list_binary:
        encoder = OneHotEncoder(drop='if_binary', sparse=False)
        encoded = encoder.fit_transform(df[list_binary])
        encoded_cols = encoder.get_feature_names_out(list_binary)
        df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        df = df.drop(columns=list_binary)
        df = pd.concat([df, df_encoded], axis=1)

    # Ordinal
    if list_ordinal:
        for col in list_ordinal:
            if ordinal_mapping and col in ordinal_mapping:
                df[col] = df[col].map(ordinal_mapping[col])
            else:
                encoder = OrdinalEncoder()
                df[col] = encoder.fit_transform(df[[col]])

    # Nominal : OneHotEncoder classique
    if list_nominal:
        encoder = OneHotEncoder(drop=None, sparse=False)
        encoded = encoder.fit_transform(df[list_nominal])
        encoded_cols = encoder.get_feature_names_out(list_nominal)
        df_encoded = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
        df = df.drop(columns=list_nominal)
        df = pd.concat([df, df_encoded], axis=1)

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

def scale_data(df: pd.DataFrame, list_standard: list[str] = None, list_minmax: list[str] = None, list_robust: list[str] = None, list_quantile: list[str] = None):   
    """
    Applique différentes méthodes de mise à l'échelle sur les colonnes spécifiées d'un DataFrame.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données à transformer.
        list_standard (list[str], optional): Liste des colonnes à standardiser (Z-score). 
            La standardisation centre les données en moyenne 0 et écart-type 1. Defaults to None.
        list_minmax (list[str], optional): Liste des colonnes à normaliser avec Min-Max Scaling. 
            Transforme les données pour qu'elles soient dans l'intervalle [0, 1]. Defaults to None.
        list_robust (list[str], optional): Liste des colonnes à transformer avec RobustScaler. 
            Échelle robuste aux outliers basée sur le médian et l'IQR (Interquartile Range). Defaults to None.
        list_quantile (list[str], optional): Liste des colonnes à transformer avec QuantileTransformer. 
            Transforme les données en suivant une distribution uniforme. Defaults to None.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes mises à l'échelle selon les transformations spécifiées.
    """
    # Standardisation (Z-score)
    if list_standard and len(list_standard) > 0:
        scaler = StandardScaler()
        df[list_standard] = scaler.fit_transform(df[list_standard])
    
    # Min-Max Scaling
    if list_minmax and len(list_minmax) > 0:
        scaler = MinMaxScaler()
        df[list_minmax] = scaler.fit_transform(df[list_minmax])
    
    # Robust Scaling
    if list_robust and len(list_robust) > 0:
        scaler = RobustScaler()
        df[list_robust] = scaler.fit_transform(df[list_robust])
    
    # Quantile Transformation
    if list_quantile and len(list_quantile) > 0:
        scaler = QuantileTransformer(output_distribution='uniform')
        df[list_quantile] = scaler.fit_transform(df[list_quantile])
    
    return df

def transform_data(df: pd.DataFrame, list_boxcox: list[str] = None, list_yeo: list[str] = None, list_log: list[str] = None, list_sqrt: list[str] = None):
    """
    Applique des transformations sur les colonnes spécifiées d'un DataFrame. 
    Les transformations incluent Box-Cox, Yeo-Johnson, logarithmique et racine carrée.

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données à transformer.
        list_boxcox (list[str], optional): Liste des colonnes à transformer avec Box-Cox. 
            La transformation Box-Cox nécessite des données strictement positives. Defaults to None.
        list_yeo (list[str], optional): Liste des colonnes à transformer avec Yeo-Johnson. 
            La transformation Yeo-Johnson permet de traiter aussi bien les données positives que négatives. Defaults to None.
        list_log (list[str], optional): Liste des colonnes à transformer avec la transformation logarithmique. 
            Nécessite que les données soient strictement positives. Defaults to None.
        list_sqrt (list[str], optional): Liste des colonnes à transformer avec la racine carrée. 
            Nécessite que les données soient positives ou nulles. Defaults to None.

    Returns:
        pd.DataFrame: Le DataFrame avec les colonnes transformées selon les transformations spécifiées.
    """
    
    # Box-Cox Transformation
    if list_boxcox and len(list_boxcox) > 0:
        for col in list_boxcox:
            df[col], _ = stats.boxcox(df[col])

    # Yeo-Johnson Transformation
    if list_yeo and len(list_yeo) > 0:
        transformer = PowerTransformer(method='yeo-johnson')
        df[list_yeo] = transformer.fit_transform(df[list_yeo])
    
    # Logarithmic Transformation
    if list_log and len(list_log) > 0:
        for col in list_log:
            # Logarithme nécessite des données strictement positives
            if (df[col] <= 0).any():
                raise ValueError(f"Les données de la colonne '{col}' doivent être strictement positives pour appliquer le logarithme.")
            df[col] = np.log(df[col])

    # Square Root Transformation
    if list_sqrt and len(list_sqrt) > 0:
        for col in list_sqrt:
            # Racine carrée nécessite des données positives ou nulles
            if (df[col] < 0).any():
                raise ValueError(f"Les données de la colonne '{col}' ne peuvent pas contenir de valeurs négatives pour appliquer la racine carrée.")
            df[col] = np.sqrt(df[col])
    
    return df

def calculate_inertia(X):
    inertias = []
    for i in range(1, X.shape[1] + 1):
        pca = PCA(n_components=i)
        pca.fit(X)
        inertias.append(pca.explained_variance_ratio_[-1]*100)  # La dernière composante expliquée à chaque étape
    return inertias

def objective(trial, task="Classification", model_type="Random Forest", multi_class=False):
    if model_type == "Linear Regression":
        # Définition des hyperparamètres pour Linear Regressio,
        model_linreg = trial.suggest_categorical("model", ["linear", "ridge", "lasso", "elasticnet"])
    
        if model_linreg == "linear":
            model = LinearRegression()
        
        elif model_linreg == "ridge":
            ridge_alpha = trial.suggest_float("ridge_alpha", 1e-3, 10.0, step=0.01)
            ridge_alpha = round(ridge_alpha, 2)
            model = Ridge(alpha=ridge_alpha)

        elif model_linreg == "lasso":
            lasso_alpha = trial.suggest_float("lasso_alpha", 1e-3, 10.0, step=0.01)
            lasso_alpha = round(lasso_alpha, 2)
            model = Lasso(alpha=lasso_alpha)

        elif model_linreg == "elasticnet":
            enet_alpha = trial.suggest_float("enet_alpha", 1e-3, 10.0, step=0.01)
            l1_ratio = trial.suggest_float("l1_ratio", 0, 1.0, step=0.01)
            enet_alpha = round(enet_alpha, 2)
            l1_ratio = round(l1_ratio, 2)
            model = ElasticNet(alpha=enet_alpha, l1_ratio=l1_ratio)
    
    if model_type == "Logistic Regression":
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
    
    elif model_type == "Random Forest":
        # Définition des hyperparamètres pour Random Forest
        n_estimators = trial.suggest_int("n_estimators", 10, 500)
        max_depth = trial.suggest_int("max_depth", 2, 15)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 20)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        
        if task == "Classification":
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap)
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap)
    
    elif model_type == "KNN":
        # Définition des hyperparamètres pour KNN
        n_neighbors = trial.suggest_int("n_neighbors", 1, 50)
        weights = trial.suggest_categorical("weights", ["uniform", "distance"])
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
        
        if task == "Classification":
            model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric
            )
        else:
            model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric
            )
    
    elif model_type == "SVM":
        # Définition des hyperparamètres pour SVM
        C = trial.suggest_float("C", 0.01, 20.01, step=0.01)
        C = round(C, 2)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        
        if task == "Classification":
            model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
        else:
            epsilon = trial.suggest_float("epsilon", 0.01, 1.0, step=0.01)
            epsilon = round(epsilon,2)
            model = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma, epsilon=epsilon)
    
    cross = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring_comp, n_jobs=1)
    mean_score = cross["test_score"].mean()
    return mean_score

def optimize_model(model_choosen, task: str, X_train: pd.DataFrame, y_train: pd.Series, cv: int =10, scoring: str="neg_root_mean_quared_error", multi_class: bool = False, n_trials: int =70, n_jobs: int =-1):
    study = optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=15), pruner=HyperbandPruner())
    
    if model_choosen == "Linear Regression":
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs, timeout=80)
        best_params = study.best_params
        best_value = study.best_value
        
        if best_params["model"] == "linear":
            best_model = LinearRegression()
        elif best_params["model"] == "ridge":
            best_model = Ridge(alpha=best_params["ridge_alpha"])
        elif best_params["model"] == "lasso":
            best_model = Lasso(alpha=best_params["lasso_alpha"])
        elif best_params["model"] == "elasticnet":
            best_model = ElasticNet(alpha=best_params["enet_alpha"], l1_ratio=best_params["l1_ratio"])

    elif model_choosen == "Logistic Regression":
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs, timeout=80)
        best_params = study.best_params
        best_value = study.best_value
        
        penalty = best_params["penalty"]
        C = best_params["C"]
        
        if penalty == "elasticnet":
            l1_ratio = best_params["l1_ratio"]
            best_model = LogisticRegression(penalty=penalty, C=C, solver='saga', l1_ratio=l1_ratio, max_iter=10000, n_jobs=-1)
        elif penalty == "l1":
            solver = "saga" if multi_class else "liblinear"
            best_model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10000, n_jobs=-1)
        elif penalty == "l2":
            solver = "saga" if multi_class else "lbfgs"
            best_model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10000, n_jobs=-1)
        else:
            solver = "saga" if multi_class else "lbfgs"
            best_model = LogisticRegression(penalty=penalty, solver=solver, max_iter=10000, n_jobs=-1)

    elif model_choosen == "Random Forest":
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs, timeout=150)
        best_params = study.best_params
        best_value = study.best_value
        
        n_estimators = best_params["n_estimators"]
        max_depth = best_params["max_depth"]
        min_samples_split = best_params["min_samples_split"]
        min_samples_leaf = best_params["min_samples_leaf"]
        max_features = best_params["max_features"]
        bootstrap = best_params["bootstrap"]
        
        if task == "Regression":
            best_model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap)
        else:
            best_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap)
    elif model_choosen == "KNN":
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs, timeout=80)
        best_params = study.best_params
        best_value = study.best_value
        
        # Récupérer les meilleurs paramètres pour KNN
        n_neighbors = best_params["n_neighbors"]
        weights = best_params["weights"]
        metric = best_params["metric"]
        
        # Créer le modèle selon la tâche
        if task == "Regression":
            best_model = KNeighborsRegressor(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric
            )
        else:
            best_model = KNeighborsClassifier(
                n_neighbors=n_neighbors,
                weights=weights,
                metric=metric
            )

    elif model_choosen == "SVM":
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs, timeout=100)
        best_params = study.best_params
        best_value = study.best_value
        
        # Récupérer les meilleurs paramètres pour SVM
        C = best_params["C"]
        kernel = best_params["kernel"]
        degree = best_params["degree"] if kernel == "poly" else 3
        gamma = best_params["gamma"]
        
        # Créer le modèle selon la tâche
        if task == "Regression":
            epsilon = best_params["epsilon"]
            best_model = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma, epsilon=epsilon)
        else:
            best_model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)    
        
    return best_model, best_params, best_value

def _draw_bootstrap_sample(rng, X, y):
    sample_indices = np.arange(X.shape[0])
    bootstrap_indices = rng.choice(sample_indices, size=sample_indices.shape[0], replace=True)
    return X[bootstrap_indices], y[bootstrap_indices]

def bias_variance_decomp(estimator, X_train, y_train, X_test, y_test, loss="0-1_loss", num_rounds=200, random_seed=None, **fit_params):
    if loss not in ["0-1_loss", "mse"]:
        raise NotImplementedError("Loss must be '0-1_loss' or 'mse'")

    rng = np.random.RandomState(random_seed)
    all_pred = np.zeros((num_rounds, y_test.shape[0]), dtype=np.float64 if loss == "mse" else np.int64)

    for i in range(num_rounds):
        X_boot, y_boot = _draw_bootstrap_sample(rng, X_train, y_train)
        pred = estimator.fit(X_boot, y_boot, **fit_params).predict(X_test)
        all_pred[i] = pred

    main_predictions = np.apply_along_axis(np.mean if loss == "mse" else lambda x: np.argmax(np.bincount(x)), axis=0, arr=all_pred)
    
    if loss == "0-1_loss":
        main_predictions = np.apply_along_axis(lambda x: np.argmax(np.bincount(x)), axis=0, arr=all_pred)
        avg_expected_loss = np.apply_along_axis(lambda x: (x != y_test).mean(), axis=1, arr=all_pred).mean()
        
        avg_bias = np.sum(main_predictions != y_test) / y_test.size
        var = np.zeros(pred.shape)
        for pred in all_pred:
            var += (pred != main_predictions).astype(np.int_)
        var /= num_rounds

        avg_var = var.sum() / y_test.shape[0]
    else:
        avg_expected_loss = np.apply_along_axis(lambda x: ((x - y_test) ** 2).mean(), axis=1, arr=all_pred).mean()
        main_predictions = np.mean(all_pred, axis=0)

        avg_bias = np.sum((main_predictions - y_test)) / y_test.size
        avg_var = np.sum((main_predictions - all_pred) ** 2) / all_pred.size
    
    return avg_expected_loss, avg_bias, avg_var

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
    - 🔍 **Optimisation des hyperparamètres** : recherche des meilleurs réglages pour 4 modèles populaires (régression linéaire/logistique, KNN, SVM, Random Forest).
    - 🏆 **Évaluation des modèles** : validation croisée, analyse biais-variance, et matrice de confusion pour les tâches de classification.
    
    **NOVA** permet à chaque utilisateur de bénéficier d’une infrastructure robuste, tout en maintenant une flexibilité totale sur le traitement fondamental des données.
    Vous contrôlez les choix, nous optimisons les outils.
    """
)

# Chargement des données
df = None
uploaded_file = st.file_uploader("Choississez un fichier (csv, xlsx et  txt acceptés seulement)", type=["csv", "xlsx", "txt"])
wrang = st.checkbox("La base de données nécessite un traitement")
valid_mod=False

if uploaded_file is not None:
    byte_data = uploaded_file.read()
    separators = [";", ",", "\t"]

    # Lecture optimisée avec sélection du premier séparateur valide
    for sep in separators:
        try:
            df = pd.read_csv(BytesIO(byte_data), sep=sep, engine="python", nrows=20)  # Charge un échantillon
            if df.shape[1] > 1:
                df = pd.read_csv(BytesIO(byte_data), sep=sep)  # Recharge tout avec le bon séparateur
                break
        except Exception:
            df, sep = None, None  # Réinitialisation en cas d'échec

    if sep is None:
        st.warning("Échec de la détection du séparateur. Vérifiez le format du fichier.")

# Sidebar pour la configuration de l'utilisateur    
if df is not None:
    st.sidebar.image(Image.open("logo_nova.png"), width=200)
    
    if wrang is True:            
        st.sidebar.title("Paramètres du traitement des données")
        
        target = st.sidebar.selectbox("Choisissez la variable cible de votre modélisation", df.columns.to_list(), help="Si vous n'avaez pas de variable cible, choisissez une variable au harsard.")
        pb = False
        wrang_finished = False
        
        st.sidebar.subheader("Mise à l'échelle des variables numériques")
        
        # Déterminer si la variable cible doit être incluse dans la mise à l'échelle
        use_target = st.sidebar.checkbox("Inclure la variable cible dans la mise à l'échelle", value=False, help="Cochez cette case si vous ne souhaotez pas inclure une variable dans le traitement.")
        drop_dupli = st.sidebar.checkbox("Supprimer toutes les observations dupliquées", value=False)
        
        if drop_dupli:
            df.drop_duplicates(inplace=True)
        
        if not use_target:
            df_copy=df.copy()
            df_copy=df_copy.drop(columns=target)
        
        # Tout mettre à l'échelle directement
        scale_all_data = st.sidebar.checkbox("Mettre à l'échelle toutes les données avec la même méthode", value=False)
        
        if scale_all_data:
            scale_method = st.sidebar.selectbox("Méthode de mise à l'échelle à appliquer",
                                                ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "Quantile Transformer (Uniform)"])
        
        # Obtenir des dataframes distinctes selon les types des données
        if not use_target:
            df_num = df_copy.select_dtypes(include=['number'])
        else:
            df_num = df.select_dtypes(include=['number'])

        df_cat = df.select_dtypes(exclude=['number'])
        
        # Mise à l'échelle (si pas de mise à l'échelle sur toutes les données d'un seul coup)
        if not scale_all_data:
            list_standard = None
            list_minmax = None
            list_robust = None
            list_quantile = None
            list_standard = st.sidebar.multiselect("Colonnes à standardiser (StandardScaler)", df_num.columns.to_list())
            list_minmax = st.sidebar.multiselect("Colonnes à normaliser (MinMaxScaler)", df_num.columns.to_list())
            list_robust = st.sidebar.multiselect("Colonnes à normaliser avec robustesse (RobustScaler)", df_num.columns.to_list())
            list_quantile = st.sidebar.multiselect("Colonnes à transformer en distribution uniforme (QuantileTransformer)", df_num.columns.to_list())
            
            # Vérification dans les listes de mise à l'échelle
            scaling_vars = list_standard + list_minmax + list_robust + list_quantile

            # Vérifier les doublons dans les listes de mise à l'échelle
            duplicates_in_scaling = set([var for var in scaling_vars if scaling_vars.count(var) > 1])
            if duplicates_in_scaling:
                pb = True
                st.sidebar.warning(f"⚠️ Les variables suivantes sont présentes plusieurs fois dans les listes des variables à mettre à l'échelle : {', '.join(duplicates_in_scaling)}") 
        
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
        
        
        # Appliquer les modifications
        
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
            
        else:
            df_scaled = scale_data(df_imputed, list_standard=list_standard, list_minmax=list_minmax, list_robust=list_robust, list_quantile=list_quantile)

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
                df_scaled = pd.concat([df_pca, df_target], axis=1)
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
    
    else:
        # Modélisation
        st.sidebar.title("Paramètres de Modélisation")
        
        # Initialisation du document PDF
        # doc = SimpleDocTemplate("rapport_modelisation.pdf", pagesize=A4)
        # story = []
        # styles = getSampleStyleSheet()
        # titre = Paragraph("Rapport de Modélisation Automatique", styles['Title'])
        # story.append(titre)
        # story.append(Spacer(1, 24)) 

        # Définition de la variable cible
        target = st.sidebar.selectbox("Choisissez la variable cible", df.columns.to_list())
        # story.append(Paragraph("Titre de section", styles['Heading2']))
        # story.append(Paragraph("Voici une description quelconque du modèle.", styles['Normal']))
        # story.append(Spacer(1, 12))
        # story.append(Paragraph(f"Variable cible sélectionnée : <b>{target}</b>", styles['Normal']))
        # story.append(Spacer(1, 12))
        
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
            models = st.sidebar.multiselect("Modèle(s) à tester", ["Random Forest", "KNN", "Linear Regression", "SVM"], default=["Linear Regression"])
        else:
            models = st.sidebar.multiselect("Modèle(s) à tester", ["Random Forest", "KNN", "Logistic Regression", "SVM"], default=["Logistic Regression"])
            
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
        if use_loocv:
            cv = df.shape[0]
        else:
            cv = st.sidebar.number_input(
                "Nombre de folds (CV)",
                min_value=2, max_value=20,
                value=7, step=1,
                disabled=use_loocv)
            
        # Demander le temps disponible selon les choix de l'utilisateur
        time_sup= st.sidebar.checkbox("Voulez vous prendre plus de temps pour améliorer les résultats ?")
        
       # Pondération de complexité selon les modèles
        complexity_weights = {
            "Linear Regression": 2,
            "KNN": 3,
            "Logistic Regression": 3,
            "Random Forest": 6,
            "SVM": 7,
        }
        
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
        
        num_rounds = trial
        repeats = trial
            
        st.sidebar.subheader("Enregistrement des modèles")
        # Demander à l'utilisateur où il souhaite enregistrer les modèles
        base_dir = st.sidebar.text_input("Entrez le chemin du dossier qui contiendra les modèles enregistrés", help="Exemple : C:\\Users\\Documents")
        
        # Valider les choix
        valid_mod = st.sidebar.button("Valider les choix de modélisation")
        
if valid_mod:
    # Effectuer la modélisation

    # 1. Division des données
    X = df.drop(columns=target)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=True)

    # 2. Choisir le meilleur modèle
    results = []
    for model in models:  
        # Déterminer chaque modèle à optimiser
        best_model, best_params, _ = optimize_model(model_choosen=model,
                                                            task=task,
                                                            X_train=X_train,
                                                            y_train=y_train,
                                                            cv=cv,
                                                            scoring=scoring_comp,
                                                            multi_class=multi_class,
                                                            n_trials=trial,
                                                            n_jobs=-1)
        
        # Ajouter les résultats à la liste
        results.append({
            'Model': str(model),
            'Best Model': best_model})

    # Créer un DataFrame à partir des résultats
    df_train = pd.DataFrame(results)        
    df_train = df_train.set_index(df_train['Model']).drop(columns='Model')

    # Afficher les résultats de la comparaison
    st.subheader("Comparaison et optimisation des modèles")
    df_train2 = df_train.copy()
    df_train2["Best Model"] = df_train2["Best Model"].astype(str)   
    st.dataframe(df_train2)
    # st.write(f"Nombre d'essais Optuna: {trial}, Nombre de rounds: {num_rounds}")
    
    # 7. Evaluer les meilleurs modèles
    list_models = df_train['Best Model'].tolist()

    list_score = []
    for model in list_models:  # Utilise les vrais objets modèles
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring_eval, n_jobs=-1)
        mean_scores = {metric: scores[f'test_{metric}'].mean() for metric in scoring_eval}
        std_scores = {metric: scores[f'test_{metric}'].std().round(5) for metric in scoring_eval}

        list_score.append({
            'Best Model': str(model),  # Affichage du nom seulement
            'Mean Scores': {metric: (val * 100).round(2) if task == "Classification" else -val.round(3) for metric, val in mean_scores.items()},
            'Std Scores': std_scores
        })

    df_score = pd.DataFrame(list_score)
    
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
    df_score.index = df_train.index
    df_score2 = df_score.drop(columns='Best Model')
    st.subheader("Validation des modèles")
    st.dataframe(df_score2)
    
    # 8. Appliquer le modèle : calcul-biais-variance et matrice de confusion
    
    # Vérifier si les modèles sont instanciés ou juste des classes    
    bias_variance_results = []
    df_score['Best Model'] = df_score['Best Model'].apply(lambda x: eval(x))
    for model in df_score['Best Model']:
            
        expected_loss, bias, var = bias_variance_decomp(
            model,
            X_train.values, y_train.values,
            X_test.values, y_test.values,
            loss="mse" if task == 'Regression' else "0-1_loss", num_rounds=num_rounds,
            random_seed=123)

        if task == 'Classification':
            bias_variance_results.append({
                # "Average 0-1 Loss": round(expected_loss, 3),
                "Bias": round(bias, 3),
                "Variance": round(var, 3)})
        else:
            bias_variance_results.append({
                # "Average Squared Loss": round(expected_loss, 3),
                "Bias": round(bias, 3),
                "Variance": round(var, 3)})        
        
    # Création du DataFrame
    df_bias_variance = pd.DataFrame(bias_variance_results)
    df_bias_variance.index = df_train.index

    # Affichage dans Streamlit
    st.subheader("Etude Bias-Variance")
    st.dataframe(df_bias_variance)

    # Matrices de confusion
    if task == 'Classification':
        st.subheader(f"Bilan des Erreurs de Classification")
        for index, row in df_score.iterrows():
            model = row['Best Model']
            y_pred = model.predict(X_test)
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
    for index, row in df_score.iterrows():
            model = row['Best Model']
            model.fit(X_train, y_train)
            
            # Calculer l'importance des features par permutation
            result = permutation_importance(model, X_test, y_test, n_repeats=int(trial*1.5), random_state=42)

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
            
    # Courbes d'apprentissage
    st.subheader(f"Courbes d'apprentissage")
    
    for index, row in df_score.iterrows():
        model = row['Best Model']
        
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
    df_drift=df_drift.set_index("Feature", inplace=True)
    df_drift = df_drift[df_drift["Drift détecté"] == True].drop(columns="Drift détecté")
    
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
        for index, row in df_score.iterrows():
            model_name = index
            best_model = row['Best Model']
            file_path = os.path.join(save_dir, f"{model_name}.pkl")
            with open(file_path, "wb") as f:
                pickle.dump(best_model, f)
        
        # # Génération du rapport PDF
        # pdf_path = os.path.join(save_dir, "rapport_modelisation.pdf")
        # doc = SimpleDocTemplate(pdf_path, pagesize=A4)
        # doc.build(story)  # story doit avoir été remplie au préalable

        # Message de succès global
        st.success(f"✅ Tous les modèles et le rapport PDF ont été enregistrés dans `{save_dir}`.")
    else:
        st.error("❌ Le chemin spécifié n'existe pas ou n'est pas valide.")
 