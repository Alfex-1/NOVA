import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from scipy import stats
from scipy.stats.mstats import winsorize
from scipy.stats import ks_2samp
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_validate, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.metrics import confusion_matrix
import io
from io import BytesIO
import os

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

def encode_data(df: pd.DataFrame, list_binary: list[str] = None, list_ordinal: list[str]=None, list_nominal: list[str]=None, ordinal_mapping: dict[str, int]=None):
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
    if list_binary is not None:
        onehot = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), list_binary)], 
                                  remainder='passthrough')
        df = onehot.fit_transform(df)
        df = pd.DataFrame(df, columns=onehot.get_feature_names_out(list_binary))
    
    # Encodage ordinal pour les variables ordinales
    if list_ordinal is not None:
        for col in list_ordinal:
            if ordinal_mapping is not None and col in ordinal_mapping:
                # Appliquer le mapping d'ordinal
                df[col] = df[col].map(ordinal_mapping[col])
            else:
                # Si le mapping n'est pas fourni, utiliser OrdinalEncoder
                encoder = OrdinalEncoder(categories=[list(ordinal_mapping[col].keys())])
                df[col] = encoder.fit_transform(df[[col]])
    
    # Encodage non-ordinal (OneHot) pour les variables nominales
    if list_nominal is not None:
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
    if list_standard is not None:
        scaler = StandardScaler()
        df[list_standard] = scaler.fit_transform(df[list_standard])
    
    # Min-Max Scaling
    if list_minmax is not None:
        scaler = MinMaxScaler()
        df[list_minmax] = scaler.fit_transform(df[list_minmax])
    
    # Robust Scaling
    if list_robust is not None:
        scaler = RobustScaler()
        df[list_robust] = scaler.fit_transform(df[list_robust])
    
    # Quantile Transformation
    if list_quantile is not None:
        scaler = QuantileTransformer(output_distribution='uniform')
        df[list_quantile] = scaler.fit_transform(df[list_quantile])
    
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
        n_estimators = trial.suggest_int("n_estimators", 10, 300, step=20)
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
        C = trial.suggest_float("C", 0.01, 20.01, log=True)
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
            
    elif model_type == "XGBoost":
        # Définition des hyperparamètres pour un XGBoost
        n_estimators = trial.suggest_int("n_estimators", 10, 300, step=20)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0, step=0.1)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.1)
        gamma = trial.suggest_float("gamma", 0.1, 5, log=True)
        reg_alpha = trial.suggest_float("reg_alpha", 0.0001, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 0.0001, 10.0, log=True)
        
        # Arrondir les float
        learning_rate = round(learning_rate, 8)
        gamma = round(gamma, 5)
        reg_alpha = round(reg_alpha, 5)
        reg_lambda = round(reg_lambda, 5)
        
        if task == "Regression":
            model = XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective="reg:squarederror",
                n_jobs=-1, verbosity=0, random_state=42
            )
        else:
            if multi_class:
                num_class = len(np.unique(y_train))
                model = XGBClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                    subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                    reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective="multi:softprob",
                    num_class=num_class, use_label_encoder=False, eval_metric="mlogloss",
                    n_jobs=-1, verbosity=0)
            else:
                model = XGBClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                    subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                    reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective="binary:logistic",
                    use_label_encoder=False, eval_metric="logloss", n_jobs=-1,
                    verbosity=0)
    
    cross = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring_comp, n_jobs=1)
    mean_score = cross["test_score"].mean()
    return mean_score

def optimize_model(model_choosen, task: str, X_train: pd.DataFrame, y_train: pd.Series, cv: int =10, scoring: str="neg_root_mean_quared_error", multi_class: bool = False, n_trials: int =70, n_jobs: int =-1):
    study = optuna.create_study(direction="maximize", sampler=TPESampler(n_startup_trials=15), pruner=HyperbandPruner())
    
    if model_choosen == "Linear Regression":
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs, timeout=80)
        best_params = study.best_params
        best_value = study.best_value
        
        # Arrondir les hyperparamètres avant de les utiliser
        if "ridge_alpha" in best_params:
            ridge_alpha = round(best_params["ridge_alpha"], 2)
        if "lasso_alpha" in best_params:
            lasso_alpha = round(best_params["lasso_alpha"], 2)
        if "enet_alpha" in best_params:
            enet_alpha = round(best_params["enet_alpha"], 2)
        if "l1_ratio" in best_params:
            l1_ratio = round(best_params["l1_ratio"], 2)

        # Créer le modèle selon le type de régression
        if best_params["model"] == "linear":
            best_model = LinearRegression()
        elif best_params["model"] == "ridge":
            best_model = Ridge(alpha=ridge_alpha)
        elif best_params["model"] == "lasso":
            best_model = Lasso(alpha=lasso_alpha)
        elif best_params["model"] == "elasticnet":
            best_model = ElasticNet(alpha=enet_alpha, l1_ratio=l1_ratio)

    elif model_choosen == "Logistic Regression":
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs, timeout=80)
        best_params = study.best_params
        best_value = study.best_value
        
        # Arrondir les hyperparamètres avant de les utiliser
        C = round(best_params["C"], 3)
        
        if "l1_ratio" in best_params:
            l1_ratio = round(best_params["l1_ratio"], 2)

        # Création du modèle selon le type de pénalité
        penalty = best_params["penalty"]
        
        if penalty == "elasticnet":
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
        C = round(best_params["C"], 2)
        kernel = best_params["kernel"]
        degree = best_params["degree"] if kernel == "poly" else 3
        gamma = round(best_params["gamma"], 5)
        
        # Créer le modèle selon la tâche
        if task == "Regression":
            epsilon = round(best_params["epsilon"],3)
            best_model = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma, epsilon=epsilon)
        else:
            best_model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
            
    elif model_choosen == "XGBoost":
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs, timeout=100)
        best_params = study.best_params
        best_value = study.best_value

        # Récupérer les meilleurs paramètres pour XGBoost
        n_estimators = best_params["n_estimators"]
        max_depth = best_params["max_depth"]
        learning_rate = round(best_params["learning_rate"], 8)
        subsample = round(best_params["subsample"], 1)
        colsample_bytree = round(best_params["colsample_bytree"], 1)
        gamma = round(best_params["gamma"], 5)
        reg_alpha = round(best_params["reg_alpha"], 5)
        reg_lambda = round(best_params["reg_lambda"], 5)

        # Créer le modèle selon la tâche
        if task == "Regression":
            best_model = XGBRegressor(
                n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective="reg:squarederror",
                n_jobs=-1, verbosity=0)
        else:
            if multi_class:
                num_class = len(np.unique(y_train))
                best_model = XGBClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                    subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                    reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective="multi:softprob",
                    num_class=num_class, use_label_encoder=False, eval_metric="mlogloss",
                    n_jobs=-1, verbosity=0)
            else:
                best_model = XGBClassifier(
                    n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate,
                    subsample=subsample, colsample_bytree=colsample_bytree, gamma=gamma,
                    reg_alpha=reg_alpha, reg_lambda=reg_lambda, objective="binary:logistic",
                    use_label_encoder=False, eval_metric="logloss", n_jobs=-1,
                    verbosity=0)
        
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

def create_xgboost_model(best_params):
    if task == "Regression":
        model = XGBRegressor(
            n_estimators=best_params.get("n_estimators", 100),
            max_depth=best_params.get("max_depth", 6),
            learning_rate=best_params.get("learning_rate", 0.1),
            subsample=best_params.get("subsample", 1),
            colsample_bytree=best_params.get("colsample_bytree", 1),
            gamma=best_params.get("gamma", 0),
            reg_alpha=best_params.get("reg_alpha", 0),
            reg_lambda=best_params.get("reg_lambda", 1)
        )
    else:  # Classification
        model = XGBClassifier(
            n_estimators=best_params.get("n_estimators", 100),
            max_depth=best_params.get("max_depth", 6),
            learning_rate=best_params.get("learning_rate", 0.1),
            subsample=best_params.get("subsample", 1),
            colsample_bytree=best_params.get("colsample_bytree", 1),
            gamma=best_params.get("gamma", 0),
            reg_alpha=best_params.get("reg_alpha", 0),
            reg_lambda=best_params.get("reg_lambda", 1)
        )
    
    return model

def create_model_from_string(model_str, best_params):
    # Si c'est XGBoost, on utilise notre fonction spécifique
    if 'XGB' in model_str:
        return create_xgboost_model(best_params)

    # Sinon, on peut utiliser eval pour les autres modèles
    try:
        # Utilisation de eval() pour instancier d'autres modèles (comme RandomForest)
        model = eval(model_str)
        return model
    except Exception as e:
        print(f"Erreur lors de l'évaluation du modèle {model_str}: {e}")
        return None

# df = pd.read_csv(r"C:\Store\Données\boston - Copie.csv", sep=';')
df = pd.read_csv(r"D:\Compet Kaggle\Predict Podcast Listening Time\train_imputed2 - test.csv", sep=';')
target = "Listening_Time_minutes"
use_target = False

if not use_target:
    df_copy=df.copy().drop(columns=target)

scale_all_data = True
scale_method = "Standard Scaler"

df_cat = df.select_dtypes(exclude=['number'])

use_pca = True
n_components = 15
pca_option = "Nombre de composantes"



cv = 10
scoring_comp = "neg_root_mean_squared_error"
scoring_eval = ['neg_root_mean_squared_error', 'neg_mean_percentage_absolute_error']
models = ["Linear Regression", "Random Forest", "XGBoost"]
task = "Regression" # "Classification", "Regression"
target="Listening_Time_minutes"
if task == "Classification" and len(df[target].unique()) > 2:
    multi_class = True
else:
    multi_class = False


# 1. Analyser la corrélation des valeurs manquantes
corr_mat, prop_nan = correlation_missing_values(df)

# 2. Détecter les outliers
df_outliers = detect_outliers_iforest_lof(df, target)

# 3. Appliquer l'encodage des variables (binaire, ordinal, nominal)
if df_cat.shape[1] > 0:
    df_encoded = encode_data(df_outliers, list_binary=None, list_ordinal=["species"], list_nominal=None, ordinal_mapping={"species": {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}})
else:
    df_encoded = df_outliers.copy()
    
# 4. Imputer les valeurs manquantes
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
    df_scaled = scale_data(df_imputed, list_standard=None, list_minmax=None, list_robust=None, list_quantile=None)


# Application de l'ACP en fonction du choix de l'utilisateur
if use_pca:
    pca = PCA()
    df_explicatives = df_scaled.drop(columns=[target])
    df_target = df_scaled[target]
    
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
    
    df_pca = pd.DataFrame(pca.fit_transform(df_explicatives), columns=[f'PC{i+1}' for i in range(df_pca.shape[1])], index=df_explicatives.index)
    df_scaled2 = pd.concat([df_pca, df_target], axis=1)

if use_pca:   
    # Calcul du critère du coude
    pca_inertias = calculate_inertia(df_explicatives)
    pca_cumulative_inertias = [sum(pca_inertias[:i+1]) for i in range(len(pca_inertias))]
    pca_infos=pd.DataFrame({'Variance expliquée': pca_inertias, 'Variance cumulée': pca_cumulative_inertias})
    pca_infos=pca_infos.reset_index().rename(columns={'index':'Nombre de composantes'})
    pca_infos['Nombre de composantes'] += 1
    
    # Visualisation avec Seaborn
    plt.figure(figsize=(8, 6))
    sns.lineplot(x=pca_infos["Nombre de composantes"], y=pca_infos["Variance expliquée"], marker='o', color='b', label="Variance expliquée")
    sns.lineplot(x=pca_infos["Nombre de composantes"], y=pca_infos["Variance cumulée"], marker='o', color='r', label="Variance cumulée")
    plt.title("Méthode du coude pour l'ACP", fontsize=14)
    plt.xlabel('Nombre de composantes principales', fontsize=12)
    plt.ylabel('Variance (%)', fontsize=12)
    plt.xticks(range(1, len(pca_inertias) + 1))
    plt.legend(loc='best')
    plt.show()







df = df_scaled.dropna(subset=[target])
X = df.drop(columns=target)
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# 6. Choisir le meilleur modèle
results = []
for model in models:  
    # Déterminer chaque modèle à optimiser
    best_model, best_params, best_value = optimize_model(model_choosen=model,
                                                         task=task,
                                                         X_train=X_train,
                                                         y_train=y_train,
                                                         cv=cv,
                                                         scoring=scoring_comp,
                                                         multi_class=multi_class,
                                                         n_trials=40,
                                                         n_jobs=-1)
    
    # Ajouter les résultats à la liste
    results.append({
        'Model': model,
        'Best Model': best_model,
        'Best Params': best_params})

# Créer un DataFrame à partir des résultats
df_train = pd.DataFrame(results)        
df_train.set_index('Model', inplace=True)

df_train2 = df_train.copy()
df_train2["Best Model"] = df_train2["Best Model"].astype(str)
print(df_train2)

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
df_score.index = df_train.index
df_score2 = df_score.drop(columns='Best Model')
print(df_score2)

# Mettre en forme exploitables les modèles
df_score['Best Model'] = df_score['Best Model'].apply(lambda x: create_model_from_string(x, best_params))

# 8. Calculer le biais et la variance
bias_variance_results = []
for model in df_score['Best Model']:
    expected_loss, bias, var = bias_variance_decomp(
        model,
        X_train.values, y_train.values,
        X_test.values, y_test.values,
        loss="mse" if task == 'Regression' else "0-1_loss", num_rounds=10,
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
print(df_bias_variance)

# 9. Afficher la matrice de confusion  
if task=='Classification':
    for index, model in df_score['Best Model'].items():
        # Prédictions pour la matrice de confusion
        y_pred = model.predict(X_test)
        
        # Si multi_classe est True, on génère une matrice de confusion adaptée
        if multi_class:
            cm = confusion_matrix(y_test, y_pred)
            
            # Affichage de la matrice de confusion sous forme de heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=[f"Class {i}" for i in range(cm.shape[1])], 
                        yticklabels=[f"Class {i}" for i in range(cm.shape[0])])
            plt.title(f'Confusion Matrix - {index}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
            
        else:
            # Si c'est un problème binaire
            cm = confusion_matrix(y_test, y_pred)
            
            # Affichage de la matrice de confusion sous forme de heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                        xticklabels=["Class 0", "Class 1"], 
                        yticklabels=["Class 0", "Class 1"])
            plt.title('Confusion Matrix (Binary)')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()