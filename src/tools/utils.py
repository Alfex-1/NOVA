import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer, SimpleImputer
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.svm import SVC, SVR
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
import optuna
from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import confusion_matrix

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

def objective_linear(trial):
    model_type = trial.suggest_categorical("model", ["linear", "ridge", "lasso", "elasticnet"])

    if model_type == "linear":
        model = LinearRegression()  # Pas d'hyperparamètre à tuner
    else:
        alpha = trial.suggest_float("alpha", 1e-3, 10, log=True)
        
        if model_type == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        elif model_type == "lasso":
            model = Lasso(alpha=alpha, random_state=42)
        else:
            model = Ridge(alpha=alpha, random_state=42)
    
    # Évaluer avec validation croisée
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
    return score

def objective_logistic(trial, multi_class=False):
    penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet", None])
    C = trial.suggest_float("C", 1e-3, 10, log=True)
    
    if penalty == "elasticnet":
        l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
        model = LogisticRegression(penalty=penalty, C=C, solver='saga', l1_ratio=l1_ratio, max_iter=10000, n_jobs=-1, random_state=42)
    elif penalty == "l1":
        solver = "saga" if multi_class else "liblinear"
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10000, n_jobs=-1, random_state=42)
    elif penalty == "l2":
        solver = "saga" if multi_class else "lbfgs"
        model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10000, n_jobs=-1, random_state=42)
    else:
        solver = "saga" if multi_class else "lbfgs"
        model = LogisticRegression(penalty=penalty, solver=solver, max_iter=10000, n_jobs=-1, random_state=42)
    
    # Évaluer avec validation croisée
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
    return score

def objective(trial, task="Classification", model_type="Random Forest"):
    if model_type == "Random Forest":
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
                bootstrap=bootstrap,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=42
            )
    
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
        C = trial.suggest_float("C", 1e-2, 1e2, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid"])
        degree = trial.suggest_int("degree", 2, 5) if kernel == "poly" else 3
        gamma = trial.suggest_categorical("gamma", ["scale", "auto"])
        
        if task == "Classification":
            model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, random_state=42)
        else:
            model = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)
    
    score = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring).mean()
    return score

def optimize_model(model_choosen, task: str, X_train: pd.DataFrame, y_train: pd.Series, cv: int =10, scoring: str="neg_root_mean_quared_error", multi_class: bool = False, n_trials: int =70, n_jobs: int =-1):
    if model_choosen == "Linear Regression":
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        study.optimize(objective_linear, n_trials=n_trials, n_jobs=n_jobs)
        best_params = study.best_params
        best_value = study.best_value
        
        if best_params["model"] == "linear":
            best_model = LinearRegression()
        elif best_params["model"] == "ridge":
            best_model = Ridge(alpha=best_params["alpha"], random_state=42)
        elif best_params["model"] == "lasso":
            best_model = Lasso(alpha=best_params["alpha"], random_state=42)
        elif best_params["model"] == "elasticnet":
            best_model = ElasticNet(alpha=best_params["alpha"], l1_ratio=best_params["l1_ratio"], random_state=42)

    elif model_choosen == "Logistic Regression":
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        study.optimize(lambda trial: objective_logistic(trial, multi_class=multi_class), n_trials=n_trials, n_jobs=n_jobs)
        best_params = study.best_params
        best_value = study.best_value
        
        penalty = best_params["penalty"]
        C = best_params["C"]
        
        if penalty == "elasticnet":
            l1_ratio = best_params["l1_ratio"]
            best_model = LogisticRegression(penalty=penalty, C=C, solver='saga', l1_ratio=l1_ratio, max_iter=10000, n_jobs=-1, random_state=42)
        elif penalty == "l1":
            solver = "saga" if multi_class else "liblinear"
            best_model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10000, n_jobs=-1, random_state=42)
        elif penalty == "l2":
            solver = "saga" if multi_class else "lbfgs"
            best_model = LogisticRegression(penalty=penalty, C=C, solver=solver, max_iter=10000, n_jobs=-1, random_state=42)
        else:
            solver = "saga" if multi_class else "lbfgs"
            best_model = LogisticRegression(penalty=penalty, solver=solver, max_iter=10000, n_jobs=-1, random_state=42)

    elif model_choosen == "Random Forest":
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs)
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
                bootstrap=bootstrap,
                random_state=42
            )
        else:
            best_model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                bootstrap=bootstrap,
                random_state=42
            )
    elif model_choosen == "KNN":
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs)
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
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())
        study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen), n_trials=n_trials, n_jobs=n_jobs)
        best_params = study.best_params
        best_value = study.best_value
        
        # Récupérer les meilleurs paramètres pour SVM
        C = best_params["C"]
        kernel = best_params["kernel"]
        degree = best_params["degree"] if kernel == "poly" else 3
        gamma = best_params["gamma"]
        
        # Créer le modèle selon la tâche
        if task == "Regression":
            best_model = SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)
        else:
            best_model = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, random_state=42)    
        
    return best_model, best_params, best_value

























def ARCH_search(data, p_max, q_max, o=0, lags_max=7, vol='GARCH', mean='Constant', dist='normal', criterion='aic'):
    """Effectue une recherche exhaustive des meilleurs paramètres pour un modèle ARCH/GARCH sur les données.

    Cette fonction crée une grille de recherche pour les paramètres du modèle ARCH/GARCH (ordre p, q, et lags),
    ajuste plusieurs modèles avec différentes combinaisons de ces paramètres, puis sélectionne le modèle avec 
    le meilleur critère d'information (AIC ou BIC).

    Args:
        data (pd.Series): Série temporelle des données à ajuster avec le modèle ARCH/GARCH.
        p_max (int): Valeur maximale de l'ordre p pour le modèle ARCH.
        q_max (int): Valeur maximale de l'ordre q pour le modèle GARCH.
        o (int, optional): Ordre de l'élément GJRGARCH ou autre forme (par défaut, 0). 
        lags_max (int, optional): Valeur maximale pour l'ordre des lags dans le modèle AR ou HAR (par défaut, 7).
        vol (str, optional): Type de modèle de volatilité, 'GARCH' par défaut mais peut aussi être 'ARCH', 'FIGARCH', etc.
        mean (str, optional): Type de moyenne à utiliser dans le modèle. Par défaut, 'Constant' mais peut être 'AR' ou 'HAR'.
        dist (str, optional): Type de distribution des résidus, 'normal' par défaut mais peut être 't' ou d'autres.
        criterion (str, optional): Critère de sélection du modèle. Par défaut, 'aic' mais peut aussi être 'bic'.

    Returns:
        tuple: Tuple contenant les meilleurs paramètres trouvés pour le modèle.
            - p (int): Ordre de l'élément ARCH (p).
            - q (int): Ordre de l'élément GARCH (q).
            - lags (list or None): Liste des lags utilisés dans le modèle AR ou HAR.
    """
    p_range = range(1, p_max + 1) if vol != 'FIGARCH' else [0, 1]
    q_range = range(1, q_max + 1) if vol != 'FIGARCH' else [0, 1]
    lags_range = range(1, lags_max + 1)

    # Définir la grille de paramètres
    param_grid = {'p': p_range, 'q': q_range, 'lags': lags_range}
    grid = ParameterGrid(param_grid)

    # Utilisation de joblib pour paralléliser le calcul
    results = Parallel(n_jobs=-1)(  # -1 utilise tous les cœurs disponibles
        delayed(fit_model)(data, params['p'], params['q'], o, params['lags'] if mean in ['AR', 'HAR'] else None, mean, dist, vol)
        for params in grid
    )

    # Convertir les résultats en DataFrame
    results_df = pd.DataFrame(results, columns=['p', 'q', 'lags', 'AIC', 'BIC'])

    # Trier les résultats par le critère spécifié
    results_df = results_df.sort_values(by=criterion.upper()).reset_index(drop=True)

    # Extraire les meilleurs paramètres
    best_params = results_df.iloc[0]
    p = int(best_params['p'])
    q = int(best_params['q'])
    lags = best_params['lags']

    return p, q, lags

def model_validation(model):
    """
    Valide les hypothèses relatives à un modèle GARCH sur les résidus d'une série temporelle.

    Args:
        model (arch.__future__.arch_model.ARCHModel): Le modèle GARCH ajusté à la série temporelle. 
               Ce modèle doit avoir des résidus et des paramètres accessibles après ajustement.

    Returns:
        pd.DataFrame: Une DataFrame contenant les résultats des tests de validation des hypothèses, y compris les p-values et un indicateur de respect des hypothèses. 
        Les hypothèses testées incluent la normalité des résidus, l'autocorrélation des résidus, l'autocorrélation des résidus au carré, l'effet ARCH et la stationnarité conditionnelle.
    
    Notes:
        - La stationnarité conditionnelle est vérifiée en s'assurant que la somme des coefficients alpha et beta du modèle GARCH est inférieure à 1. Aucune P-Value n'y est donc associée.
    """
    # Création d'un dictionnaire pour stocker les résultats
    results = {
        'Hypothèse': [],
        'Respect': [],
        'P-Value':[]
    }
    
    # Résidus et paramètres
    resid = model.resid
    resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
    params = pd.DataFrame(model.params)
    params = params[params.index.str.contains('alpha|beta')]
    
    # 1. Normalité des résidus (p-value > 0.05)
    _, p_shapiro = shapiro(resid)
    results['Hypothèse'].append('Normalité des résidus')
    results['Respect'].append(1 if p_shapiro >= 0.05 else 0)
    results['P-Value'].append(p_shapiro)
    
    # 2. Autocorrélation des résidus (p-value >= 0.05 pour toutes les lags)
    lb_resid = acorr_ljungbox(resid, lags=[i for i in range(1, 8)], return_df=True)
    autocorr_resid_pvalues = lb_resid['lb_pvalue']
    results['Hypothèse'].append('Autocorrélation des résidus')
    results['Respect'].append(1 if all(p >= 0.05 for p in autocorr_resid_pvalues) else 0)
    results['P-Value'].append(autocorr_resid_pvalues.tolist())
    
    # 3. Autocorrélation des résidus au carré (p-value >= 0.05 pour toutes les lags)
    lb_resid_sq = acorr_ljungbox(resid**2, lags=[i for i in range(1, 8)], return_df=True)
    autocorr_resid_sq_pvalues = lb_resid_sq['lb_pvalue']
    results['Hypothèse'].append('Autocorrélation des résidus au carré')
    results['Respect'].append(1 if all(p >= 0.05 for p in autocorr_resid_sq_pvalues) else 0)
    results['P-Value'].append(autocorr_resid_sq_pvalues.tolist())
    
    # 4. Hétéroscédasticité conditionnelle (effet ARCH), p-value >= 0.05
    lm_test = het_arch(resid)
    results['Hypothèse'].append('Effet ARCH')
    results['Respect'].append(1 if lm_test[1] >= 0.05 else 0)
    results['P-Value'].append(lm_test[1])
    
    # 5. Stationnarité conditionnelle
    results['Hypothèse'].append('Stationnarité conditionnelle')
    results['Respect'].append(1 if float(np.sum(params, axis=0).iloc[0]) < 1 else 0)
    results['P-Value'].append("None")
    
    # Création d'une DataFrame avec les résultats
    df_results = pd.DataFrame(results)
    return df_results

def distribution(resid):
    """
    Calcule la kurtosis et la skewness (asymétrie) d'une série de résidus pour évaluer la forme de la distribution.

    Args:
        resid (pd.Series ou np.ndarray): Série de résidus pour laquelle la kurtosis et la skewness doivent être calculées.

    Returns:
        float: La kurtosis de la série des résidus, indiquant l'aplatissement ou l'acuité de la distribution.
        float: La skewness (asymétrie) de la série des résidus, indiquant si la distribution est asymétrique vers la gauche ou la droite.
    """
    resid = resid.replace([np.inf, -np.inf], np.nan).dropna()
    kurt = resid.kurtosis()
    skewness = resid.skew()
    
    return kurt, skewness