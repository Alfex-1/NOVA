import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
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
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
from sklearn.metrics import confusion_matrix
import lightgbm as lgb

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
        model = lgb.LGBMRegressor(**param, verbose=-1) if task == 'Regression' else lgb.LGBMClassifier(**param, verbose=-1)

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
            model = xgb.XGBRegressor(**param)
        else:
            model = xgb.XGBClassifier(**param)

    elif model_type == "Random Forest":
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
        }
        model = RandomForestRegressor(**param) if task == 'Regression' else RandomForestClassifier(**param)

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
        model = KNeighborsRegressor(**param) if task == 'Regression' else KNeighborsClassifier(**param)

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
            alpha = best_params.get('elasticnet_alpha')
            l1_ratio = best_params.get('l1_ratio')
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        
    elif model_name == 'Logistic Regression':
        model = LogisticRegression(*best_params)
           
    elif model_name == 'SVM':
        if task == 'Classification':
            model = SVC(**best_params)  # SVC pour classification
        else:
            model = SVR(**best_params)  # SVR pour régression
    
    return model

df = pd.read_csv(r"C:\Store\Données\boston.csv", sep=';')
target = "MEDV"
use_target = False

if not use_target:
    df_copy=df.copy().drop(columns=target)

scale_all_data = True
scale_method = "Robust Scaler"

df_cat = df.select_dtypes(exclude=['number'])

use_pca = False
n_components = 15
pca_option = "Nombre de composantes"



cv = 10
scoring_comp = "neg_root_mean_squared_error"
scoring_eval = ['neg_root_mean_squared_error']
models = ["Linear Regression"]
task = "Regression" # "Classification", "Regression"
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
print(df_train2)

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
print(df_score2)

# 8. Calculer le biais et la variance
bias_variance_results = []
for idx, best_model in df_score['Best Model'].items():
    model = instance_model(idx, df_train2, task)
    expected_loss, bias, var = bias_variance_decomp(
        model,
        X_train.values, y_train.values,
        X_test.values, y_test.values,
        loss="mse" if task == 'Regression' else "0-1_loss", num_rounds=20,
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
df_bias_variance.index = df_train2.index
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
            
# Feature importance
for index, mdl in df_score['Best Model'].items():
    model = instance_model(idx, df_train2, task)
    
    # Calculer l'importance des features par permutation
    result = permutation_importance(model, X_test, y_test, n_repeats=20, random_state=42)

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
    plt.show()
        
# Courbes d'apprentissage
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
    plt.show()
    
# Analyse de drift
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
    print(df_drift)
else:
    print("Aucun drift détecté entre les distributions de la base d'apprentissage et la base de validation.")
