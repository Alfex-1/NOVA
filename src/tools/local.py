import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
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
from sklearn.model_selection import train_test_split, cross_validate, learning_curve, KFold
from sklearn.inspection import permutation_importance
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.metrics import confusion_matrix, get_scorer
import lightgbm as lgb
from joblib import Parallel, delayed

def correlation_missing_values(df_train: pd.DataFrame, df_test: pd.DataFrame = None):
    """
    Analyse la corrélation entre les valeurs manquantes dans deux DataFrames (train et test).

    Cette fonction identifie les colonnes contenant des valeurs manquantes, 
    calcule la proportion de NaN par colonne et retourne les matrices de corrélation 
    des valeurs manquantes pour les bases d'entraînement, de test et la base combinée.

    Args:
        df_train (pd.DataFrame): Le DataFrame d'entraînement
        df_test (pd.DataFrame, optional): Le DataFrame de test (par défaut None)

    Returns:
        tuple: 
            - cor_mat_train : Matrice de corrélation des valeurs manquantes pour df_train
            - cor_mat_test : Matrice de corrélation des valeurs manquantes pour df_test
            - cor_mat_combined : Matrice de corrélation des valeurs manquantes pour df_combined (train + test)
            - prop_nan_train : Proportion des valeurs manquantes pour df_train
            - prop_nan_test : Proportion des valeurs manquantes pour df_test
            - prop_nan_combined : Proportion des valeurs manquantes pour df_combined (train + test)
    """
    
    def compute_missing_info(df):
        # Filtrer les colonnes avec des valeurs manquantes
        df_missing = df.iloc[:, [i for i, n in enumerate(np.var(df.isnull(), axis=0)) if n > 0]]
        
        # Calculer la proportion de valeurs manquantes et ajouter le type de variable
        prop_nan = pd.DataFrame({
            "NaN proportion": round(df.isnull().sum() / len(df) * 100, 2),
            "Type": df.dtypes.astype(str)
        })

        # Calculer la matrice de corrélation des valeurs manquantes
        corr_mat = round(df_missing.isnull().corr() * 100, 2)

        return corr_mat, prop_nan

    # Calculs pour df_train
    cor_mat_train, prop_nan_train = compute_missing_info(df_train)

    # Si df_test existe, calculer aussi pour df_test
    if df_test is not None:
        cor_mat_test, prop_nan_test = compute_missing_info(df_test)
        
        # Calcul pour la base combinée (train + test)
        df_combined = pd.concat([df_train, df_test], axis=0)
        cor_mat_combined, prop_nan_combined = compute_missing_info(df_combined)
    else:
        cor_mat_test, prop_nan_test, cor_mat_combined, prop_nan_combined = None, None, cor_mat_train, prop_nan_train

    # Retourner les résultats sous forme de variables séparées
    return cor_mat_train, cor_mat_test, cor_mat_combined, prop_nan_train, prop_nan_test, prop_nan_combined

def encode_data(df_train: pd.DataFrame, df_test: pd.DataFrame = None, list_binary: list[str] = None, list_ordinal: list[str] = None, list_nominal: list[str] = None, ordinal_mapping: dict[str, dict[str, int]] = None) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Encode les variables catégorielles du DataFrame selon leur nature.
    - Binaire : OneHotEncoder avec drop='if_binary' (garde une seule colonne si possible)
    - Ordinale : Mapping manuel ou OrdinalEncoder
    - Nominale : OneHotEncoder classique (toutes les modalités)

    Args:
        df_train: DataFrame d'entraînement
        df_test: DataFrame de test (optionnel)
        list_binary: liste des variables binaires
        list_ordinal: liste des variables ordinales
        list_nominal: liste des variables nominales
        ordinal_mapping: dictionnaire pour mapping ordinal (optionnel)

    Returns:
        df_train_encoded, df_test_encoded (ou None si df_test n'est pas fourni)
    """
    df_train = df_train.copy()
    df_test = df_test.copy() if df_test is not None else None

    # --- Binaire
    if list_binary:
        encoder = OneHotEncoder(drop='if_binary', sparse_output=False)
        encoder.fit(df_train[list_binary])

        encoded_train = pd.DataFrame(
            encoder.transform(df_train[list_binary]),
            columns=encoder.get_feature_names_out(list_binary),
            index=df_train.index
        )

        df_train.drop(columns=list_binary, inplace=True)
        df_train = pd.concat([df_train, encoded_train], axis=1)

        if df_test is not None:
            encoded_test = pd.DataFrame(
                encoder.transform(df_test[list_binary]),
                columns=encoder.get_feature_names_out(list_binary),
                index=df_test.index
            )
            df_test.drop(columns=list_binary, inplace=True)
            df_test = pd.concat([df_test, encoded_test], axis=1)

    # --- Ordinal
    if list_ordinal:
        for col in list_ordinal:
            if ordinal_mapping and col in ordinal_mapping:
                df_train[col] = df_train[col].map(ordinal_mapping[col])
                if df_test is not None:
                    df_test[col] = df_test[col].map(ordinal_mapping[col])
            else:
                encoder = OrdinalEncoder()
                encoder.fit(df_train[[col]])
                df_train[col] = encoder.transform(df_train[[col]])
                if df_test is not None:
                    df_test[col] = encoder.transform(df_test[[col]])

    # --- Nominal
    if list_nominal:
        encoder = OneHotEncoder(drop=None, sparse_output=False)
        encoder.fit(df_train[list_nominal])

        encoded_train = pd.DataFrame(
            encoder.transform(df_train[list_nominal]),
            columns=encoder.get_feature_names_out(list_nominal),
            index=df_train.index
        )

        df_train.drop(columns=list_nominal, inplace=True)
        df_train = pd.concat([df_train, encoded_train], axis=1)

        if df_test is not None:
            encoded_test = pd.DataFrame(
                encoder.transform(df_test[list_nominal]),
                columns=encoder.get_feature_names_out(list_nominal),
                index=df_test.index
            )
            df_test.drop(columns=list_nominal, inplace=True)
            df_test = pd.concat([df_test, encoded_test], axis=1)

    return df_train, df_test if df_test is not None else df_train

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

class ParametricImputer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.fitted = False
        self.params = {}
        self.distribution = None

    def fit(self, X, y=None):
        if not isinstance(X, pd.Series):
            raise ValueError("L'entrée doit être une série pandas.")
        data = X.dropna()

        # Fit normal
        mu, sigma = stats.norm.fit(data)
        _, p_norm = stats.kstest(data, 'norm', args=(mu, sigma))

        # Fit uniforme
        a, b = np.min(data), np.max(data)
        _, p_unif = stats.kstest(data, 'uniform', args=(a, b - a))

        # Fit exponentielle
        lambda_hat = 1 / data.mean()  # paramètre de la loi exponentielle
        _, p_exp = stats.kstest(data, 'expon', args=(0, lambda_hat))

        # Fit log-normale
        log_data = np.log(data)
        mu_log, sigma_log = stats.norm.fit(log_data)
        _, p_lognorm = stats.kstest(data, 'lognorm', args=(sigma_log, 0, np.exp(mu_log)))

        # Choix : distribution avec le plus grand p-value
        p_values = {
            'norm': p_norm,
            'uniform': p_unif,
            'expon': p_exp,
            'lognorm': p_lognorm
        }

        best_dist = max(p_values, key=p_values.get)  # On choisit la distribution avec la plus grande p-value

        if best_dist == 'norm':
            self.distribution = 'norm'
            self.params = {'mu': mu, 'sigma': sigma}
        elif best_dist == 'uniform':
            self.distribution = 'uniform'
            self.params = {'a': a, 'b': b}
        elif best_dist == 'expon':
            self.distribution = 'expon'
            self.params = {'lambda': lambda_hat}
        else:  # log-normale
            self.distribution = 'lognorm'
            self.params = {'mu': mu_log, 'sigma': sigma_log}

        self.fitted = True
        return self

    def sample(self, size):
        if not self.fitted:
            raise RuntimeError("Le fit doit être exécuté avant le sampling.")
        rng = np.random.default_rng(self.random_state)

        if self.distribution == 'norm':
            return rng.normal(loc=self.params['mu'], scale=self.params['sigma'], size=size)
        elif self.distribution == 'uniform':
            return rng.uniform(low=self.params['a'], high=self.params['b'], size=size)
        else:
            raise RuntimeError("Distribution non reconnue.")

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Impossible de transformer avant le fit.")
        series = X.copy()
        missing = series.isnull()
        n_missing = missing.sum()
        if n_missing > 0:
            sampled_values = self.sample(n_missing)
            series.loc[missing] = sampled_values
        return series

class MultiParametricImputer:
    def __init__(self, distribution='lognorm', random_state=42):
        self.distribution = distribution
        self.random_state = random_state
        self.imputers = {}
        self.fitted = False
        self.imputed_info = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("L'entrée doit être un DataFrame pandas.")
        for col in X.columns:
            imputer = ParametricImputer(distribution=self.distribution, random_state=self.random_state)
            imputer.fit(X[col])
            self.imputers[col] = imputer
            self.imputed_info[col] = {
                'params': imputer.params,
                'distribution': imputer.distribution,
                'n_missing_train': X[col].isna().sum()
            }
        self.fitted = True
        return self

    def transform(self, X):
        if not self.fitted:
            raise RuntimeError("Tu dois fitter avant de transformer.")
        df_copy = X.copy()
        for col, imputer in self.imputers.items():
            if col in df_copy.columns:
                df_copy[col] = imputer.transform(df_copy[col])
        return df_copy
    
def impute_from_supervised(df_train, df_test, cols_to_impute, cv=5):
    """
    Impute les valeurs manquantes des colonnes sélectionnées en utilisant des modèles supervisés (arbres de décision).

    Pour chaque colonne cible, entraîne un arbre de décision (classifieur pour les variables catégorielles, régressseur pour les variables continues)
    sur les données connues, puis impute les valeurs manquantes dans les ensembles d'entraînement et de test.

    Args:
        df_train (pd.DataFrame): Jeu de données d'entraînement contenant des valeurs manquantes.
        df_test (pd.DataFrame ou None): Jeu de données de test contenant des valeurs manquantes. Peut être None si indisponible.
        cols_to_impute (list of str): Liste des noms de colonnes à imputer.
        cv (int, optionnel): Nombre de plis pour la validation croisée utilisée pour évaluer la performance des modèles. Par défaut à 5.

    Returns:
        pd.DataFrame: Jeu de données d'entraînement mis à jour avec les imputations.
        pd.DataFrame ou None: Jeu de données de test mis à jour avec les imputations (ou None si non fourni).
        pd.DataFrame: DataFrame contenant les scores de performance pour chaque colonne imputée.
    """
    df_train = df_train.copy()
    df_test = df_test.copy() if df_test is not None else None
    
    scores = []

    for target_col in cols_to_impute:
        target_is_categorical = df_train[target_col].dtype == 'object' or str(df_train[target_col].dtype) == 'category'

        model = DecisionTreeClassifier(criterion='entropy', class_weight='balanced', ccp_alpha=0.01, random_state=42) if target_is_categorical else DecisionTreeRegressor(criterion='squared_error', ccp_alpha=0.01, random_state=42)

        train_known = df_train[df_train[target_col].notna()].copy()
        train_missing = df_train[df_train[target_col].isna()].copy()
        test_missing = df_test[df_test[target_col].isna()].copy()

        feature_cols = [col for col in cols_to_impute if col != target_col]

        X_fit = train_known[feature_cols].copy()
        y_fit = train_known[target_col].copy()

        for col in X_fit.select_dtypes(include='object').columns:
            le = LabelEncoder()
            X_fit.loc[:, col] = le.fit_transform(X_fit[col].astype(str))

        if not train_missing.empty:
            X_missing_train = train_missing[feature_cols].copy()
            for col in X_missing_train.select_dtypes(include='object').columns:
                le = LabelEncoder()
                X_missing_train.loc[:, col] = le.fit_transform(X_missing_train[col].astype(str))
        else:
            X_missing_train = None

        if not test_missing.empty:
            X_missing_test = test_missing[feature_cols].copy()
            for col in X_missing_test.select_dtypes(include='object').columns:
                le = LabelEncoder()
                X_missing_test.loc[:, col] = le.fit_transform(X_missing_test[col].astype(str))
        else:
            X_missing_test = None

        if target_is_categorical:
            le_target = LabelEncoder()
            y_fit = le_target.fit_transform(y_fit.astype(str))

        model.fit(X_fit, y_fit)

        # --- SCORING ---
        if target_is_categorical:
            score_array = cross_val_score(model, X_fit, y_fit, cv=cv, scoring='accuracy')
            score_value = round(np.mean(score_array)*100, 2)
            metric_used = 'accuracy (%)'
        else:
            rmse_scores = -cross_val_score(model, X_fit, y_fit, cv=cv, scoring='neg_root_mean_squared_error')
            score_value = round(np.mean(rmse_scores), 2)
            metric_used = 'rmse'

        scores.append({
            'feature': target_col,
            'metric': metric_used,
            'score': score_value
        })

        # --- IMPUTATION ---
        if X_missing_train is not None:
            preds_train = model.predict(X_missing_train)
            if target_is_categorical:
                preds_train = le_target.inverse_transform(preds_train)
            df_train.loc[df_train[target_col].isna(), target_col] = preds_train

        if X_missing_test is not None:
            preds_test = model.predict(X_missing_test)
            if target_is_categorical:
                preds_test = le_target.inverse_transform(preds_test)
            df_test.loc[df_test[target_col].isna(), target_col] = preds_test

    scores_df = pd.DataFrame(scores)

    return df_train, df_test, scores_df

def impute_missing_values(df_train, df_test=None,  target=None, prop_nan=None, corr_mat=None, cv=5):
    """
    Imputation avancée des valeurs manquantes :
    - Variables numériques faiblement corrélées => MultiParametricImputer (échantillonnage paramétrique normal)
    - Autres variables => imputation supervisée (arbre de décision)

    Args:
        df_train (pd.DataFrame): DataFrame d'entraînement
        df_test (pd.DataFrame, optional): DataFrame de test
        prop_nan (pd.DataFrame): Table des proportions de NaN et types des variables
        corr_mat (pd.DataFrame): Matrice de corrélation (%) des patterns de NaN
        cv (int): Nombre de folds pour la cross-validation de l'imputation supervisée
        target (str, optional): Nom de la variable cible à exclure pendant l'imputation

    Returns:
        df_train_imputed, df_test_imputed (ou None), scores_supervised, imputation_report
    """
    if prop_nan is None or corr_mat is None:
        raise ValueError("Les tables prop_nan et corr_mat doivent être fournies toutes les deux.")

    df_train = df_train.copy()
    df_test = df_test.copy() if df_test is not None else None

    # --- Initialisation du rapport d'imputation ---
    imputation_report = []

    # --- Retirer la variable cible des bases de données ---
    if target in df_train.columns:
        val_target_train = df_train[target].copy()
        df_train = df_train.drop(columns=[target])
    
    if df_test is not None:
        if target in df_test.columns:
            val_target_test = df_test[target].copy()
            df_test = df_test.drop(columns=[target])
        else: 
            val_target_test = None

    # --- Sélection des variables peu corrélées ---
    low_corr_features = []
    for feature in corr_mat.columns:
        # Si la variable a des corrélations faibles (<20%) avec toutes les autres
        if (corr_mat[feature].drop(labels=[feature]).abs() <= 20).all():
            low_corr_features.append(feature)

    # Vérification qu'elles sont bien numériques
    low_corr_features = [
        feature for feature in low_corr_features
        if ('float' in prop_nan.loc[feature, 'Type'] or 'int' in prop_nan.loc[feature, 'Type']) and prop_nan.loc[feature, 'NaN proportion'] > 0
    ]

    # Les autres
    other_features = [f for f in prop_nan.index if f not in low_corr_features and prop_nan.loc[f, 'NaN proportion'] > 0]

    # --- Imputation paramétrique ---
    if low_corr_features:
        parametric_imputer = MultiParametricImputer(distribution='norm')
        parametric_imputer.fit(df_train[low_corr_features])
        df_train[low_corr_features] = parametric_imputer.transform(df_train[low_corr_features])
        if df_test is not None:
            df_test[low_corr_features] = parametric_imputer.transform(df_test[low_corr_features])

        # Clipping pour éviter les envolées lyriques
        min_max_dict = {col: (df_train[col].min(), df_train[col].max()) for col in low_corr_features}
        for col, (min_val, max_val) in min_max_dict.items():
            df_train[col] = df_train[col].clip(lower=min_val, upper=max_val)
            if df_test is not None:
                df_test[col] = df_test[col].clip(lower=min_val, upper=max_val)

        # Ajout au rapport
        for feature in low_corr_features:
            imputation_report.append({
                'feature': feature,
                'method': 'Parametric Imputation',
                'distribution': parametric_imputer.imputers[feature].distribution,
                'params': parametric_imputer.imputers[feature].params,
                'base': 'train'
            })
        if df_test is not None:
            for feature in low_corr_features:
                imputation_report.append({
                    'feature': feature,
                    'method': 'Parametric Imputation',
                    'distribution': parametric_imputer.imputers[feature].distribution,
                    'params': parametric_imputer.imputers[feature].params,
                    'base': 'test'
                })

    # --- Imputation supervisée ---
    if other_features:
        df_train, df_test, scores_supervised = impute_from_supervised(
            df_train, df_test, other_features, cv=cv
        )
        if df_test is not None and (df_test is df_train):
            df_test = None

        # Ajout au rapport pour les variables imputation supervisée
        for feature in other_features:
            imputation_report.append({
                'feature': feature,
                'method': 'Supervised Imputation (Decision Tree)',
                'base': 'train'
            })
        if df_test is not None:
            for feature in other_features:
                imputation_report.append({
                    'feature': feature,
                    'method': 'Supervised Imputation (Decision Tree)',
                    'base': 'test'
                })

    else:
        scores_supervised = pd.DataFrame(columns=['feature', 'metric', 'score'])

    # --- Ajouter la variable cible de retour si nécessaire ---
    if target:
        df_train[target] = val_target_train
        if df_test is not None and val_target_test is not None:
            df_test[target] = val_target_test

    # Conversion du rapport en DataFrame
    imputation_report = pd.DataFrame(imputation_report)

    return df_train, df_test, scores_supervised, imputation_report

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
        model = lgb.LGBMRegressor(random_state=42, **param, verbose=-1, n_jobs=-1) if task == 'Regression' else lgb.LGBMClassifier(random_state=42, **param, verbose=-1, n_jobs=-1)

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
        model = xgb.XGBRegressor(random_state=42, **param, n_jobs=-1) if task == 'Regression' else xgb.XGBClassifier(random_state=42, **param, n_jobs=-1)

    elif model_type == "Random Forest":
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 5, 50),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features': trial.suggest_categorical('max_features', [None, 'sqrt', 'log2'])
        }
        model = RandomForestRegressor(random_state=42, **param, n_jobs=-1) if task == 'Regression' else RandomForestClassifier(random_state=42, **param, n_jobs=-1)

    elif model_type == "Linear Regression":
        # Définition des hyperparamètres pour la régression linéaire et les modèles régularisés
        model_linreg = trial.suggest_categorical("model", ["linear", "ridge", "lasso", "elasticnet"])
    
        if model_linreg == "linear":
            model = LinearRegression()
        
        elif model_linreg == "ridge":
            ridge_alpha = trial.suggest_float("ridge_alpha", 0.01, 10.01, log=True)
            ridge_alpha = round(ridge_alpha, 2)
            model = Ridge(alpha=ridge_alpha, random_state=42)

        elif model_linreg == "lasso":
            lasso_alpha = trial.suggest_float("lasso_alpha", 0.01, 10.01, log=True)
            lasso_alpha = round(lasso_alpha, 2)
            model = Lasso(alpha=lasso_alpha, random_state=42)

        elif model_linreg == "elasticnet":
            enet_alpha = trial.suggest_float("enet_alpha", 0.01, 10.01, log=True)
            l1_ratio = trial.suggest_float("l1_ratio", 0, 1.0, step=0.01)
            enet_alpha = round(enet_alpha, 2)
            l1_ratio = round(l1_ratio, 2)
            model = ElasticNet(alpha=enet_alpha, l1_ratio=l1_ratio, random_state=42)

    elif model_type == "Logistic Regression":
        # Paramètres pour la régression logistique
        penalty = trial.suggest_categorical("penalty", ["l2", "l1", "elasticnet", None])
        C = trial.suggest_float("C", 1e-3, 10.001, step=0.01)
        C = round(C, 3)
        
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0, 1, step=0.01)
            l1_ratio = round(l1_ratio, 2)
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
    study = optuna.create_study(direction='maximize',
                                sampler=TPESampler(prior_weight=0.5, n_startup_trials=10,
                                                   n_ei_candidates=12,warn_independent_sampling=False,
                                                   seed=42),
                                pruner=SuccessiveHalvingPruner(min_resource=1, reduction_factor=3, min_early_stopping_rate=0, bootstrap_count=0))
    study.optimize(lambda trial: objective(trial, task=task, model_type=model_choosen, multi_class=multi_class, X=X_train, y=y_train, cv=cv, scoring_comp=scoring), n_trials=n_trials, n_jobs=n_jobs)
    
    # Créer le modèle avec les meilleurs hyperparamètres
    if model_choosen == "LightGBM":
        best_model = lgb.LGBMRegressor(**study.best_params, verbose=-1) if task == 'Regression' else lgb.LGBMClassifier(**study.best_params, verbose=-1)
    elif model_choosen == "XGBoost":
        best_model = xgb.XGBRegressor(**study.best_params) if task == 'Regression' else xgb.XGBClassifier(**study.best_params)
    elif model_choosen == "Random Forest":
        best_model = RandomForestRegressor(**study.best_params) if task == 'Regression' else RandomForestClassifier(**study.best_params)
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

def evaluate_and_decompose(estimator, X, y, scoring_dict, task, cv=5, random_seed=None):
    # Initialisation
    rng = np.random.RandomState(random_seed)
    kf = KFold(n_splits=cv, shuffle=True, random_state=rng)
    
    # Convertir X et y en tableaux numpy pour éviter l'erreur
    X = X.to_numpy() if isinstance(X, pd.DataFrame) else X
    y = y.to_numpy() if isinstance(y, pd.Series) else y

    # Variables pour stocker les prédictions et les vraies valeurs
    all_pred = []
    y_tests = []
    
    # Variables pour stocker les scores de chaque pli
    fold_scores = {metric: [] for metric in scoring_dict}
    
    # Boucle sur les folds de validation croisée
    for train_idx, test_idx in kf.split(X):
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # Entraînement du modèle
        model = estimator.fit(X_train_fold, y_train_fold)
        preds = model.predict(X_test_fold)
        
        # Stockage des prédictions et des vraies valeurs
        all_pred.append(preds)
        y_tests.append(y_test_fold)
        
        # Calcul des scores pour chaque métrique
        for metric, scorer in scoring_dict.items():
            score = scorer(estimator, X_test_fold, y_test_fold)
            fold_scores[metric].append(score)

    # Concatenate predictions and true values for global analysis
    all_pred = np.concatenate(all_pred)
    y_tests = np.concatenate(y_tests)

    # Calcul des scores moyens et des écarts types pour chaque métrique
    mean_scores = {metric: np.mean(fold_scores[metric]) for metric in scoring_dict}
    std_scores = {metric: np.std(fold_scores[metric]).round(5) for metric in scoring_dict}

    # Calcul du biais et de la variance
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

    return mean_scores, std_scores, avg_expected_loss, avg_bias, avg_var, bias_relative, var_relative

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
use_loocv=False
scoring_comp = "neg_root_mean_squared_error"
scoring_eval = ['neg_root_mean_squared_error']
models = ["Linear Regression", 'KNN']
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
if use_loocv:
    cv = X_train.shape[0]

# 6. Choisir le meilleur modèle
results = []
for model in models:  
    # Déterminer chaque modèle à optimiser
    best_model, best_params, best_value = optimize_model(model_choosen=model, task=task,
                                                         X_train=X_train, y_train=y_train,
                                                         cv=cv, scoring=scoring_comp,
                                                         multi_class=multi_class,
                                                         n_trials=5, n_jobs=-1)
    
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

# Dictionnaire des métriques    
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
    scoring_dict = metrics_regression
else:
    scoring_dict = metrics_classification

all_results = []    
for model in list_models:
    mean_scores, std_scores, avg_expected_loss, avg_bias, avg_var, bias_relative, var_relative = evaluate_and_decompose(
        estimator=model,  # Ton modèle
        X=X_test,              # Données d'entraînement
        y=y_test,              # Cibles
        scoring_dict=scoring_dict,  # Dictionnaire des métriques
        task=task,  # Ou "Regression" si tu utilises un modèle de régression
        cv=cv,  # Nombre de splits pour la validation croisée
        random_seed=42  # Graine pour la reproductibilité
    )
    
    all_results.append({
        'Model': str(model),
        'Scores': mean_scores,
        'Std Scores': std_scores,
        'Avg Expected Loss': avg_expected_loss,
        'Avg Bias': avg_bias,
        'Avg Variance': avg_var,
        'Bias Relative': bias_relative,
        'Variance Relative': var_relative
    })
    
final_results_df = pd.DataFrame(all_results)

# 9. Afficher la matrice de confusion  
if task=='Classification':
    for index, model in df_score['Best Model'].items():
        model = instance_model(idx, df_train2, task)
        model.fit(X_train, y_train)
        
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
    model.fit(X_train, y_train)
    
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
