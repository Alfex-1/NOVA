# Imports
import numpy as np
import pandas as pd
import polars as pl
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, PowerTransformer, LabelEncoder
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency
from sklearn.ensemble import RandomForestRegressor, IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet, LinearRegression, LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split, cross_validate, learning_curve, cross_val_score
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.multiclass import type_of_target
from sklearn.inspection import permutation_importance
import xgboost as xgb
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import SuccessiveHalvingPruner
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_absolute_percentage_error
import io
from io import BytesIO
import os
import streamlit as st
from PIL import Image
import zipfile
import shap
from lime.lime_tabular import LimeTabularExplainer
import streamlit.components.v1 as components
import networkx as nx

def load_file(file_data):
    """
    Lit un fichier CSV à partir d'un objet binaire en détectant automatiquement le séparateur.

    Args:
        file_data (BinaryIO): Fichier binaire téléchargé (par exemple via une interface Streamlit).

    Returns:
        pl.DataFrame | None: Un DataFrame Polars si un séparateur valide est détecté, sinon None.
    """

    byte_data = file_data.read()
    separators = [";", ",", "\t"]
    detected_sep = None

    for sep in separators:
        try:
            tmp_df = pl.read_csv(BytesIO(byte_data), separator=sep, n_rows=20)
            if tmp_df.shape[1] > 1:
                detected_sep = sep
                break
        except Exception:
            continue

    if detected_sep is not None:
        return pl.read_csv(BytesIO(byte_data), separator=detected_sep)
    else:
        return None

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

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape[0] < 2 or confusion_matrix.shape[1] < 2:
        return 0
    chi2 = chi2_contingency(confusion_matrix, correction=False)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def build_cramers_matrix(df_cat):
    cols = df_cat.columns
    n = len(cols)
    matrix = pd.DataFrame(np.eye(n), columns=cols, index=cols)
    for i in range(n):
        for j in range(i + 1, n):
            v = cramers_v(df_cat.iloc[:, i], df_cat.iloc[:, j])
            matrix.iloc[i, j] = matrix.iloc[j, i] = v
    return matrix

def select_representative_categorial(df, target, threshold=0.9):
    df_cat = df.select_dtypes(include=['object', 'category']).copy()
    
    if not df_cat.empty:
        v_matrix = build_cramers_matrix(df_cat)

        # Créer le graphe pondéré
        G = nx.Graph()
        for col in df_cat.columns:
            G.add_node(col)
        for i, var1 in enumerate(df_cat.columns):
            for j, var2 in enumerate(df_cat.columns):
                if i >= j: continue
                v = v_matrix.loc[var1, var2]
                if v >= threshold:
                    G.add_edge(var1, var2, weight=v)

        # Info mutuelle
        df_encoded = df_cat.apply(lambda col: col.astype("category").cat.codes)
        target_type = type_of_target(df[target])
        if "continuous" in target_type:
            mi_scores = mutual_info_regression(df_encoded, df[target])
        else:
            mi_scores = mutual_info_classif(df_encoded, df[target], discrete_features=True)
        
        mi_dict = dict(zip(df_cat.columns, mi_scores))

        # Sélection
        to_keep = set()
        for component in nx.connected_components(G):
            best = max(component, key=lambda var: mi_dict.get(var, 0)) if len(component) > 1 else list(component)[0]
            to_keep.add(best)

        to_drop = set(df_cat.columns) - to_keep

        # Création de la figure
        fig = None
        if G.number_of_edges() > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
            edge_cmap = plt.cm.Blues
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_weights, edge_cmap=edge_cmap, 
                                edge_vmin=norm.vmin, edge_vmax=norm.vmax, width=2)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#90ee90", node_size=1800, edgecolors='black', linewidths=0.5)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight='bold')
            sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Force d'association", rotation=270, labelpad=15)
            plt.title(f"Réseau des variables catégorielles fortement associées (V ≥ {int(threshold*100)}%)", fontsize=13)
            plt.axis("off")
            plt.tight_layout()
        
        return list(to_drop), fig
    
    else:
        return [], None

def select_representative_numerical(df, target, threshold=0.9):
    df_num = df.select_dtypes(include=['int', 'float']).copy()
    
    if not df_num.empty:
    
        corr_matrix = df_num.corr().abs()

        # Construction du graphe
        G = nx.Graph()
        for col in df_num.columns:
            G.add_node(col)
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                corr_val = corr_matrix.iloc[i, j]
                if corr_val >= threshold:
                    G.add_edge(var1, var2, weight=corr_val)

        # Info mutuelle
        y = df[target]
        target_type = type_of_target(y)
        if target_type in ['binary', 'multiclass', 'multiclass-multioutput']:
            mi_scores = mutual_info_classif(df_num.fillna(0), y, discrete_features=False)
        else:
            mi_scores = mutual_info_regression(df_num.fillna(0), y)

        mi_dict = dict(zip(df_num.columns, mi_scores))

        # Sélection
        to_keep = set()
        for component in nx.connected_components(G):
            best = max(component, key=lambda var: mi_dict.get(var, 0)) if len(component) > 1 else list(component)[0]
            to_keep.add(best)

        to_drop = set(df_num.columns) - to_keep

        # Création de la figure
        fig = None
        if G.number_of_edges() > 0:
            fig, ax = plt.subplots(figsize=(10, 8))
            pos = nx.spring_layout(G, seed=42)
            edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
            norm = mcolors.Normalize(vmin=min(edge_weights), vmax=max(edge_weights))
            edge_cmap = plt.cm.Blues
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_weights, edge_cmap=edge_cmap, 
                                edge_vmin=norm.vmin, edge_vmax=norm.vmax, width=2)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color="#90ee90", node_size=1800, edgecolors='black', linewidths=0.5)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=11, font_weight='bold')
            sm = plt.cm.ScalarMappable(cmap=edge_cmap, norm=norm)
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax)
            cbar.set_label("Force de corrélation", rotation=270, labelpad=15)
            plt.title(f"Réseau des variables numériques fortement corrélées (ρ ≥ {int(threshold*100)}%)", fontsize=13)
            plt.axis("off")
            plt.tight_layout()
                    
        return list(to_drop), fig
    
    else:
        return [], None


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

    return df_train, df_test    
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

        p_values = {}

        # Normale
        mu, sigma = stats.norm.fit(data)
        _, p_norm = stats.kstest(data, 'norm', args=(mu, sigma))
        p_values['norm'] = p_norm

        # Uniforme
        a, b = np.min(data), np.max(data)
        _, p_unif = stats.kstest(data, 'uniform', args=(a, b - a))
        p_values['uniform'] = p_unif

        # Exponentielle
        lambda_hat = 1 / data.mean()
        _, p_exp = stats.kstest(data, 'expon', args=(0, lambda_hat))
        p_values['expon'] = p_exp

        # Log-normale (que si data > 0)
        if (data > 0).all():
            log_data = np.log(data)
            mu_log, sigma_log = stats.norm.fit(log_data)
            _, p_lognorm = stats.kstest(data, 'lognorm', args=(sigma_log, 0, np.exp(mu_log)))
            p_values['lognorm'] = p_lognorm

        # Meilleure distribution = max p-value
        best_dist = max(p_values, key=p_values.get)

        if best_dist == 'norm':
            self.distribution = 'norm'
            self.params = {'mu': mu, 'sigma': sigma}
        elif best_dist == 'uniform':
            self.distribution = 'uniform'
            self.params = {'a': a, 'b': b}
        elif best_dist == 'expon':
            self.distribution = 'expon'
            self.params = {'lambda': lambda_hat}
        elif best_dist == 'lognorm':
            self.distribution = 'lognorm'
            self.params = {'mu': mu_log, 'sigma': sigma_log}
        else:
            raise RuntimeError("Distribution sélectionnée non reconnue.")

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
        elif self.distribution == 'lognorm':
            return rng.lognormal(mean=self.params['mu'], sigma=self.params['sigma'], size=size)
        elif  self.distribution == 'expon':
            return rng.exponential(scale=1 / self.params['lambda'], size=size)
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
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.imputers = {}
        self.fitted = False
        self.imputed_info = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("L'entrée doit être un DataFrame pandas.")
        for col in X.columns:
            imputer = ParametricImputer(random_state=self.random_state)
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

    
def impute_from_supervised(df_train, cols_to_impute, df_test=None):
    """
    Impute les valeurs manquantes des colonnes sélectionnées en utilisant des modèles supervisés (arbres de décision).

    Pour chaque colonne cible, entraîne un arbre de décision (classifieur pour les variables catégorielles, régressseur pour les variables continues)
    sur les données connues, puis impute les valeurs manquantes dans les ensembles d'entraînement et de test.

    Args:
        df_train (pd.DataFrame): Jeu de données d'entraînement contenant des valeurs manquantes.
        df_test (pd.DataFrame ou None): Jeu de données de test contenant des valeurs manquantes. Peut être None si indisponible.
        cols_to_impute (list of str): Liste des noms de colonnes à imputer.

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
        if df_test is not None:
            test_missing = df_test[df_test[target_col].isna()].copy()
        else:
            test_missing = None

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

        if test_missing is not None:
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
        
        # --- Détermine le nombre de splits CV ---
        if target_is_categorical:
            class_counts = pd.Series(y_fit).value_counts()
            if class_counts.min() < 5:
                cv = 1
            else:
                cv = 5
        else:
            cv = 5 if len(y_fit) >= 5 else 1

        # --- SCORING ---
        if target_is_categorical:
            if cv > 1:
                f1_macro = cross_val_score(model, X_fit, y_fit, cv=cv, scoring='f1_macro')
                score_value = round(np.mean(f1_macro)*100, 2)
            else:
                model.fit(X_fit, y_fit)
                y_pred = model.predict(X_fit)
                score_value = round(f1_score(y_fit, y_pred, average='weighted')*100, 2)
            metric_used = 'F1-score (weighted) (%)'
        else:
            use_mape = not np.any(y_fit == 0)
            if use_mape:
                if cv > 1:
                    mae_scores = -cross_val_score(model, X_fit, y_fit, cv=cv, scoring='neg_mean_absolute_percentage_error')
                    score_value = round(np.mean(mae_scores), 2)
                else:
                    model.fit(X_fit, y_fit)
                    y_pred = model.predict(X_fit)
                    score_value = round(mean_absolute_percentage_error(y_fit, y_pred)*100, 2)
                metric_used = 'Mean Absolute Percentage Error (%)'
            else:
                if cv > 1:
                    mae_scores = -cross_val_score(model, X_fit, y_fit, cv=cv, scoring='neg_mean_absolute_error')
                    score_value = round(np.mean(mae_scores), 4)
                else:
                    model.fit(X_fit, y_fit)
                    y_pred = model.predict(X_fit)
                    score_value = round(mean_absolute_error(y_fit, y_pred), 4)
                metric_used = 'Mean Absolute Error'

        scores.append({
            'Variable': target_col,
            'Métrique': metric_used,
            'Score': score_value
        })
        scores_df = pd.DataFrame(scores)

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

    return df_train, df_test, scores_df

def impute_missing_values(df_train, df_test=None,  target=None, prop_nan=None, corr_mat=None):
    """
    Imputation avancée des valeurs manquantes :
    - Variables numériques faiblement corrélées => MultiParametricImputer (échantillonnage paramétrique normal)
    - Autres variables => imputation supervisée (arbre de décision)

    Args:
        df_train (pd.DataFrame): DataFrame d'entraînement
        df_test (pd.DataFrame, optional): DataFrame de test
        prop_nan (pd.DataFrame): Table des proportions de NaN et types des variables
        corr_mat (pd.DataFrame): Matrice de corrélation (%) des patterns de NaN
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

    # Retirer la target si elle est dans les listes
    if 'target' in locals() and target:
        if target in low_corr_features:
            low_corr_features.remove(target)
        if target in other_features:
            other_features.remove(target)
    
    # --- Imputation paramétrique ---
    if low_corr_features:
        parametric_imputer = MultiParametricImputer()
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
                'Variable': feature,
                'Méthode': 'Parametric Imputation',
                'Distribution': parametric_imputer.imputers[feature].distribution,
                'Paramètres': parametric_imputer.imputers[feature].params})

    # --- Imputation supervisée ---
    if other_features:
        df_train, df_test, scores_supervised = impute_from_supervised(
            df_train, df_test=df_test, cols_to_impute=other_features)
        if df_test is not None and (df_test is df_train):
            df_test = None

        # Ajout au rapport pour les variables imputation supervisée
        for feature in other_features:
            imputation_report.append({
                'Variable': feature,
                'Méthode': 'Imputation supervisé par arbre de décision'})

    else:
        scores_supervised = pd.DataFrame(columns=['Variable', 'Métrique', 'Score'])

    # --- Ajouter la variable cible de retour si nécessaire ---
    if target:
        df_train[target] = val_target_train
        if df_test is not None and val_target_test is not None:
            df_test[target] = val_target_test

    # Conversion du rapport en DataFrame
    imputation_report = pd.DataFrame(imputation_report)

    return df_train, df_test, scores_supervised, imputation_report

def detect_and_winsorize(df_train: pd.DataFrame, df_test: pd.DataFrame = None, target: str = None, contamination: float = 0.01):
    """
    Détecte les outliers sur df_train avec Isolation Forest + LOF, winsorize les variables numériques.
    Si df_test est fourni, applique la même winsorization dessus.
    
    Args:
        df_train (pd.DataFrame): Base d'entraînement.
        target (str): Nom de la cible à exclure.
        df_test (pd.DataFrame, optional): Base de test. Si None, seul df_train est traité.
        contamination (float): Contamination supposée pour IsolationForest et LOF.

    Returns:
        (df_train_winsorized, df_test_winsorized (si fourni sinon None), nombre_total_modifications (int))
    """
    df_train = df_train.copy()
    df_test = df_test.copy() if df_test is not None else None

    # Extraire uniquement les variables numériques
    features = df_train.drop(columns=[target], errors='ignore').select_dtypes(include=[np.number])

    # Virer les lignes où il manque des valeurs (seulement pour l'outlier detection)
    valid_idx = features.dropna().index
    features_valid = features.loc[valid_idx]

    if features_valid.shape[1] == 0:
        raise ValueError("Aucune variable numérique exploitable dans df_train.")
    if features_valid.shape[0] < 10:
        raise ValueError("Pas assez de données valides pour détecter les outliers.")

    # Détection sur les données sans NaN
    iso = IsolationForest(n_estimators=500, contamination=contamination, random_state=42, n_jobs=-1)
    out_iso = iso.fit_predict(features_valid)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=contamination, n_jobs=-1)
    out_lof = lof.fit_predict(features_valid)

    # Fusion des deux détecteurs
    outliers = ((out_iso == -1) & (out_lof == -1)).astype(int)
    outliers = pd.Series(outliers, index=features_valid.index)

    # Définir bornes winsorization sur les données sans outliers
    bounds = {}
    for col in features.columns:
        if col in features_valid.columns:
            mask = (outliers == 1)
            if mask.any():
                lower = np.percentile(features_valid.loc[~mask, col], 1)
                upper = np.percentile(features_valid.loc[~mask, col], 99)
            else:
                lower = np.percentile(features_valid[col], 1)
                upper = np.percentile(features_valid[col], 99)
            bounds[col] = (lower, upper)

    # Winsorize train
    n_modif_train = 0
    for col, (lower, upper) in bounds.items():
        if col in df_train.columns:
            original = df_train[col].copy()
            df_train[col] = np.clip(df_train[col], lower, upper)
            n_modif_train += (original != df_train[col]).sum()

    # Winsorize test si dispo
    n_modif_test = 0
    if df_test is not None:
        for col, (lower, upper) in bounds.items():
            if col in df_test.columns:
                original = df_test[col].copy()
                df_test[col] = np.clip(df_test[col], lower, upper)
                n_modif_test += (original != df_test[col]).sum()

    nb_outliers = int(n_modif_train + n_modif_test)

    if df_test is not None:
        return df_train, df_test, nb_outliers
    else:
        return df_train, nb_outliers

def transform_data(df_train: pd.DataFrame, df_test: pd.DataFrame = None, list_boxcox: list[str] = None, list_yeo: list[str] = None, list_log: list[str] = None, list_sqrt: list[str] = None):
    """
    Applique des transformations statistiques (Box-Cox, Yeo-Johnson, Logarithme, Racine carrée) sur les colonnes spécifiées.
    
    Args:
        df_train (pd.DataFrame): DataFrame d'entraînement.
        df_test (pd.DataFrame, optional): DataFrame de test, transformé avec les mêmes paramètres que df_train.
        list_boxcox (list[str], optional): Colonnes pour transformation Box-Cox (valeurs > 0).
        list_yeo (list[str], optional): Colonnes pour transformation Yeo-Johnson (valeurs quelconques).
        list_log (list[str], optional): Colonnes pour transformation Log (valeurs > 0).
        list_sqrt (list[str], optional): Colonnes pour transformation Racine carrée (valeurs ≥ 0).

    Returns:
        tuple: (df_train transformé, df_test transformé si fourni sinon seulement df_train transformé)
    """

    df_train = df_train.copy()
    df_test = df_test.copy() if df_test is not None else None

    def apply_transform(transformer, cols):
        transformer.fit(df_train[cols])
        df_train[cols] = transformer.transform(df_train[cols])
        if df_test is not None:
            df_test[cols] = transformer.transform(df_test[cols])

    def simple_transform(func, cols, condition=lambda x: x < 0, error_msg="Valeurs invalides"):
        for col in cols:
            if condition(df_train[col]).any():
                raise ValueError(f"{error_msg} pour '{col}'")
            df_train[col] = func(df_train[col])
            if df_test is not None:
                df_test[col] = func(df_test[col])

    # Box-Cox (strictement positif)
    if list_boxcox:
        simple_transform(lambda x: x <= 0, list_boxcox, "Box-Cox nécessite des valeurs > 0")
        apply_transform(PowerTransformer(method='box-cox'), list_boxcox)

    # Yeo-Johnson (n'importe quelle valeur)
    if list_yeo:
        apply_transform(PowerTransformer(method='yeo-johnson'), list_yeo)

    # Log (strictement positif)
    if list_log:
        simple_transform(np.log, list_log, lambda x: x <= 0, "Log nécessite des valeurs > 0")

    # Sqrt (positif ou nul)
    if list_sqrt:
        simple_transform(np.sqrt, list_sqrt, lambda x: x < 0, "Racine carrée nécessite des valeurs ≥ 0")

    return df_train, df_test if df_test is not None else df_train

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
            if multi_class:
                objective = 'multi:softmax'
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

def bias_variance_decomp(estimator, X, y, task, cv=5, random_seed=None):
    """Calcule le biais et la variance d'un estimateur via une décomposition par validation croisée.

    Cette fonction effectue une décomposition du biais et de la variance d'un modèle d'estimation en utilisant 
    la validation croisée (KFold). Elle permet d'évaluer la performance de l'estimateur en termes de biais et 
    de variance en fonction de la tâche (régression ou classification).

    Args:
        estimator (sklearn.base.BaseEstimator): L'estimateur (modèle) à évaluer.
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
    # Initialisation
    rng = np.random.RandomState(random_seed)
    kf = KFold(n_splits=cv, shuffle=True, random_state=rng)

    all_pred = []
    y_tests = []

    # Boucle sur les folds de validation croisée
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
    
    return model

def heatmap_corr(corr_mat):
    mask = np.triu(np.ones_like(corr_mat, dtype=bool))

    n_vars = corr_mat.shape[0]
    figsize = (max(8, n_vars*1.5), max(6, n_vars * 0.8))

    plt.figure(figsize=figsize)
    heatmap = sns.heatmap(
        round(corr_mat, 1),
        mask=mask,
        annot=True,
        cmap='coolwarm',
        annot_kws={"size": max(6, n_vars)},
        cbar_kws={'shrink': 0.8}
    )
    heatmap.collections[0].colorbar.ax.tick_params(labelsize=max(8, 1.25*n_vars))
    plt.xticks(fontsize=max(6, int(n_vars * 0.85)), rotation=90)
    plt.yticks(fontsize=max(6, int(n_vars * 0.85)), rotation=0)
    plt.tight_layout()
    
    return plt

def advance_progress(n_steps_total):
    global current_step
    current_step += 1
    progress_bar.progress(current_step / n_steps_total)

# python -m streamlit run src/app/_main_.py
st.set_page_config(page_title="NOVA", layout="wide")

st.title("✨ NOVA : Numerical Optimization & Validation Assistant")
st.subheader("Votre assistant flexible pour le traitement des données et la modélisation.")

st.write(
    """
    L'assistant **NOVA** vous accompagne dans la préparation et l’expérimentation de vos modèles de Machine Learning.  
    Plus besoinde coder pour des traitements et de la modélisation basiques. Conçue pour les professionnels qui savent que chaque projet est unique, **NOVA** offre des outils automatisés puissants
    pour la gestion des données et l'ajustement des modèles, tout en laissant l'exploration et la personnalisation à votre charge.

    **Fonctionnalités principales :**
    - 🔄 **Prétraitement des données** : mise à l’échelle, encodage, gestion des valeurs manquantes, outliers, et transformations adaptées.
    - 🔍 **Optimisation des hyperparamètres** : recherche des meilleurs réglages pour 7 modèles populaires (régression linéaire/logistique, KNN, Random Forest, LightGBM, XGboost).
    - 🏆 **Évaluation et interprétation des modèles** : validation croisée, analyse biais-variance, importance par permutation, analyse de drift, LIME/SHAPE et matrice de confusion pour les tâches de classification.
    
    **NOVA** permet à chaque utilisateur de bénéficier d’une infrastructure robuste, tout en maintenant une flexibilité totale sur le traitement fondamental des données.
    Vous contrôlez les choix, nous optimisons les outils.
    """
)

# Initialisation des variables
df_train = None
df_test = None
df = None
uploaded_file_train = st.file_uploader("Choisissez un fichier d'entraînement (csv, xlsx, txt)", type=["csv", "xlsx", "txt"], key="train")
uploaded_file_test = st.file_uploader("Choisissez un fichier de validation (csv, xlsx, txt)", type=["csv", "xlsx", "txt"], key="test")
wrang = st.checkbox("La base de données nécessite un traitement")
split_data = False
valid_train = False
valid_test = False
valid_mod=False
valid_wrang=False

# Chargement du fichier d'entraînement
if uploaded_file_train is not None:
    df_train = load_file(uploaded_file_train)
    if isinstance(df_train, pl.DataFrame):
        df_train = df_train.to_pandas()
    if df_train is not None:
        valid_train = True
    else:
        st.warning("Échec de la détection du séparateur pour le fichier d'entraînement. Vérifiez le format du fichier.")

# Chargement du fichier de test
if uploaded_file_test is not None:
    df_test = load_file(uploaded_file_test)
    if isinstance(df_test, pl.DataFrame):
        df_test = df_test.to_pandas()
    if df_test is not None:
        valid_test = True
    else:
        st.warning("Échec de la détection du séparateur pour le fichier de test. Vérifiez le format du fichier.")

if df_train is not None:
    df = df_train.copy()
    del df_train

# Sidebar pour la configuration de l'utilisateur    
if df is not None:
    st.sidebar.image(Image.open("logo_nova.png"), width=200)
    
    if wrang is True:            
        st.sidebar.title("Paramètres du traitement des données")
        
        st.sidebar.subheader("Informations générales")
        
        # Supprimer les colonnes inutiles
        drop_columns = None
        drop_columns = st.sidebar.multiselect("Quelle(s) variable(s) voulez-vous supprimer instantanément ?", df.columns, help="Sélectionnez les colonnes que vous souhaitez supprimer de la base de données.")
        
        # Demander la variable cible
        if drop_columns is not None and len(drop_columns) > 0:
            target = st.sidebar.selectbox("Choisissez la variable cible de votre future modélisation", df.drop(columns=drop_columns).columns, help="Si vous n'avez pas de variable cible, choisissez une variable au harsard.")
        else:    
            target = st.sidebar.selectbox("Choisissez la variable cible de votre future modélisation", df.columns, help="Si vous n'avez pas de variable cible, choisissez une variable au harsard.")
        
        if df_test is None:
            # Demander s'il faut demander de diviser la base
            split_data = st.sidebar.checkbox("Diviser la base de données en apprentissage/validation ?", value=True, help="La division des données durant leur traitement est fondamentale pour éviter la fuite de données lors de votre modélisation.")
            
            if split_data:
                train_size = st.sidebar.slider("Proportion des données utilisées pour l'apprentissage des modèles (en %)", min_value=50, max_value=90, value=75)
                train_size = train_size/100
                df_train, df_test = train_test_split(df, train_size=train_size, shuffle=True, random_state=42)
        
        # Demander si l'utilisateur souhaite supprimer les doublons
        drop_dupli = st.sidebar.checkbox("Supprimer toutes les observations dupliquées", value=False)
        
        pb = False
        wrang_finished = False
        
        st.sidebar.subheader("Contrôle des variables")
        
        # Demander si l'utilisateur veut supprimer les variables redondantes
        drop_redundant = st.sidebar.checkbox("Supprimer les variables redondantes", value=False)
        if drop_redundant:
            threshold = st.sidebar.slider("Seuil de redondance (corrélation ou V de Cramér), en %", min_value=0, max_value=100, value=75, help="Plus la valeur est proche de 1, plus il y a une redondance entre les variables.")
            threshold = threshold / 100
        
        st.sidebar.subheader("Contrôle des individus")
        
        # Outliers
        wrang_outliers = st.sidebar.checkbox("Voulez-vous traiter les valeurs aberrantes/outliers ?", value=False)
        
        if wrang_outliers:
            contamination = st.sidebar.slider("Proportion des individus que vous suspectez d'être des outliers en (%)", min_value=0, max_value=100, value=0, help="Si vous n'en avez aucune idée, laissez à 0")
            if contamination == 0:
                contamination = 'auto'
            else:
                contamination = contamination/100
        
        st.sidebar.subheader("Mise à l'échelle des variables numériques")
        
        # Déterminer si la variable cible doit être incluse dans la mise à l'échelle
        use_target = st.sidebar.checkbox("Inclure la variable cible dans la mise à l'échelle", value=False, help="Si vous avez une variable cible, ne cochez pas cette case, sinon cochez-là")
        
        if not use_target:
            if target and target in df.columns:
                if split_data:
                    df_to_wrang = df_train.drop(columns=target)
                else:
                    df_to_wrang = df.drop(columns=target)
            else:
                st.warning("La variable cible est invalide ou non définie.")
                pb = True
        else:
            df_to_wrang = df if df_test is None and not split_data else df_train
        
        # Tout mettre à l'échelle directement
        scale_all_data = st.sidebar.checkbox("Voulez-vous mettre à l'échelle vos données ?")
        
        if scale_all_data:
            scale_method = st.sidebar.selectbox("Méthode de mise à l'échelle à appliquer",
                                                ["Standard Scaler", "MinMax Scaler", "Robust Scaler", "Quantile Transformer (Uniform)"])
            if scale_method:
                if scale_method == "Standard Scaler":
                    scaler = StandardScaler()
                elif scale_method == "MinMax Scaler":
                    scaler = MinMaxScaler()
                elif scale_method == "Robust Scaler":
                    scaler = RobustScaler()
                else:
                    scaler = QuantileTransformer(output_distribution='uniform')
        
        # Obtenir des dataframes distinctes selon les types des données            
        df_num = df_to_wrang.select_dtypes(include=['number'])
        df_cat = df_to_wrang.drop(columns=drop_columns).select_dtypes(exclude=['number']) if len(drop_columns) > 0 else df_to_wrang.select_dtypes(exclude=['number'])

        # Sélection des variables à encoder
        have_to_encode = False
        if df_cat.shape[1] > 0:
            have_to_encode = True
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
            strictly_positive_vars = df_num.drop(columns=drop_columns, errors='ignore').loc[:, (df_num.drop(columns=drop_columns, errors='ignore') > 0).all()].columns.to_list()
            # Déterminer les variables positives ou nulles
            positive_or_zero_vars = df_num.drop(columns=drop_columns, errors='ignore').loc[:, (df_num.drop(columns=drop_columns, errors='ignore') >= 0).all()].columns.to_list()
            
            list_boxcox = None
            list_yeo = None
            list_log = None
            list_sqrt = None            
            list_boxcox = st.sidebar.multiselect("Variables à transformer avec Box-Cox", strictly_positive_vars)
            list_yeo = st.sidebar.multiselect("Variables à transformer avec Yeo-Johnson", df_num.drop(columns=drop_columns, errors='ignore').columns.to_list())
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
                df_num_acp = df.select_dtypes(include=['number'])
                n_components = st.sidebar.slider("Nombre de composantes principales", min_value=1, max_value=df_num_acp.shape[1]-1, value=1, help="Il se peut que le nombre de composantes conservées diminue si des variables venaient à être supprimées durant le traitement.")
            elif pca_option == "Variance expliquée":
                explained_variance = st.sidebar.slider("Variance expliquée à conserver (%)", min_value=00, max_value=100, value=95)
        
        # Valider les choix
        valid_wrang = st.sidebar.button("Valider les choix de traitement")
    
    else:
        # Modélisation
        st.sidebar.title("Paramètres de Modélisation")

        # Définition de la variable cible
        target = st.sidebar.selectbox("Choisissez la variable cible", df.columns.to_list())
        
        if df_test is None:
            # Division des données (si non déjà fait)
            train_size = st.sidebar.slider("Proportion des données utilisées pour l'apprentissage des modèles (en %)", min_value=50, max_value=90, value=75)
            train_size=train_size/100
 
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
            models = st.sidebar.multiselect("Modèle(s) à tester", ["Linear Regression", "KNN", "Random Forest", "XGBoost", "LightGBM"], default=["Linear Regression"])
        else:
            models = st.sidebar.multiselect("Modèle(s) à tester", ["Logistic Regression", "KNN", "Random Forest", "XGBoost", "LightGBM"], default=["Logistic Regression"])
            
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
        use_loocv = st.sidebar.checkbox("Utiliser une seule observation par évaluation", help="Recommandé pour les petits ensembles de données uniquement")

        # Si LOO-CV est coché, le champ des folds est désactivé
        if not use_loocv:
            cv = st.sidebar.number_input(
                "Nombre de folds (CV)",
                min_value=2, max_value=20,
                value=7, step=1,
                disabled=use_loocv)
            
        st.sidebar.subheader("Enregistrement des modèles")
        # Demander à l'utilisateur où il souhaite enregistrer les modèles
        base_dir = st.sidebar.text_input("Entrez le chemin du dossier qui contiendra les modèles enregistrés", help="Exemple : C:\\Users\\Documents")
        
        # Valider les choix
        valid_mod = st.sidebar.button("Valider les choix de modélisation")

if valid_wrang:    
    # Faire les traitements selon si split_data = True
    df_test_exists = 'df_test' in globals() and df_test is not None and not df_test.empty
    split_data_val = globals().get('split_data', False)

    if df_test_exists or split_data_val:
        progress_bar = st.progress(0)
        n_steps_total = 11
        current_step = 0
        
        # Suppression des colonnes inutiles
        with st.spinner("Suppression des colonnes inutiles..."):
            if drop_columns:
                df = df.drop(columns=drop_columns)
                df_test = df_test.drop(columns=drop_columns)
        
        advance_progress(n_steps_total)
           
        # Suppression des doublons
        with st.spinner("Suppression des doublons..."):
            if drop_dupli:
                len_before_dupli = len(df)
                df = df.drop_duplicates()
                len_after_dupli = len(df)
                len_diff = len_before_dupli - len_after_dupli
            else:
                len_diff = "Les doublons n'ont pas été traités."
        
        advance_progress(n_steps_total)     
        
        # Etude des valeurs manquantes
        with st.spinner("Etude des valeurs manquantes..."):
            len_before_nan_target_train = len(df)
            df_train = df.dropna(subset=[target])
            len_after_nan_target_train = len(df_train)
            len_diff_nan_target_train = len_before_nan_target_train - len_after_nan_target_train
            
            if target in df_test.columns:
                len_before_nan_target_test = len(df_test)
                df_test = df_test.dropna(subset=[target])
                len_after_nan_target_test = len(df_test)
                len_diff_nan_target_test = len_before_nan_target_test - len_after_nan_target_test
        
        corr_mat_train, corr_mat_test, corr_mat, prop_nan_train, prop_nan_test, prop_nan = correlation_missing_values(df_train, df_test)
        
        advance_progress(n_steps_total)       
        
        # Détecter les outliers
        with st.spinner("Détection et traitement des outliers..."):
            if wrang_outliers:
                df_train_outliers, df_test_outliers, nb_outliers = detect_and_winsorize(df_train, df_test, target = target, contamination = contamination)
            else:
                df_train_outliers, df_test_outliers, nb_outliers = df_train.copy(), df_test.copy(), "Aucun outlier traité."
        
        advance_progress(n_steps_total)
            
        # Imputer les valeurs manquantes
        with st.spinner("Imputation des données manquantes..."):
            df_train_imputed, df_test_imputed, scores_supervised, imputation_report = impute_missing_values(df_train_outliers, df_test_outliers, target=target, prop_nan=prop_nan, corr_mat=corr_mat)
        
        advance_progress(n_steps_total)
        
        # Suppression des variables redondantes
        with st.spinner("Suppression des variables redondantes..."):
            fig_cramer_cat, fig_cramer_num = False, False
            if drop_redundant:
                drop_cramer_cat, fig_cramer_cat = select_representative_categorial(df_train_imputed, target, threshold)
                drop_cramer_num, fig_cramer_num = select_representative_numerical(df_train_imputed, target, threshold)
                cramer_to_drop = drop_cramer_cat + drop_cramer_num
                
                df_train_imputed.drop(columns=cramer_to_drop, inplace=True, errors='ignore')
                if df_test_exists:
                    df_test_imputed.drop(columns=cramer_to_drop, inplace=True, errors='ignore')
        
        advance_progress(n_steps_total)
        
        # Appliquer l'encodage des variables (binaire, ordinal, nominal)
        with st.spinner("Encodage des variables catégorielles..."):
            if have_to_encode:
                df_train_encoded, df_test_encoded = encode_data(df_train_imputed, df_test_imputed, list_binary=list_binary, list_ordinal=list_ordinal, list_nominal=list_nominal, ordinal_mapping=ordinal_mapping)
            else:
                df_train_encoded, df_test_encoded = df_train_imputed.copy(), df_test_imputed.copy()
        
        advance_progress(n_steps_total)
    
        # Sélection des vraies variables numériques depuis df_train_imputed
        num_cols = df_train_imputed.select_dtypes(include=['number']).drop(columns=target).columns if not use_target else df_train_imputed.select_dtypes(include=['number']).columns

        # Mise à l'échelle
        with st.spinner("Mise à l'échelle des données..."):
            if scale_all_data:
                if scale_method:
                    scaler.fit(df_train_encoded[num_cols])

                    df_train_scaled = df_train_encoded.copy()
                    df_train_scaled[num_cols] = scaler.transform(df_train_encoded[num_cols])

                    df_test_scaled = df_test_encoded.copy()
                    df_test_scaled[num_cols] = scaler.transform(df_test_encoded[num_cols])
                else:
                    st.warning("⚠️ Veuillez sélectionner une méthode de mise à l'échelle.")
                    
        advance_progress(n_steps_total)
                
        # Appliquer les transformations individuelles
        with st.spinner("Transformation individuelles..."):
            if not scale_all_data:
                df_train_scaled, df_test_scaled = transform_data(df_train_imputed, df_test_imputed, list_boxcox=list_boxcox, list_yeo=list_yeo, list_log=list_log, list_sqrt=list_sqrt)

        advance_progress(n_steps_total)
        
        # Application de l'ACP en fonction du choix de l'utilisateur
        with st.spinner("Transformation factorielle des variables (ACP)..."):
            if use_pca:
                # Initialisation de l'ACP avec les paramètres choisis par l'utilisateur
                if pca_option == "Nombre de composantes":
                    n_components_valid = min(n_components, df_train_scaled.shape[1]-1)
                    pca = PCA(n_components=n_components_valid)
                elif pca_option == "Variance expliquée":
                    if explained_variance == 100:
                        pca = PCA(n_components=None)
                    else:
                        pca = PCA(n_components=explained_variance / 100)  # Conversion du % en proportion
                else:
                    pca = PCA()  # Par défaut, on prend tous les composants

                # Appliquer l'ACP sur les variables explicatives d'entrainement
                if not use_target:
                    df_explicatives_train = df_train_scaled.drop(columns=[target])
                else:
                    df_explicatives_train = df_train_scaled.copy()

                # Apprentissage de l'ACP sur l'ensemble d'entraînement
                pca.fit(df_explicatives_train)

                # Transformation des données d'entraînement
                df_pca_train = pca.transform(df_explicatives_train)
                
                # Créer le DataFrame avec les composantes principales pour l'entraînement
                df_pca_train = pd.DataFrame(df_pca_train, columns=[f'PC{i+1}' for i in range(df_pca_train.shape[1])], index=df_explicatives_train.index)

                # Ajouter le target si nécessaire pour l'entraînement
                if not use_target:
                    df_target_train = df_train_scaled[target]
                    df_train_scaled = pd.concat([df_pca_train, df_target_train], axis=1)
                else:
                    df_train_scaled = df_pca_train.copy()

                # Transformation des données de test avec le même modèle PCA
                if not use_target:
                    df_explicatives_test = df_test_scaled.drop(columns=[target])
                else:
                    df_explicatives_test = df_test_scaled.copy()

                # Transformation des données de test en utilisant l'ACP ajustée sur les données d'entraînement
                df_pca_test = pca.transform(df_explicatives_test)
                
                # Créer le DataFrame avec les composantes principales pour le test
                df_pca_test = pd.DataFrame(df_pca_test, columns=[f'PC{i+1}' for i in range(df_pca_test.shape[1])], index=df_explicatives_test.index)

                # Ajouter le target si nécessaire pour les données de test
                if not use_target:
                    df_target_test = df_test_scaled[target]
                    df_test_scaled = pd.concat([df_pca_test, df_target_test], axis=1)
                else:
                    df_test_scaled = df_pca_test.copy()

                # Calcul des inerties (variances expliquées par composante) sur l'ensemble d'entraînement
                pca_inertias = (pca.explained_variance_ratio_ * 100).tolist()
                pca_cumulative_inertias = [sum(pca_inertias[:i+1]) for i in range(len(pca_inertias))]

                # Création du DataFrame pour la variance expliquée et cumulative
                pca_infos = pd.DataFrame({'Variance expliquée': pca_inertias, 'Variance expliquée cumulée': pca_cumulative_inertias}).round(2)
                pca_infos = pca_infos.reset_index().rename(columns={'index': 'Nombre de composantes'})
                pca_infos['Nombre de composantes'] += 1

                # Visualisation avec Plotly (ou Seaborn si tu préfères)
                fig = px.line(pca_infos, x='Nombre de composantes', y=['Variance expliquée', 'Variance expliquée cumulée'],
                            markers=True, title="Evolution de la variance expliquée par les composantes principales",
                            labels={'value': 'Variance (%)', 'variable': 'Type de variance'},
                            color_discrete_map={'Variance expliquée': 'red', 'Variance expliquée cumulée': 'blue'})
                fig.update_layout(
                    xaxis_title='Nombre de composantes principales',
                    yaxis_title='Variance (%)',
                    legend_title='Type de variance',
                    width=900, height=600)
        
        advance_progress(n_steps_total)
            
        # Finir le traitement
        wrang_finished = True

        # Afficher le descriptif de la base de données
        st.write("### Descriptif de la base de données :")
        st.write("**Nombre d'observations (train) :**", df_train.shape[0])
        st.write("**Nombre de variables (train) :**", df_train.shape[1])
        st.write("**Nombre d'observations (test) :**", df_test.shape[0])
        st.write("**Nombre de variables (test) :**", df_test.shape[1])

        # Description des données
        if df_train is not None:
            description_train = []
            for col in df_train.columns:
                if pd.api.types.is_numeric_dtype(df_train[col]):
                    var_type = 'Numérique'
                    n_modalites = np.nan
                else:
                    var_type = 'Catégorielle'
                    n_modalites = df_train[col].nunique()

                description_train.append({
                    'Variable': col,
                    'Type': var_type,
                    'Nb modalités': n_modalites
                })
            st.dataframe(pd.DataFrame(description_train), use_container_width=True, hide_index=True)
        
            with st.expander("Diagnostic des données", expanded=False):
                st.write("**Matrice de corrélation entre les valeurs manquantes (train), en % :**")
                plt = heatmap_corr(corr_mat_train)
                st.pyplot(plt, use_container_width=True)

                st.write("**Matrice de corrélation entre les valeurs manquantes (test), en % :**")
                plt = heatmap_corr(corr_mat_test)
                st.pyplot(plt, use_container_width=True)

                st.write("**Proportion de valeurs manquantes par variable (train), en % :**")
                st.dataframe(prop_nan_train.sort_values(by='NaN proportion', ascending=False), use_container_width=True)

                st.write("**Proportion de valeurs manquantes par variable (test), en % :**")
                st.dataframe(prop_nan_test.sort_values(by='NaN proportion', ascending=False), use_container_width=True)

            with st.expander("Rapport du preprocessing", expanded=False):
                st.write("**Nombre de doublons traités :**", len_diff)
                st.write("**Nombre d'observations supprimées car la variable cible est manquante (train) :**", len_diff_nan_target_train)
                if df_test is not None and target in df_test.columns:
                    st.write("**Nombre d'observations supprimées car la variable cible est manquante (test) :**", len_diff_nan_target_test)
                st.write("**Nombre d'outliers traités :**", nb_outliers)

                st.write("**Résumé des méthodes d'imputation utilisées :**")
                st.dataframe(imputation_report, use_container_width=True, hide_index=True)

                if not scores_supervised.empty:
                    st.write("**Score de l'imputation supervisée :**")
                    st.dataframe(scores_supervised, use_container_width=True, hide_index=True)
                
                if 'cramer_to_drop' in locals():
                    st.write("**Variables redondantes supprimées :**")
                    df_vars_to_drop = pd.DataFrame(cramer_to_drop, columns=["Variables supprimées"])
                    st.dataframe(df_vars_to_drop, use_container_width=True, hide_index=True)
                
                if fig_cramer_cat and fig_cramer_cat is not None:
                    st.write("**Graphique des redondances catégorielles (Cramer's V):**")
                    st.pyplot(fig_cramer_cat, use_container_width=True)
                    if fig_cramer_cat is None:
                        st.info("Aucune redondance significative détectée entre les variables catégorielles selon le seuil spécifié.")

                if fig_cramer_num and fig_cramer_num is not None:
                    st.write("**Graphique des redondances numériques (correlations):**")
                    st.pyplot(fig_cramer_num, use_container_width=True)
                    if fig_cramer_num is None:
                        st.info("Aucune redondance significative détectée entre les variables numériques selon le seuil spécifié.")
    
        # Affichage du graphique PCA si nécessaire
        if use_pca:
            st.plotly_chart(fig)

        # Préparation pour le téléchargement
        if df_train is not None and df_test is not None and wrang_finished and not pb:
            # Créer un dossier temporaire pour stocker les fichiers CSV
            with io.BytesIO() as buffer:
                with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                    # Sauvegarder train et test dans des fichiers CSV dans le zip
                    with io.StringIO() as csv_buffer_train, io.StringIO() as csv_buffer_test:
                        df_train_scaled.to_csv(csv_buffer_train, index=False)
                        df_test_scaled.to_csv(csv_buffer_test, index=False)
                        
                        zip_file.writestr("train.csv", csv_buffer_train.getvalue())
                        zip_file.writestr("test.csv", csv_buffer_test.getvalue())
                
                # Préparer le téléchargement du dossier zip contenant les deux fichiers
                # st.write("### Aperçu des données traitées (entraînement) :")                
                # if df.shape[0] <= 10000 and df.shape[1] <= 30:
                #     st.dataframe(df_train_scaled, use_container_width=True, hide_index=True)
                # else:
                #     st.warning("Le DataFrame est trop volumineux pour être affiché.")

                # Télécharger le fichier zip contenant les deux bases
                st.download_button(
                    label="📥 Télécharger les données traitées",
                    data=buffer.getvalue(),
                    file_name="data_processed.zip",
                    mime="application/zip"
                )            
    
    else:
        # Suppression des colonnes inutiles
        progress_bar = st.progress(0)
        n_steps_total = 11
        current_step = 0
        with st.spinner("Suppression des colonnes inutiles..."):
            if drop_columns:
                df = df.drop(columns=drop_columns)
        advance_progress(n_steps_total)

        # Suppression des doublons
        with st.spinner("Suppression des doublons..."):
            if drop_dupli:
                len_before_dupli = len(df)
                df = df.drop_duplicates()
                len_after_dupli = len(df)
                len_diff = len_before_dupli - len_after_dupli
            else:
                len_diff = "Les doublons n'ont pas été traités."
        advance_progress(n_steps_total)

        # Étude des valeurs manquantes
        with st.spinner("Etude des valeurs manquantes..."):
            len_before_nan_target = len(df)
            df = df.dropna(subset=[target])
            len_after_nan_target = len(df)
            len_diff_nan_target = len_before_nan_target - len_after_nan_target

            corr_mat, _, _, prop_nan, _, _ = correlation_missing_values(df)
        advance_progress(n_steps_total)

        # Détection des outliers
        with st.spinner("Détection et traitement des outliers..."):
            if wrang_outliers:
                df_outliers, nb_outliers = detect_and_winsorize(df, target=target, contamination=contamination)
            else:
                df_outliers, nb_outliers = df.copy(), "Aucun outlier traité."
        advance_progress(n_steps_total)

        # Imputation des valeurs manquantes
        with st.spinner("Imputation des valeurs manquantes..."):
            df_imputed, _, scores_supervised, imputation_report = impute_missing_values(df_outliers, target=target, prop_nan=prop_nan, corr_mat=corr_mat)
        advance_progress(n_steps_total)

        # Suppression des variables redondantes
        with st.spinner("Suppression des variables redondantes..."):
            if drop_redundant:
                drop_cramer_cat, fig_cramer_cat = select_representative_categorial(df_imputed, target, threshold)
                drop_cramer_num, fig_cramer_num = select_representative_numerical(df_imputed, target, threshold)
                cramer_to_drop = drop_cramer_cat + drop_cramer_num
                df_imputed.drop(columns=cramer_to_drop, inplace=True, errors='ignore')
        advance_progress(n_steps_total)

        # Encodage
        with st.spinner("Encodage des variables catégorielles..."):
            if have_to_encode:
                df_encoded,_ = encode_data(df_imputed, list_binary=list_binary, list_ordinal=list_ordinal, list_nominal=list_nominal, ordinal_mapping=ordinal_mapping)
            else:
                df_encoded = df_outliers.copy()
        advance_progress(n_steps_total)

        # Mise à l’échelle
        with st.spinner("Mise à l'échelle des données..."):
            if scale_all_data:
                if scale_method:
                    num_cols = df_imputed.select_dtypes(include=['number']).drop(columns=target).columns if not use_target else df_imputed.select_dtypes(include=['number']).columns
                    df_scaled = df_encoded.copy()
                    scaler.fit(df_scaled[num_cols])
                    df_scaled[num_cols] = scaler.transform(df_scaled[num_cols])
                else:
                    st.warning("⚠️ Veuillez sélectionner une méthode de mise à l'échelle.")
        advance_progress(n_steps_total)

        # Transformations individuelles
        with st.spinner("Transformations individuelles..."):
            if not scale_all_data:
                df_scaled = transform_data(df_encoded, list_boxcox=list_boxcox, list_yeo=list_yeo, list_log=list_log, list_sqrt=list_sqrt)
        advance_progress(n_steps_total)
        
        with st.spinner("Transformation factorielle des variables (ACP)..."):
            if use_pca:
                # Initialisation de l'ACP avec les paramètres choisis par l'utilisateur
                if pca_option == "Nombre de composantes":
                    n_components = min(n_components, df_scaled.shape[1])
                    pca = PCA(n_components=n_components)
                
                elif pca_option == "Variance expliquée":
                    if explained_variance == 100:
                        pca = PCA(n_components=None)
                    else:
                        pca = PCA(n_components=explained_variance / 100)  # Conversion du % en proportion
                else:
                    pca = PCA()  # Par défaut, on prend tous les composants

                # Appliquer l'ACP sur les variables explicatives
                if not use_target:
                    df_explicatives = df_scaled.drop(columns=[target])
                else:
                    df_explicatives = df_scaled.copy()

                # Apprentissage de l'ACP
                pca.fit(df_explicatives)

                # Transformation des données
                df_pca = pca.transform(df_explicatives)
                
                # Créer le DataFrame avec les composantes principales
                df_pca = pd.DataFrame(df_pca, columns=[f'PC{i+1}' for i in range(df_pca.shape[1])], index=df_explicatives.index)

                # Ajouter le target si nécessaire
                if not use_target:
                    df_target = df_scaled[target]
                    df_scaled = pd.concat([df_pca, df_target], axis=1)
                else:
                    df_scaled = df_pca.copy()

                # Calcul des inerties (variances expliquées par composante)
                pca_inertias = (pca.explained_variance_ratio_ * 100).tolist()
                pca_cumulative_inertias = [sum(pca_inertias[:i+1]) for i in range(len(pca_inertias))]

                # Création du DataFrame pour la variance expliquée et cumulative
                pca_infos = pd.DataFrame({'Variance expliquée': pca_inertias, 'Variance expliquée cumulée': pca_cumulative_inertias}).round(2)
                pca_infos = pca_infos.reset_index().rename(columns={'index': 'Nombre de composantes'})
                pca_infos['Nombre de composantes'] += 1

                # Visualisation avec Plotly (ou Seaborn si tu préfères)
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
                st.plotly_chart(fig)
        
        advance_progress(n_steps_total)
                
        # Finir le traitement
        wrang_finished = True
        # Afficher le descriptif de la base de données
        st.write("### Descriptif de la base de données :")
        st.write("**Nombre d'observations :**", df.shape[0])
        st.write("**Nombre de variables :**", df.shape[1])
            
        if df is not None:
            description = []
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    var_type = 'Numérique'
                    n_modalites = np.nan
                else:
                    var_type = 'Catégorielle'
                    n_modalites = df[col].nunique()
                
                description.append({
                    'Variable': col,
                    'Type': var_type,
                    'Nb modalités': n_modalites
                })
            st.dataframe(pd.DataFrame(description), use_container_width=True, hide_index=True)
            
            with st.expander("Diagnostic des données", expanded=False):
                st.write("**Matrice de corrélation entre les valeurs manquantes (en %) :**")
                plt = heatmap_corr(corr_mat)
                st.pyplot(plt, use_container_width=True)
                                
                st.write("**Proportion de valeurs manquantes par variable (en %) :**")
                st.dataframe(prop_nan.sort_values(by='NaN Proportion', ascending=False), use_container_width=True)

            with st.expander("Rapport du preprocessing", expanded=False):
                st.write("**Nombre de doublons traités :**", len_diff)
                st.write("**Nombre d'observations supprimées car la variable cible est manquante :**", len_diff_nan_target)
                st.write("**Nombre d'outliers traités :**", nb_outliers)

                st.write("**Résumé des méthodes d'imputation utilisées :**")
                st.dataframe(imputation_report, use_container_width=True, hide_index=True)

                if not scores_supervised.empty:
                    st.write("**Score de l'imputation supervisée :**")
                    st.dataframe(scores_supervised, use_container_width=True, hide_index=True)
                
                if 'cramer_to_drop' in locals():
                    st.write("**Variables redondantes supprimées :**")
                    df_vars_to_drop = pd.DataFrame(cramer_to_drop, columns=["Variables supprimées"])
                    st.dataframe(df_vars_to_drop, use_container_width=True, hide_index=True)
                
                if fig_cramer_cat:
                    st.write("**Graphique des redondances catégorielles (Cramer's V):**")
                    st.pyplot(fig_cramer_cat, use_container_width=True)

                if fig_cramer_num:
                    st.write("**Graphique des redondances numériques (correlations):**")
                    st.pyplot(fig_cramer_num, use_container_width=True)
        
            if use_pca:
                st.plotly_chart(fig)
        
        # Téléchargement du fichier encodé
        if df is not None and wrang_finished and not pb:
            df = df_scaled.copy()
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
        # Afficher l'aperçu des données traitées seulement si raisonnable
        # st.write("### Aperçu des données traitées :")
        # if df.shape[0] <= 10000 and df.shape[1] <= 30:
        #     st.dataframe(df, use_container_width=True, hide_index=True)
        # else:
        #     st.warning("Le DataFrame est trop volumineux pour être affiché.")

            # Afficher le bouton pour télécharger le fichier
            st.download_button(
                label="📥 Télécharger les données traitées",
                data=csv_data,
                file_name="data.csv",
                mime="text/csv"
            )
        
if valid_mod:
    # Effectuer la modélisation

    # Division des données
    if df_test is None:
        X = df.drop(columns=target)
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True)
    else:
        X_train = df.drop(columns=target)
        y_train = df[target]
        X_test = df_test.drop(columns=target)
        y_test = df_test[target]

    if use_loocv:
        cv = X_train.shape[0]
    # Choisir le meilleur modèle
    results = []
    for model in models:  
        # Déterminer chaque modèle à optimiser
        if model in ["Linear Regression", "Logistic Regression", "KNN"]:
            n_trial = 50
        else:
            n_trial = 25
        repeats = n_trial
        
        best_model, best_params, best_value = optimize_model(model_choosen=model, task=task,
                                                            X_train=X_train, y_train=y_train,
                                                            cv=cv, scoring=scoring_comp,
                                                            multi_class=multi_class,
                                                            n_trials=n_trial, n_jobs=-1)
        
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
    
    # Evaluer les meilleurs modèles
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
    inv_metrics = {v: k for k, v in metrics_regression.items()} if task == "Regression" else {v: k for k, v in metrics_classification.items()}

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
    
    # Afficher les coefficients des modèles linéaires
    for idx, best_model in df_score['Best Model'].items():
        model = instance_model(idx, df_train2, task)

        if task == 'Regression' and isinstance(model, (LinearRegression, ElasticNet, Ridge, Lasso)):
            model.fit(X_train, y_train)
            coefs = model.coef_
            intercept = model.intercept_
            df_coefs = pd.DataFrame({
                'Variable': X_train.columns,
                'Coefficient': coefs
            }).sort_values(by='Coefficient', key=abs, ascending=False)
            df_coefs['Coefficient'] = df_coefs['Coefficient'].round(5)
            
            # Ajout de la constante
            df_intercept = pd.DataFrame({
                'Variable': ['Intercept'],
                'Coefficient': [intercept if np.isscalar(intercept) else intercept[0]]
            })
            df_coefs = pd.concat([df_intercept, df_coefs], ignore_index=True)
            
            # Affichage
            st.subheader(f"Coefficients – Régression Linéaire")
            st.dataframe(df_coefs, use_container_width=True, hide_index=True)

        elif task == 'Classification' and isinstance(model, LogisticRegression):
            model.fit(X_train, y_train)
            intercept = model.intercept_

            # Si régression logistique multinomiale
            if model.coef_.ndim > 1:
                # Ici, tu veux afficher les coefficients pour chaque classe
                for i, coefs in enumerate(model.coef_):
                    df_coefs = pd.DataFrame({
                        'Variable': X_train.columns,
                        'Coefficient': coefs
                    }).sort_values(by='Coefficient', key=abs, ascending=False)
                    df_coefs['Coefficient'] = df_coefs['Coefficient'].round(5)
                    
                    # Ajout de la constante pour chaque classe
                    df_intercept = pd.DataFrame({
                        'Variable': ['Intercept'],
                        'Coefficient': [intercept[i] if np.isscalar(intercept) else intercept[i]]
                    })
                    df_coefs = pd.concat([df_intercept, df_coefs], ignore_index=True)

                    # Affichage des coefficients pour chaque classe
                    st.subheader(f"Coefficients – Régression Logistique (Classe {i})")
                    st.dataframe(df_coefs, use_container_width=True, hide_index=True)
            else:
                coefs = model.coef_[0]
                df_coefs = pd.DataFrame({
                    'Variable': X_train.columns,
                    'Coefficient': coefs
                }).sort_values(by='Coefficient', key=abs, ascending=False)
                df_coefs['Coefficient'] = df_coefs['Coefficient'].round(3)

                # Ajout de l'intercept
                df_intercept = pd.DataFrame({
                    'Variable': ['Intercept'],
                    'Coefficient': [intercept if np.isscalar(intercept) else intercept[0]]
                })
                df_coefs = pd.concat([df_intercept, df_coefs], ignore_index=True)

                # Affichage
                st.subheader(f"Coefficients – Régression Logistique")
                st.dataframe(df_coefs, use_container_width=True, hide_index=True)
                 
    
    # Calculer les odds-ratios pour la régression logistique
    for idx, best_model in df_score['Best Model'].items():
        model = instance_model(idx, df_train2, task)
        if isinstance(model, LogisticRegression) and task == 'Classification':
            model.fit(X_train, y_train)
            odds_ratios = np.exp(model.coef_[0])
            df_odds = pd.DataFrame({
                'Variable': X_train.columns,
                'Odds Ratio': odds_ratios
            }).sort_values(by='Odds Ratio', ascending=False)
            df_odds['Odds Ratio'] = df_odds['Odds Ratio'].round(2)
            st.dataframe(df_odds)
    
    # Afficher SHAPE et LIME
    st.subheader("Interprétation globale ou locale des modèles")       
    for idx, best_model in df_score['Best Model'].items():
        model = instance_model(idx, df_train2, task)
        model.fit(X_train, y_train)

        try:
            # SHAP - modèles linéaires
            if isinstance(model, (LinearRegression, LogisticRegression, ElasticNet, Ridge, Lasso)):
                explainer = shap.LinearExplainer(model, X_train)
                shap_values = explainer(X_train)
                plt.clf()
                shap.summary_plot(shap_values, X_train, show=False)
                fig = plt.gcf()
                st.pyplot(fig)

            # LIME - uniquement pour KNN
            elif isinstance(model, (KNeighborsClassifier, KNeighborsRegressor)):
                mode = "classification" if task == "Classification" else "regression"
                lime_explainer = LimeTabularExplainer(X_train.values, mode=mode, feature_names=X_train.columns)
                explanation = lime_explainer.explain_instance(X_train.iloc[0].values, model.predict)
                html = explanation.as_html()
                html = html.replace("<body>", '<body style="background-color:white; color:black;">')
                components.html(html, height=375, scrolling=True)

            # SHAP - arbres (RandomForest, XGBoost, LightGBM)
            else:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_train)
                plt.clf()
                shap.summary_plot(shap_values, X_train, show=False)
                fig = plt.gcf()
                st.pyplot(fig)

        except Exception as e:
            st.warning(f"Erreur pour {idx} : {e}")    
    
    
    # Appliquer le modèle : calcul-biais-variance    
    bias_variance_results = []
    for idx, best_model in df_score['Best Model'].items():
        model = instance_model(idx, df_train2, task)
        expected_loss, bias, var, bias_relative, var_relative = bias_variance_decomp(
            model, task=task,
            X=X_train.values, y=y_train.values,
            cv=cv)

        # Création d'un dictionnaire pour stocker les résultats
        result = {
            "Bias": round(bias, 3),  # Biais moyen, arrondi à 3 décimales
            "Variance": round(var, 3),  # Variance moyenne, arrondi à 3 décimales
            "Bias relatif": round(bias_relative, 3),  # Biais relatif, arrondi à 3 décimales
            "Variance relatif": round(var_relative, 3),  # Variance relative, arrondi à 3 décimales
        }
        
        # Logique de conclusion en fonction du biais relatif et de la variance relative
        if abs(bias_relative) > 0.2 and var_relative > 0.2:
            result["Conclusion"] = "Problème majeur : le modèle est vraisembbalement pas adapté"
        elif abs(bias_relative) > 0.2:
            result["Conclusion"] = "Biais élevé : suspicion de sous-ajustement"
        elif var_relative > 0.2:
            result["Conclusion"] = "Variance élevée : suspicion de sur-ajustement"
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
    for idx, mdl in df_score['Best Model'].items():
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
        plt.title(f"Importance des variables par permutation - {idx}", fontsize=8)
        plt.gca().invert_yaxis()
        st.pyplot(plt)
        plt.close()
            
    # Courbes d'apprentissage
    st.subheader(f"Courbes d'apprentissage")
    
    for idx, mdl in df_score['Best Model'].items(): 
        model = instance_model(idx, df_train2, task)       
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=cv, scoring=scoring_comp,
            train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
        )

        train_scores_mean = np.mean(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)

        plt.figure(figsize=(5, 3))
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Score entraînement")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Score validation")
        plt.title(f"Learning Curve - {idx}", fontsize=8)
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