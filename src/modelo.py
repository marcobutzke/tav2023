import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.neighbors import NearestNeighbors
from pyod.models.knn import KNN

from funcoes import zscore, rfm_variables, fit_data
data = pd.read_feather('dados/gs.feather')
variaveis = [
    'f_vendas', 'f_lucro', 'm_entrega', 'm_lucro', 'm_qtde',
    'm_vendas', 'r_dias'
]


def deep_learning_dnn(df_dl, dep_var, classes):
    # Separa a variável dependente das demais
    deep_feat = df_dl.drop(columns=[dep_var], axis=1)
    deep_label = df_dl[dep_var]
    # Verifica os tipos das variáveis
        # Verifica as colunas para normalização - as demais serão discretizadas - Função Bucketize do Tensor Flow
    categorical_columns = [col for col in deep_feat.columns if len(deep_feat[col].unique()) == 2 or deep_feat[col].dtype == 'O']
    continuous_columns = [col for col in deep_feat.columns if len(deep_feat[col].unique()) > 2 and (deep_feat[col].dtype == 'int64' or deep_feat[col].dtype == 'float64')]    
    cols_to_scale = continuous_columns[:]
    #cols_to_scale.remove('meses')
    # Ajusta as bases de treino e de teste
    XX_T = df_dl.drop(columns=[dep_var], axis=1)
    XX_t = df_dl.drop(columns=[dep_var], axis=1)
    yy_T = df_dl[dep_var]
    yy_t = df_dl[dep_var]
    # Normaliza as variáveis nas bases de treino e teste
    scaler = StandardScaler()
    XX_T.loc[:, cols_to_scale] = scaler.fit_transform(XX_T.loc[:, cols_to_scale])
    XX_t.loc[:, cols_to_scale] = scaler.fit_transform(XX_t.loc[:, cols_to_scale])
    # Ajustes das Variáveis Categórica - Não presentes neste modelo
    categorical_object_feat_cols = [tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_hash_bucket(key=col, hash_bucket_size=1000), dimension=len(df_dl[col].unique()))
    for col in categorical_columns if df_dl[col].dtype == 'O']
    # Ajustes das Variáveis Categórica - Não presentes neste modelo
    categorical_integer_feat_cols = [tf.feature_column.embedding_column(
    tf.feature_column.categorical_column_with_identity(key=col, num_buckets=2), dimension=len(df_dl[col].unique()))
    for col in categorical_columns if df[col].dtype=='int64']
    continuous_feat_cols = [tf.feature_column.numeric_column(key=col) for col in continuous_columns]
    feat_cols = categorical_object_feat_cols + \
                categorical_integer_feat_cols + \
                continuous_feat_cols
    # Rotina de DNN (Deep Neural Network)
    input_fun = tf.compat.v1.estimator.inputs.pandas_input_fn(XX_T, yy_T, batch_size=50, num_epochs=1000, shuffle=True)
    pred_input_fun = tf.compat.v1.estimator.inputs.pandas_input_fn(XX_t, batch_size=50, shuffle=False)
    DNN_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=classes)
    DNN_model.train(input_fn=input_fun, steps=5000)
    # Resgata os resultados da DNN
    predictions = DNN_model.predict(pred_input_fun)
    pred = list(predictions)
    return pred


def outliers_detection(orig, vars):
    sc_x = StandardScaler()
    datax = sc_x.fit_transform(vars)
    clf = KNN().fit(vars)
    outliers = clf.predict(vars)
    dataod = orig.copy()
    dataod['outlier'] = outliers
    return dataod