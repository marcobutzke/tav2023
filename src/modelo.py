import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pyod.models.knn import KNN
from sklearn.linear_model import LogisticRegression

from funcoes import zscore, rfm_variables, fit_data, outliers_detection
data = pd.read_feather('../dados/gs.feather')
variaveis = [
    'f_vendas', 'f_lucro', 'm_entrega', 'm_lucro', 'm_qtde',
    'm_vendas', 'r_dias'
]

print('Cálculo da probabilidade de lucro por país...')
gr_pais = data.groupby('Country')[['Sales','Quantity','Profit', 'Shipping Cost']].mean().copy()
gr_pais['Lucro'] = gr_pais['Profit'].apply(lambda x : 0 if x < 0 else 1)
X_Train = gr_pais.drop(columns=['Lucro'], axis=1)
X_Test = gr_pais.drop(columns=['Lucro'], axis=1)
y_Train = gr_pais['Lucro']
y_Test = gr_pais['Lucro']
sc_x = StandardScaler()
X_Train = sc_x.fit_transform(X_Train)
X_Test = sc_x.fit_transform(X_Test)
logreg = LogisticRegression(solver="lbfgs", max_iter=500)
logreg.fit(X_Train, y_Train)
pred_logreg = logreg.predict(X_Test)
pred_proba = logreg.predict_proba(X_Test)
gr_pais['previsao'] = pred_logreg
lista_proba = pred_proba.tolist()
lista_proba = pd.DataFrame(
    lista_proba, columns = ['prob_prejuizo', 'prob_lucro']
)
gr_pais = gr_pais.reset_index()
gr_pais = pd.merge(gr_pais, lista_proba, left_index=True, right_index=True)
gr_pais.to_feather('../dados/probabilidade_pais.feather')

print('Cálculo da associação por país')
original = fit_data(data, 'Country')
original = original.fillna(0)
base = original[variaveis]
vizinhos = NearestNeighbors(n_neighbors=min(4, len(base))).fit(base)
similares = []
for index, row in original.iterrows():
    # print('Referencia: {0}'.format(row['referencia']))
    original_referencia = original[
        original['referencia'] == row['referencia']][variaveis]
    similar = vizinhos.kneighbors(original_referencia, return_distance=False)[0]
    original_similar = original.iloc[similar][variaveis].reset_index()
    referencia = original.iloc[similar]['referencia'].reset_index()
    referencia = referencia.merge(original_similar, on='index', how='left')
    referencia = referencia.drop(columns=['index'])
    for ind, rw in referencia.iterrows():    
        if row['referencia'] != rw['referencia']:            
            similares.insert(0, [row['referencia'], rw['referencia']])
similares = pd.DataFrame(
    similares,
    columns = ['referencia', 'vizinho']
)            
similares.to_feather('../dados/knn_pais.feather')

print('Classificação do Consumidor...')
gr_con = data.groupby(
    [
        'Customer ID',
        'Country',
        'City',
        'Market',
        'Region'
    ]
)[
    [
        'Sales',
        'Quantity',
        'Profit',
        'Shipping Cost'
    ]
].mean().reset_index()
for col in ['Sales', 'Quantity', 'Profit', 'Shipping Cost']:
    gr_con = zscore(gr_con, 'Customer ID', col, 'z'+col)
gr_con['score'] = gr_con['zSales'] \
                + gr_con['zQuantity'] \
                + gr_con['zProfit'] \
                + gr_con['zShipping Cost']
media_score = gr_con['score'].mean()
dpadr_score = gr_con['score'].std()
gr_con['classe'] = gr_con['score'].apply(lambda x : int((x - media_score) / dpadr_score) + 3)
gr_con['classe'] = gr_con['classe'].apply(lambda x : 0 if x < 0 else x)
gr_con['classe'] = gr_con['classe'].apply(lambda x : 6 if x > 6 else x)
gr_con['rank'] = gr_con['score'].rank(ascending=False)
gr_con['lucro'] = gr_con['Profit'].apply(lambda x : 0 if x < 0 else 1)
gr_con.to_feather('../dados/classificacao_consumidor.feather')

print('Clusterização por país...')
country_rfm = fit_data(data, 'Country')
country_rfm = country_rfm.fillna(0)
country_rfm['cluster'] = KMeans(
    n_clusters=3,
    random_state=0,
    n_init='auto'
).fit(
    country_rfm[
        variaveis
    ]
).labels_
cluster = []
for index, row in enumerate(KMeans(n_clusters=5, random_state=0, n_init='auto'
    ).fit(country_rfm[variaveis]).cluster_centers_):
    cluster.insert(0, 
        [index, row[0], row[1], row[2], row[3], row[4], row[5], row[6]]
    )
cluster = pd.DataFrame(
    cluster,
    columns = [
        'cluster', 'clf_vendas', 'cls_lucro', 'clm_entrega',
        'clm_lucro', 'clm_qtde', 'clm_vendas', 'clr_dias'
    ]
)
country_rfm = country_rfm.merge(
    cluster,
    on='cluster',
    how='left'
)
country_rfm.to_feather('../dados/clusterizacao_pais.feather')

print('Regressão (com sazonalidades) por Mercado e Regiao...')
regressao_market_region = pd.DataFrame()
primeiro = 0
for index, row in data[['Market', 'Region']].drop_duplicates().iterrows():
    regressao = data[
        (data['Market'] == row['Market']) &
        (data['Region'] == row['Region'])
    ][['Order Date Month', 'Sales']
    ].groupby('Order Date Month')['Sales'].sum().reset_index()
    regressao = regressao.rename(columns={
        'Order Date Month': 'ds', 'Sales': 'y'
    })
    m = Prophet().fit(regressao)
    future = m.make_future_dataframe(periods=12, freq='MS')
    forecast = m.predict(future)
    forecast['Market'] = row['Market']
    forecast['Region'] = row['Region']
    forecast = forecast.merge(regressao, on='ds', how='left')
    forecast = forecast[
        ['Market', 'Region', 'ds', 'yhat', 'y', 'yhat_lower', 'yhat_upper']
    ].copy()
    if primeiro == 0:
        regressao_market_region = forecast
        primeiro = 1
    else:
        regressao_market_region = pd.concat(
            [
                regressao_market_region,
                forecast
            ]
        )       
regressao_market_region = regressao_market_region.reset_index()         
regressao_market_region.to_feather('../dados/regressao_mercado_regiao.feather')

print('Detecção de Anomalias por País...')
df_out = fit_data(data, 'Country')
out = df_out[variaveis].fillna(0).copy()
outliers = outliers_detection(df_out, out)
outliers.to_feather('../dados/outliers_pais.feather')

print('Cálculo da associação por subcategoria')
original = fit_data(data, 'Sub-Category')
original = original.fillna(0)
base = original[variaveis]
vizinhos = NearestNeighbors(n_neighbors=min(4, len(base))).fit(base)
similares = []
for index, row in original.iterrows():
    # print('Referencia: {0}'.format(row['referencia']))
    original_referencia = original[
        original['referencia'] == row['referencia']][variaveis]
    similar = vizinhos.kneighbors(original_referencia, return_distance=False)[0]
    original_similar = original.iloc[similar][variaveis].reset_index()
    referencia = original.iloc[similar]['referencia'].reset_index()
    referencia = referencia.merge(original_similar, on='index', how='left')
    referencia = referencia.drop(columns=['index'])
    for ind, rw in referencia.iterrows():    
        if row['referencia'] != rw['referencia']:            
            similares.insert(0, [row['referencia'], rw['referencia']])
similares = pd.DataFrame(
    similares,
    columns = ['referencia', 'vizinho']
)            
similares.to_feather('../dados/knn_subcategoria.feather')

print('Cálculo da associação por produto')
original = fit_data(data, 'Product Name')
original = original.fillna(0)
base = original[variaveis]
vizinhos = NearestNeighbors(n_neighbors=min(4, len(base))).fit(base)
similares = []
for index, row in original.iterrows():
    # print('Referencia: {0}'.format(row['referencia']))
    original_referencia = original[
        original['referencia'] == row['referencia']][variaveis]
    similar = vizinhos.kneighbors(original_referencia, return_distance=False)[0]
    original_similar = original.iloc[similar][variaveis].reset_index()
    referencia = original.iloc[similar]['referencia'].reset_index()
    referencia = referencia.merge(original_similar, on='index', how='left')
    referencia = referencia.drop(columns=['index'])
    for ind, rw in referencia.iterrows():    
        if row['referencia'] != rw['referencia']:            
            similares.insert(0, [row['referencia'], rw['referencia']])
similares = pd.DataFrame(
    similares,
    columns = ['referencia', 'vizinho']
)            
similares.to_feather('../dados/knn_produto.feather')

print('Concluído!')