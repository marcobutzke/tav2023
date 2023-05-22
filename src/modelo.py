import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from prophet import Prophet
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pyod.models.knn import KNN

from funcoes import zscore, rfm_variables, fit_data
data = pd.read_feather('../dados/gs.feather')
variaveis = [
    'f_vendas', 'f_lucro', 'm_entrega', 'm_lucro', 'm_qtde',
    'm_vendas', 'r_dias'
]


def outliers_detection(orig, vars):
    sc_x = StandardScaler()
    datax = sc_x.fit_transform(vars)
    clf = KNN().fit(vars)
    outliers = clf.predict(vars)
    dataod = orig.copy()
    dataod['outlier'] = outliers
    return dataod


print('Cálculo da probabilidade de lucro por país...')
# dimensao = 'Country'
# medidas = ['Sales', 'Quantity', 'Profit']
# grupo = data.groupby(dimensao)[medidas].mean().reset_index()
# grupo['Benefit'] = grupo['Profit'].apply(lambda x : 0 if x < 0 else 1)
# grupo = grupo.set_index(dimensao)
# probabilidade = deep_learning_dnn(grupo, 'Benefit', 2)
# probabilidade_classe = []
# for i in range(len(probabilidade)):
#     probabilidade_classe.append(probabilidade[i]["class_ids"][0])
# probabilidade_prob0 = []
# for i in range(len(probabilidade)):
#     probabilidade_prob0.append(probabilidade[i]["probabilities"][0])
# probabilidade_prob1 = []
# for i in range(len(probabilidade)):
#     probabilidade_prob1.append(probabilidade[i]["probabilities"][1])
# grupo['dl_classe'] = probabilidade_classe
# grupo['lucro_0'] = probabilidade_prob0
# grupo['lucro_1'] = probabilidade_prob1
# grupo.to_feather('../dados/probabilidade_pais.feather')   

# print('Cálculo da associação por país')
# original = fit_data(data, 'Country')
# original = original.fillna(0)
# base = original[variaveis]
# vizinhos = NearestNeighbors(n_neighbors=min(4, len(base))).fit(base)
# similares = []
# for index, row in original.iterrows():
#     print('Referencia: {0}'.format(row['referencia']))
#     original_referencia = original[
#         original['referencia'] == row['referencia']][variaveis]
#     similar = vizinhos.kneighbors(original_referencia, return_distance=False)[0]
#     original_similar = original.iloc[similar][variaveis].reset_index()
#     referencia = original.iloc[similar]['referencia'].reset_index()
#     referencia = referencia.merge(original_similar, on='index', how='left')
#     referencia = referencia.drop(columns=['index'])
#     for ind, rw in referencia.iterrows():    
#         if row['referencia'] != rw['referencia']:            
#             similares.insert(0, [row['referencia'], rw['referencia']])
# similares = pd.DataFrame(
#     similares,
#     columns = ['referencia', 'vizinho']
# )            
# similares.to_feather('../dados/knn_pais.feather')

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
for col in gr_con.columns:
    if col != 'Customer ID':
        if gr_con[col].dtype == 'int64':
            gr_con = zscore(gr_con, col, 'Profit', 'z'+col)
        else:
            gr_con = zscore(gr_con, 'Customer ID', col, 'z'+col)
gr_con['score'] = gr_con['zMarket'] \
                + gr_con['zRegion'] \
                + gr_con['zCountry'] \
                + gr_con['zCity'] \
                + gr_con['zSales'] \
                + gr_con['zQuantity'] \
                + gr_con['zProfit'] \
                + gr_con['zShipping Cost']
media_score = gr_con['score'].mean()
dpadr_score = gr_con['score'].std()
gr_con['classe'] = gr_con['score'].apply(lambda x : int((x - media_score) / dpadr_score) + 3)
gr_con['classe'] = gr_con['classe'].apply(lambda x : 0 if x < 0 else x)
gr_con['classe'] = gr_con['classe'].apply(lambda x : 6 if x > 6 else x)
gr_con['rank'] = gr_con['score'].rank(ascending=False)
gr_con['lucro'] = gr_con['valor_lucro'].apply(lambda x : 0 if x < 0 else 1)
gr_con.to_feather('../dados/classificacao_consumidor.feather')

print('Concluído!')