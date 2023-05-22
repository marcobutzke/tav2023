import pandas as pd
import numpy as np
import datetime as dt

print('Leitura dos dados originais...')
data = pd.read_excel('origem/GS.xlsx')

print('--> Preparação dos Dados...')

print('Eliminando colunas...')
data = data.drop(columns=['Row ID', 'Postal Code'])

print('Ajuste de datas...')
data['Year'] = data['Order Date'].dt.year
data['Month'] = data['Order Date'].dt.month
data['Period'] = ((data['Year'] - data['Year'].min()) * 12) + data['Month']
data['Order Date Month'] = data['Order Date'].apply(lambda x : x.strftime("%Y-%m-01"))

print('Novas medidas... Delivery e Price')
data['Delivery'] = (data['Ship Date'] - data['Order Date']).dt.days
data['Price'] = round((data['Sales'] / data['Quantity']),2)

print('Variável Dependente: Benefit...')
data['Benefit'] = data['Profit'].apply(lambda x : 1 if x > 0 else 0)

print('--> Gravando dados preparados...')
data.to_feather('dados/gs.feather')

print('Coordenadas...')
print('Leitura dos países')
country = pd.read_json(
    'origem/countries.json', 
    orient='index'
).reset_index()
country = country.drop(columns=['native'])
print('Leitura dos continentes...')
continent = pd.read_json(
    'origem/continents.json', 
    orient='index'
).reset_index()
print('Leitura das Cidades...')
cities = pd.read_json('origem/cities.json')
grupo = cities.groupby('country')[
    [
        'lat', 
        'lng'
    ]
].mean().reset_index().copy()
grupo = grupo.rename(
    columns={
        'lat': 'country_lat',
        'lng': 'country_lng'
    }
)
country = country.merge(
    grupo, 
    left_on='index', 
    right_on='country', 
    how='left'
)
capital = country.merge(
    cities, 
    left_on=[
        'index',
        'capital'
    ], 
    right_on=[
        'country',
        'name'
    ], 
    how='left'
)
capital = capital.drop(
    columns=[
        'country_x',
        'country_y',
        'name_y'
    ]
)
capital = capital.rename(
    columns={
        'index': 'pais_sigla',
        'name_x': 'country',
        'lat': 'capital_lat',
        'lng': 'capital_lng'
    }
)
cities = cities.merge(
    capital, 
    left_on='country', 
    right_on='pais_sigla', 
    how='left'
)
print('Localizando Cidades...')
localizacao = []
pais = ''
for index, row in data[
    [
        'City',
        'Country'
    ]
].drop_duplicates().sort_values(by=['Country','City']).iterrows():
    if row['Country'] != pais:
        print(row['Country'])
        pais = row['Country']
    if len(cities[
        (cities['name'] == row['City']) & (cities['country_y'] == row['Country'])
    ]) > 0:
        cidade = cities[
            (cities['name'] == row['City']) & (cities['country_y'] == row['Country'])
        ][['name', 'country_y', 'lat', 'lng']].reset_index()
        localizacao.insert(0, [
            cidade['name'].values[0],
            cidade['country_y'].values[0],
            cidade['lat'].values[0],
            cidade['lng'].values[0]
        ])
    else:
        if len(capital[capital['country'] == row['Country']]) > 0:
            cidade = capital[capital['country'] == row['Country']][
                ['capital', 'country', 'capital_lat', 'capital_lng']].reset_index()
            localizacao.insert(0, [
                cidade['capital'].values[0],
                cidade['country'].values[0],
                cidade['capital_lat'].values[0],
                cidade['capital_lng'].values[0]
            ])
localizacao = pd.DataFrame(
    localizacao,
    columns = ['cidade', 'pais', 'lat', 'lng']
)
localizacao = localizacao.fillna(0)
# localizacao = localizacao.drop(columns=['index'])
print('Salvando localização das cidades...')
localizacao.to_feather('dados/localizacao.feather')
print('Concluido')
