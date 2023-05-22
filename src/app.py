import pandas as pd
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from plotly import tools
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.express as px
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster

st.set_page_config(layout="wide")
st.title('App - Tópicos Avançados')

@st.cache_data
def load_database():
    return pd.read_feather('../dados/gs.feather'), \
        pd.read_feather('../dados/knn_pais.feather'), \
        pd.read_feather('../dados/probabilidade_pais.feather'), \
        pd.read_feather('../dados/classificacao_consumidor.feather'), \
        pd.read_feather('../dados/localizacao.feather')

gs, knn_pais, prb_pai, cla_con, coordenadas = load_database() 

taberp, tabbi, tabstore = st.tabs(['Sistema Interno', 'Gestão', 'E-Commerce'])   
with taberp:
    st.header('Dados do Sistema Interno')
    consumidor = st.selectbox(
        'Selecione o consumidor',
        gs['Customer ID'].unique()
    )
    gs_con = gs[gs['Customer ID'] == consumidor]
    cla_con_con = cla_con[cla_con['Customer ID'] == consumidor].reset_index()
    st.dataframe(gs_con[['Customer Name', 'Segment']].drop_duplicates())
    with st.expander('Paises similares'):
        st.write(gs_con['Country'].values[0])
        st.dataframe(knn_pais[knn_pais['referencia'] == gs_con['Country'].values[0]])
        st.write('Probabilidade:')
        st.dataframe(prb_pai[prb_pai['Country'] == gs_con['Country'].values[0]])
    cl1, cl2, cl3, cl4 = st.columns(4)
    cl1.metric('Score', round(cla_con_con['score'][0],4))
    cl2.metric('Classe', round(cla_con_con['classe'][0],4))
    cl3.metric('Rank', round(cla_con_con['rank'][0],4))
    cl4.metric('Lucro', round(cla_con_con['lucro'][0],4))
    cl1.metric('Valor Total Comprado', round(gs_con['Sales'].sum(),2))
    cl2.metric('Valor Lucro', round(gs_con['Profit'].sum(),2))
    cl3.metric('Valor Médio Comprado', round(gs_con['Sales'].mean(),2))
    cl4.metric('Quantidade Comprada', round(gs_con['Quantity'].sum(),2))

     