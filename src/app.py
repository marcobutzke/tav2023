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
        pd.read_feather('../dados/localizacao.feather')

gs, gs_con, coordenadas = load_database() 

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
        st.write(gs_con['Country'][0])
        st.dataframe(knn_pais[knn_pais['referencia'] == gs_con['Country'][0]] )
        # st.write('Probabilidade:')
        # st.dataframe(prb_pai[prb_pai['Country'] == gs_con['Country'][0]])

     