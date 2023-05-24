import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyod.models.knn import KNN

def zscore(df, grupo, var, score):
    media = df[var].mean()
    dpadr = df[var].std()
    gr = df.groupby(grupo)[var]
    dc = gr.describe()
    dc = dc.reset_index()
    dc[score] = dc['mean'].apply(lambda x : (x - media) / dpadr)
    dcz = dc[[grupo, score]]
    df = df.merge(dcz, on=grupo, how='left')
    return df

def rfm_variables(df):
    f_sales = len(df)
    f_profit = len(df[df['Profit'] > 0])
    m_sales = round(df['Sales'].sum(),2)
    m_profit = round(df['Profit'].sum(),2)
    m_qty = df['Quantity'].sum()
    m_deliver = round(df['Shipping Cost'].sum(),2)
    df_sort = df[['Order Date']].sort_values(by='Order Date').drop_duplicates()
    df_sort['diff'] = df_sort['Order Date'] - df_sort['Order Date'].shift(1)
    df_sort['diff_int'] = df_sort['diff'].dt.days
    r_days = round(df_sort['diff_int'].mean(),2)
    return f_sales, f_profit, m_sales, m_profit, m_qty, m_deliver, r_days

def fit_data(data, variable):
    rfm = []
    variaveis = data[variable].unique()
    for variavel in variaveis:
        dados = data[data[variable] == variavel]
        f_vendas, f_lucro, m_vendas, m_lucro, m_qtde, m_entrega, \
            r_dias = rfm_variables(dados)
        rfm.insert(0, [
            variavel, 
            m_vendas, 
            m_lucro, 
            m_qtde, 
            m_entrega, 
            r_dias, 
            f_vendas, 
            f_lucro
        ])
    return pd.DataFrame(
        rfm, 
        columns = [
            'referencia', 
            'm_vendas', 
            'm_lucro', 
            'm_qtde', 
            'm_entrega', 
            'r_dias', 
            'f_vendas', 
            'f_lucro'
        ]
    )

def outliers_detection(orig, vars):
    sc_x = StandardScaler()
    datax = sc_x.fit_transform(vars)
    clf = KNN().fit(vars)
    outliers = clf.predict(vars)
    dataod = orig.copy()
    dataod['outlier'] = outliers
    return dataod