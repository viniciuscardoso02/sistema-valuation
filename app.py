import streamlit as st
import pandas as pd
import glob

# Configurações de layout
st.set_page_config(page_title="Valuation ITBI SP", layout="wide")
st.title("🏢 Ferramenta de Análise Home 2 Invest")

# Função para carregar e unir as 15 partes do banco de dados
@st.cache_data
def carregar_dados():
    arquivos = sorted(glob.glob('base_itbi_parte_*.parquet'))
    df_lista = [pd.read_parquet(f) for f in arquivos]
    return pd.concat(df_lista, ignore_index=True)

with st.spinner('A carregar a base de dados... Isto pode demorar 1 minuto no primeiro acesso.'):
    df = carregar_dados()

# --- BARRA LATERAL COM FILTROS ---
st.sidebar.header("Filtros de Busca")
rua = st.sidebar.text_input("Nome da Rua / Logradouro")

coluna_uso = 'Descrição do uso (IPTU)'
tipos = ["Todos"] + sorted(list(df[coluna_uso].dropna().unique()))
tipo_selecionado = st.sidebar.selectbox("Tipo de Imóvel", tipos)

mod_filtro = st.sidebar.selectbox("Estado de Conservação", ["Ambos", "Apenas Modernizadas", "Apenas Antigas"])

# --- LÓGICA DE FILTRAGEM ---
dados_filtrados = df.copy()

if rua:
    dados_filtrados = dados_filtrados[dados_filtrados['Nome do Logradouro'].str.contains(rua, case=False, na=False)]

if tipo_selecionado != "Todos":
    dados_filtrados = dados_filtrados[dados_filtrados[coluna_uso] == tipo_selecionado]

if mod_filtro == "Apenas Modernizadas":
    dados_filtrados = dados_filtrados[dados_filtrados['Modernizada'] == True]
elif mod_filtro == "Apenas Antigas":
    dados_filtrados = dados_filtrados[dados_filtrados['Modernizada'] == False]

# --- RESULTADOS ---
st.subheader(f"📊 Resultados: {len(dados_filtrados)} transações encontradas")

coluna_valor = 'Valor de Transação (declarado pelo contribuinte)'
if len(dados_filtrados) > 0:
    v_num = pd.to_numeric(dados_filtrados[coluna_valor], errors='coerce')
    media = v_num.mean()
    st.metric("Preço Médio na Região", f"R$ {media:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

st.dataframe(dados_filtrados, use_container_width=True)
