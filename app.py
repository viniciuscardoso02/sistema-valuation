import streamlit as st
import pandas as pd
import duckdb
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from datetime import date
import glob

# Configuração da página
st.set_page_config(page_title="Valuation Home 2 Invest", layout="wide")
st.title("🏢 Sistema de Valuation Inteligente - Home 2 Invest")

# --- VERIFICAÇÃO DE DADOS ---
arquivos = glob.glob('base_itbi_parte_*.parquet')
if not arquivos:
    st.error("Erro Crítico: Ficheiros de dados não encontrados na raiz.")
    st.stop()

# --- FUNÇÕES DE APOIO ---
def formata_moeda(valor):
    try:
        if pd.isna(valor): return "-"
        return f"R$ {float(valor):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except:
        return "-"

@st.cache_data
def obter_filtros():
    try:
        query = 'SELECT DISTINCT "Descrição do uso (IPTU)" FROM read_parquet("base_itbi_parte_*.parquet", union_by_name=true)'
        df = duckdb.query(query).df()
        return ["Todos"] + sorted([t for t in df.iloc[:,0].unique() if t and str(t) != '-'])
    except:
        return ["Todos"]

tipos_disp = obter_filtros()

# --- BARRA LATERAL ---
st.sidebar.header("📍 Parâmetros de Busca")
rua = st.sidebar.text_input("Logradouro (ex: Rua Tamanas)")
num = st.sidebar.text_input("Número")
raio = st.sidebar.slider("Raio de busca (metros)", 100, 2500, 500)
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtros de Ativo")
tipo = st.sidebar.selectbox("Uso do Imóvel", tipos_disp)

# NOVO: Slider duplo para o período das transações
ano_min, ano_max = st.sidebar.slider(
    "Ano da Transação", 
    min_value=2010, 
    max_value=date.today().year, 
    value=(2021, date.today().year)
)

# --- MOTOR DE EXECUÇÃO OTIMIZADO ---
if rua and num:
    with st.spinner("A mapear o perímetro espacial e a calcular o Valuation..."):
        geolocator = Nominatim(user_agent="h2i_valuation_pro")
        loc = geolocator.geocode(f"{rua}, {num}, São Paulo, SP", timeout=10)
        
        if loc:
            lat_c, lon_c = loc.latitude, loc.longitude
            
            filtro_uso = f"AND \"Descrição do uso (IPTU)\" = '{tipo}'" if tipo != "Todos" else ""
            
            # A busca espacial ultra-rápida continua no DuckDB
            query = f"""
            WITH base_distancia AS (
                SELECT *,
                (6371000 * acos(
                    cos(radians({lat_c})) * cos(radians(Latitude)) * cos(radians(Longitude) - radians({lon_c})) + 
                    sin(radians({lat_c})) * sin(radians(Latitude))
                )) as dist_metros
                FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true)
                WHERE Latitude IS NOT NULL {filtro_uso}
            )
            SELECT * FROM base_distancia WHERE dist_metros <= {raio}
            """
            
            try:
                df = duckdb.query(query).df()
            except Exception as e:
                st.error(f"Falha de execução no Datalake: {e}")
                st.stop()
            
            if not df.empty:
                # --- NOVO: FILTRO INTELIGENTE DE ANO DA TRANSAÇÃO ---
                col_data = 'Data de Transação'
                if col_data in df.columns:
                    # Extrai à força os 4 dígitos do ano (Regex) e converte para número
                    df['Ano_Transacao_Ext'] = df[col_data].astype(str).str.extract(r'(\d{4})').astype(float)
                    # Corta do DataFrame as transações fora do período selecionado
                    df = df[(df['Ano_Transacao_Ext'] >= ano_min) & (df['Ano_Transacao_Ext'] <= ano_max)]

                # --- LIMPEZA ESTRUTURAL DE DADOS ---
                col_val = 'Valor de Transação (declarado pelo contribuinte)'
                col_area = 'Área Construída (m2)'
                col_ano = 'Ano_Construcao_Geo'
                
                df[col_val] = pd.to_numeric(df[col_val], errors='coerce')
                df[col_area] = pd.to_numeric(df[col_area], errors='coerce')
                df[col_ano] = pd.to_numeric(df[col_ano], errors='coerce')
                
                df = df.dropna(subset=[col_val])

                if not df.empty:
                    # --- DASHBOARD ---
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Transações Válidas", len(df))
                    col2.metric("Valor Médio", formata_moeda(df[col_val].mean()))
                    col3.metric("Média / m² (Construído)", formata_moeda((df[col_val] / df[col_area]).mean()))
                    
                    ano_medio = df[col_ano].mean()
                    texto_idade = f"{int(date.today().year - ano_medio)} anos" if not pd.isna(ano_medio) else "Sem Registo"
                    col4.metric("Idade Média Oficial", texto_idade)
                    
                    st.markdown("---")
                    
                    # --- MAPA INTERATIVO ---
                    m = folium.Map([lat_c, lon_c], zoom_start=15)
                    folium.Marker([lat_c, lon_c], popup="Alvo", icon=folium.Icon(color="red", icon="home")).add_to(m)
                    folium.Circle([lat_c, lon_c], radius=raio, color="blue", fill=True, fill_opacity=0.1).add_to(m)
                    
                    for _, r in df.iterrows():
                        if pd.notna(r['Latitude']) and pd.notna(r['Longitude']):
                            # Agora o mapa exibe a data exata da venda ao clicar no pino
                            popup_txt = f"{r.get('Nome do Logradouro', 'Sem Rua')}, {r.get('Número', '')}<br><b>Data:</b> {r.get(col_data, 'N/A')}"
                            folium.CircleMarker([r['Latitude'], r['Longitude']], radius=6, color="darkblue", fill_color="lightblue", fill_opacity=0.8, popup=popup_txt).add_to(m)
                    
                    folium_static(m, width=1200, height=500)
                    
                    # --- TABELA DE SUPORTE ---
                    st.subheader("Extração da Base Filtrada")
                    colunas_remover = ['dist_metros', 'Ano_Transacao_Ext']
                    df_visual = df.drop(columns=[c for c in colunas_remover if c in df.columns]).copy()
                    df_visual[col_val] = df_visual[col_val].apply(formata_moeda)
                    st.dataframe(df_visual, use_container_width=True)
                else:
                    st.warning(f"Existem imóveis na área, mas nenhum possui um registo financeiro válido fechado entre {ano_min} e {ano_max}.")
            else:
                st.warning(f"Sem transações identificadas num raio de {raio}m. Alargue a busca ou remova os filtros.")
        else:
            st.error("Não foi possível processar as coordenadas deste endereço.")
else:
    st.info("👈 Indique o Logradouro e o Número para acionar o motor espacial.")
