import streamlit as st
import pandas as pd
import duckdb
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from datetime import date
import os
import glob

# Configuração da página
st.set_page_config(page_title="Valuation Home 2 Invest", layout="wide")
st.title("🏢 Sistema de Valuation Inteligente - Home 2 Invest")

# --- VERIFICAÇÃO DE DADOS ---
arquivos = glob.glob('base_itbi_parte_*.parquet')
if not arquivos:
    st.error("Erro: Arquivos de dados não encontrados. Verifique se estão na raiz.")
    st.stop()

# --- FUNÇÕES DE APOIO ---
def formata_moeda(valor):
    try:
        return f"R$ {float(valor):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except:
        return "-"

@st.cache_data
def obter_filtros():
    query = 'SELECT DISTINCT "Descrição do uso (IPTU)", Bairro FROM read_parquet("base_itbi_parte_*.parquet", union_by_name=true)'
    df = duckdb.query(query).df()
    tipos = ["Todos"] + sorted([t for t in df.iloc[:,0].unique() if t and str(t) != '-'])
    bairros = ["Todos"] + sorted([b for b in df.iloc[:,1].unique() if b and str(b) != '-'])
    return tipos, bairros

tipos_disp, bairros_disp = obter_filtros()

# --- BARRA LATERAL ---
st.sidebar.header("📍 Parâmetros de Busca")
rua = st.sidebar.text_input("Logradouro (ex: Rua Tamanas)")
num = st.sidebar.text_input("Número")
raio = st.sidebar.slider("Raio de busca (metros)", 100, 2000, 500)
st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtros de Ativo")
tipo = st.sidebar.selectbox("Uso do Imóvel", tipos_disp)
mod_filtro = st.sidebar.selectbox("Estado de Conservação", ["Ambos", "Apenas Modernizadas", "Apenas Antigas"])

# --- LÓGICA DE EXECUÇÃO ---
if rua and num:
    with st.spinner("Geocodificando endereço e filtrando base..."):
        geolocator = Nominatim(user_agent="h2i_valuation_v2")
        loc = geolocator.geocode(f"{rua}, {num}, São Paulo, SP", timeout=10)
        
        if loc:
            lat_c, lon_c = loc.latitude, loc.longitude
            
            # Construção da query SQL com filtros
            filtros_sql = []
            if tipo != "Todos": filtros_sql.append(f'"Descrição do uso (IPTU)" = \'{tipo}\'')
            if mod_filtro == "Apenas Modernizadas": filtros_sql.append("Modernizada = true")
            if mod_filtro == "Apenas Antigas": filtros_sql.append("Modernizada = false")
            condicao_extra = " AND " + " AND ".join(filtros_sql) if filtros_sql else ""
            
            query = f"""
            SELECT *,
            (6371000 * acos(cos(radians({lat_c})) * cos(radians(Latitude)) * cos(radians(Longitude) - radians({lon_c})) + sin(radians({lat_c})) * sin(radians(Latitude)))) as dist_metros
            FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true)
            WHERE Latitude IS NOT NULL {condicao_extra}
            """
            
            df_all = duckdb.query(query).df()
            df = df_all[df_all['dist_metros'] <= raio].copy()
            
            if not df.empty:
                # Métricas Dashboard
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Transações encontradas", len(df))
                col2.metric("Valor Médio", formata_moeda(df['Valor de Transação (declarado pelo contribuinte)'].mean()))
                col3.metric("Média / m² (Construído)", formata_moeda((df['Valor de Transação (declarado pelo contribuinte)'] / df['Área Construída (m2)']).mean()))
                col4.metric("Idade Média", f"{int(date.today().year - df['Ano_Construcao_Geo'].mean())} anos")
                
                st.markdown("---")
                
                # Mapa
                m = folium.Map([lat_c, lon_c], zoom_start=16)
                folium.Marker([lat_c, lon_c], icon=folium.Icon(color="red", icon="home")).add_to(m)
                folium.Circle([lat_c, lon_c], radius=raio, color="blue", fill=True, fill_opacity=0.1).add_to(m)
                for _, r in df.iterrows():
                    folium.CircleMarker([r['Latitude'], r['Longitude']], radius=5, popup=f"{r['Nome do Logradouro']}, {r['Número']}").add_to(m)
                folium_static(m)
                
                # Tabela
                st.subheader("Dados detalhados")
                df['Distância (m)'] = df['dist_metros'].astype(int)
                st.dataframe(df.drop(columns=['dist_metros']), use_container_width=True)
            else:
                st.warning("Nenhum imóvel encontrado neste raio.")
        else:
            st.error("Não foi possível localizar o endereço no mapa. Verifique o logradouro.")
else:
    st.info("👈 Digite um Logradouro e Número na barra lateral para iniciar.")
