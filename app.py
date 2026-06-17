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

# --- MOTOR DE EXECUÇÃO OTIMIZADO ---
if rua and num:
    with st.spinner("A mapear o perímetro espacial e a calcular o Valuation..."):
        geolocator = Nominatim(user_agent="h2i_valuation_pro")
        loc = geolocator.geocode(f"{rua}, {num}, São Paulo, SP", timeout=10)
        
        if loc:
            lat_c, lon_c = loc.latitude, loc.longitude
            
            # Construção do filtro opcional de uso
            filtro_uso = f"AND \"Descrição do uso (IPTU)\" = '{tipo}'" if tipo != "Todos" else ""
            
            # ARQUITETURA OTIMIZADA: O filtro de raio (WHERE dist_metros <= raio) corre no motor SQL.
            # O Streamlit só recebe os dados finais, poupando a memória do servidor.
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
                # --- LIMPEZA ESTRUTURAL DE DADOS ---
                col_val = 'Valor de Transação (declarado pelo contribuinte)'
                col_area = 'Área Construída (m2)'
                col_ano = 'Ano_Construcao_Geo'
                
                # Força a tipagem numérica, descartando lixo textual (resolve o TypeError de médias)
                df[col_val] = pd.to_numeric(df[col_val], errors='coerce')
                df[col_area] = pd.to_numeric(df[col_area], errors='coerce')
                df[col_ano] = pd.to_numeric(df[col_ano], errors='coerce')
                
                # Remove transações que ficaram sem valor financeiro após a limpeza
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
                            popup_txt = f"{r.get('Nome do Logradouro', 'Sem Rua')}, {r.get('Número', '')}"
                            folium.CircleMarker([r['Latitude'], r['Longitude']], radius=6, color="darkblue", fill_color="lightblue", fill_opacity=0.8, popup=popup_txt).add_to(m)
                    
                    folium_static(m, width=1200, height=500)
                    
                    # --- TABELA DE SUPORTE ---
                    st.subheader("Extração da Base Filtrada")
                    df_visual = df.drop(columns=['dist_metros']).copy()
                    df_visual[col_val] = df_visual[col_val].apply(formata_moeda)
                    st.dataframe(df_visual, use_container_width=True)
                else:
                    st.warning("Foram encontrados imóveis na área, mas nenhum possui um registo financeiro válido.")
            else:
                st.warning(f"Sem transações identificadas num raio de {raio}m para este perfil. Alargue a busca.")
        else:
            st.error("Não foi possível processar as coordenadas deste logradouro.")
else:
    st.info("👈 Indique o Logradouro e o Número para acionar o motor espacial.")
