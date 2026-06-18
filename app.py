import streamlit as st
import pandas as pd
import duckdb
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from datetime import date
import glob

# Configuração da página corporativa
st.set_page_config(page_title="Valuation Home 2 Invest", layout="wide")
st.title("🏢 Sistema de Valuation Inteligente - Home 2 Invest")

# --- VERIFICAÇÃO DE DADOS ---
arquivos = glob.glob('base_itbi_parte_*.parquet')
if not arquivos:
    st.error("Erro Crítico: Ficheiros de dados não encontrados na raiz do repositório.")
    st.stop()

# --- FUNÇÕES DE APOIO ---
def formata_moeda(valor):
    try:
        if pd.isna(valor): return "-"
        return f"R$ {float(valor):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except:
        return "-"

# --- BARRA LATERAL ---
st.sidebar.header("📍 Parâmetros de Busca")
rua = st.sidebar.text_input("Logradouro (ex: Rua Tamanas)")
num = st.sidebar.text_input("Número (Opcional)")
raio = st.sidebar.slider("Raio de busca (metros)", 100, 2500, 500)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtros de Ativo")
tipo = st.sidebar.selectbox("Uso do Imóvel", ["Residenciais", "Apartamentos"])

# Filtro temporal preservado para consistência de mercado
ano_min, ano_max = st.sidebar.slider(
    "Ano da Transação", 
    min_value=2010, 
    max_value=date.today().year, 
    value=(2020, date.today().year)
)

st.sidebar.markdown("---")
st.sidebar.header("📐 Dimensões do Alvo (Opcional)")
area_const_alvo = st.sidebar.number_input("Área Construída Alvo (m²)", min_value=0, value=0, step=10)
area_terr_alvo = st.sidebar.number_input("Área do Terreno Alvo (m²)", min_value=0, value=0, step=10)

# --- MOTOR DE EXECUÇÃO OTIMIZADO ---
if rua:
    with st.spinner("A mapear o perímetro espacial e a calcular o Valuation..."):
        geolocator = Nominatim(user_agent="h2i_valuation_pro_v6")
        
        # Montagem dinâmica do endereço
        endereco_busca = f"{rua}, {num}, São Paulo, SP" if num else f"{rua}, São Paulo, SP"
        loc = geolocator.geocode(endereco_busca, timeout=10)
        
        if loc:
            lat_c, lon_c = loc.latitude, loc.longitude
            
            # --- CONSTRUÇÃO DINÂMICA DOS FILTROS SQL ---
            filtros_sql = []
            
            if tipo == "Residenciais":
                filtros_sql.append("(UPPER(\"Descrição do uso (IPTU)\") LIKE '%RESIDÊN%' OR UPPER(\"Descrição do uso (IPTU)\") LIKE '%CASA%')")
            elif tipo == "Apartamentos":
                filtros_sql.append("UPPER(\"Descrição do uso (IPTU)\") LIKE '%APARTAMENTO%'")
            
            filtros_sql.append(f"TRY_CAST(REGEXP_EXTRACT(\"Data de Transação\", '(\\d{{4}})') AS INTEGER) BETWEEN {ano_min} AND {ano_max}")
            
            if area_const_alvo > 0:
                filtros_sql.append(f"TRY_CAST(\"Área Construída (m2)\" AS FLOAT) BETWEEN {area_const_alvo * 0.75} AND {area_const_alvo * 1.25}")
            if area_terr_alvo > 0:
                filtros_sql.append(f"TRY_CAST(\"Área do Terreno (m2)\" AS FLOAT) BETWEEN {area_terr_alvo * 0.75} AND {area_terr_alvo * 1.25}")
                
            condicao_extra = " AND " + " AND ".join(filtros_sql) if filtros_sql else ""
            
            query = f"""
            WITH base_distancia AS (
                SELECT *,
                (6371000 * acos(
                    cos(radians({lat_c})) * cos(radians(Latitude)) * cos(radians(Longitude) - radians({lon_c})) + 
                    sin(radians({lat_c})) * sin(radians(Latitude))
                )) as dist_metros
                FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true)
                WHERE Latitude IS NOT NULL {condicao_extra}
            )
            SELECT * FROM base_distancia WHERE dist_metros <= {raio}
            """
            
            try:
                df = duckdb.query(query).df()
            except Exception as e:
                st.error(f"Falha técnica na consulta ao Datalake: {e}")
                st.stop()
            
            if not df.empty:
                col_val = 'Valor de Transação (declarado pelo contribuinte)'
                col_area = 'Área Construída (m2)'
                col_terr = 'Área do Terreno (m2)'
                col_ano = 'Ano_Construcao_Geo'
                
                df[col_val] = pd.to_numeric(df[col_val], errors='coerce')
                df[col_area] = pd.to_numeric(df[col_area], errors='coerce')
                df[col_terr] = pd.to_numeric(df[col_terr], errors='coerce')
                df[col_ano] = pd.to_numeric(df[col_ano], errors='coerce')
                
                df = df.dropna(subset=[col_val])

                if not df.empty:
                    ano_atual = date.today().year
                    anos_validos = df[col_ano][(df[col_ano] > 1800) & (df[col_ano] <= ano_atual)]
                    
                    if not anos_validos.empty:
                        idade_media = ano_atual - anos_validos.mean()
                        texto_idade = f"{int(idade_media)} anos"
                    else:
                        texto_idade = "Sem Registro Oficial"

                    # --- PAINEL DE MÉTRICAS ---
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Amostras Encontradas", len(df))
                    col2.metric("Valor Médio", formata_moeda(df[col_val].mean()))
                    col3.metric("Média / m² Construído", formata_moeda((df[col_val] / df[col_area]).mean()))
                    col4.metric("Idade Média Predial", texto_idade)
                    
                    st.markdown("---")
                    
                    # --- GRÁFICO: ANÁLISE DE VALORIZAÇÃO (MODERNIZADAS VS ANTIGAS) ---
                    st.subheader("📈 Análise de Valorização: Modernizadas vs Antigas")
                    st.markdown("Comparativo do prêmio de mercado (Valor/m²) gerado por modernizações/retrofits recentes na região.")
                    
                    df_grafico = df.copy()
                    
                    # Regra de Classificação de Idade
                    def classificar_idade(ano):
                        if pd.isna(ano) or ano <= 1800:
                            return 'Sem Classificação'
                        elif ano >= 2018:
                            return 'Modernizadas (≥ 2018)'
                        else:
                            return 'Antigas (< 2018)'
                            
                    df_grafico['Categoria'] = df_grafico[col_ano].apply(classificar_idade)
                    df_grafico = df_grafico[df_grafico['Categoria'] != 'Sem Classificação']
                    
                    if not df_grafico.empty:
                        # Cálculo dos indexadores
                        df_grafico['R$/m² Construído'] = df_grafico[col_val] / df_grafico[col_area]
                        df_grafico['R$/m² Terreno'] = df_grafico.apply(
                            lambda r: r[col_val] / r[col_terr] if pd.notna(r[col_terr]) and r[col_terr] > 0 else None, axis=1
                        )
                        
                        # Agrupamento para o gráfico de barras duplas
                        resumo_grafico = df_grafico.groupby('Categoria')[['R$/m² Construído', 'R$/m² Terreno']].mean()
                        
                        # Renderiza o gráfico nativo do Streamlit
                        st.bar_chart(resumo_grafico, use_container_width=True)
                    else:
                        st.info("Não há dados suficientes com Ano de Construção oficial do GeoSampa neste raio para plotar o comparativo.")
                    
                    st.markdown("---")
                    
                    # --- MAPA DE LOCALIZAÇÃO ---
                    st.subheader("📍 Distribuição Espacial")
                    m = folium.Map([lat_c, lon_c], zoom_start=15)
                    
                    txt_alvo = f"Alvo: {rua}, {num}" if num else f"Alvo: {rua} (Centro)"
                    folium.Marker([lat_c, lon_c], popup=txt_alvo, icon=folium.Icon(color="red", icon="home")).add_to(m)
                    folium.Circle([lat_c, lon_c], radius=raio, color="blue", fill=True, fill_opacity=0.1).add_to(m)
                    
                    for _, r in df.iterrows():
                        if pd.notna(r['Latitude']) and pd.notna(r['Longitude']):
                            popup_txt = f"{r.get('Nome do Logradouro', 'Sem Rua')}, {r.get('Número', '')}<br><b>Valor:</b> {formata_moeda(r[col_val])}"
                            folium.CircleMarker([r['Latitude'], r['Longitude']], radius=6, color="darkblue", fill_color="lightblue", fill_opacity=0.8, popup=popup_txt).add_to(m)
                    
                    folium_static(m, width=1200, height=500)
                    
                    st.markdown("---")
                    
                    # --- TABELA SECUNDÁRIA: TOP 5 VALOR / ÁREA DE TERRENO ---
                    st.subheader("🔝 Top 5 Amostras mais Representativas (Valor / m² Terreno)")
                    
                    df_terreno_valido = df[df[col_terr] > 0].copy()
                    if not df_terreno_valido.empty:
                        df_terreno_valido['Valor/m² Terreno'] = df_terreno_valido[col_val] / df_terreno_valido[col_terr]
                        df_top5 = df_terreno_valido.sort_values(by='Valor/m² Terreno', ascending=False).head(5).copy()
                        
                        df_top5_visual = df_top5.drop(columns=['dist_metros'], errors='ignore')
                        df_top5_visual['Valor/m² Terreno'] = df_top5_visual['Valor/m² Terreno'].apply(formata_moeda)
                        df_top5_visual[col_val] = df_top5_visual[col_val].apply(formata_moeda)
                        st.dataframe(df_top5_visual, use_container_width=True)
                    else:
                        st.info("Nenhuma amostra com área de terreno válida para calcular a tabela de indicadores.")

                    st.markdown("---")

                    # --- TABELA COMPLETA ---
                    st.subheader("📋 Planilha de Análise de Amostras Detalhada")
                    df_visual = df.drop(columns=['dist_metros'], errors='ignore').copy()
                    df_visual[col_val] = df_visual[col_val].apply(formata_moeda)
                    st.dataframe(df_visual, use_container_width=True)
                else:
                    st.warning("Imóveis identificados no perímetro, mas nenhum possui registros financeiros computáveis.")
            else:
                st.warning(f"Nenhum comparável localizado no raio de {raio}m para as especificações digitadas.")
        else:
            st.error("Não foi possível geocodificar o logradouro digitado. Verifique o nome da rua.")
else:
    st.info("👈 Indique o Logradouro na barra lateral para iniciar a pesquisa espacial.")
