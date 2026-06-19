import streamlit as st
import pandas as pd
import duckdb
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from datetime import date
import altair as alt
import glob
import numpy as np

# Configuração da página corporativa
st.set_page_config(page_title="Valuation Home 2 Invest", layout="wide")
st.title("🏢 Sistema de Valuation Inteligente - Home 2 Invest")

# --- VERIFICAÇÃO DE DADOS HIGIENIZADOS ---
if not glob.glob('base_itbi_limpa_*.parquet'):
    st.error("Erro Crítico: Os arquivos fatiados 'base_itbi_limpa_*.parquet' não foram encontrados na raiz do repositório.")
    st.stop()

# --- FUNÇÕES DE APOIO ---
def formata_moeda(valor):
    try:
        if pd.isna(valor): return "-"
        return f"R$ {float(valor):,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    except:
        return "-"

def remover_acentos(txt):
    if pd.isna(txt): return ""
    txt = str(txt).upper().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', txt) if unicodedata.category(c) != 'Mn')

# --- CARREGAMENTO INSTANTÂNEO DE BAIRROS ---
@st.cache_data
def carregar_lista_bairros():
    try:
        query = "SELECT DISTINCT Bairro FROM read_parquet('base_itbi_limpa_*.parquet') WHERE Bairro IS NOT NULL"
        df = duckdb.query(query).df()
        return ["Selecione..."] + sorted(df['Bairro'].astype(str).unique())
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar bairros: {e}")
        return ["Selecione..."]

bairros_disp = carregar_lista_bairros()
tipos_disp = ["Residenciais", "Apartamentos"]

# --- BARRA LATERAL ---
st.sidebar.header("📍 Parâmetros de Busca")
rua = st.sidebar.text_input("Logradouro (Para busca por Raio)")
num = st.sidebar.text_input("Número (Opcional)")
raio = st.sidebar.slider("Raio de busca (metros)", 100, 2500, 500)

st.sidebar.markdown("**OU**")
bairro_alvo = st.sidebar.selectbox("Buscar por Bairro Inteiro", bairros_disp)

st.sidebar.markdown("---")
st.sidebar.header("🎯 Filtros de Ativo")
tipo = st.sidebar.selectbox("Uso do Imóvel", tipos_disp)

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

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configurações Avançadas")
remover_outliers = st.sidebar.toggle("Remover Outliers (Método IQR)", value=True)

# --- MOTOR DE EXECUÇÃO ---
if rua or bairro_alvo != "Selecione...":
    with st.spinner("A processar inteligência de mercado de alta velocidade..."):
        
        filtros_sql = []
        if tipo == "Residenciais":
            filtros_sql.append("(UPPER(\"Descrição do uso (IPTU)\") LIKE '%RESIDÊN%' OR UPPER(\"Descrição do uso (IPTU)\") LIKE '%CASA%')")
        elif tipo == "Apartamentos":
            filtros_sql.append("UPPER(\"Descrição do uso (IPTU)\") LIKE '%APARTAMENTO%'")
        
        if area_const_alvo > 0:
            filtros_sql.append(f"TRY_CAST(\"Área Construída (m2)\" AS FLOAT) BETWEEN {area_const_alvo * 0.75} AND {area_const_alvo * 1.25}")
        if area_terr_alvo > 0:
            filtros_sql.append(f"TRY_CAST(\"Área do Terreno (m2)\" AS FLOAT) BETWEEN {area_terr_alvo * 0.75} AND {area_terr_alvo * 1.25}")
            
        condicao_extra = " AND " + " AND ".join(filtros_sql) if filtros_sql else ""
        
        df_bruto = pd.DataFrame()
        lat_c, lon_c = None, None
        
        try:
            if rua:
                geolocator = Nominatim(user_agent="h2i_valuation_pro_v18")
                endereco_busca = f"{rua}, {num}, São Paulo, SP" if num else f"{rua}, São Paulo, SP"
                loc = geolocator.geocode(endereco_busca, timeout=10)
                
                if loc:
                    lat_c, lon_c = loc.latitude, loc.longitude
                    query = f"""
                    WITH base_distancia AS (
                        SELECT *,
                        (6371000 * acos(
                            cos(radians={lat_c})) * cos(radians(Latitude)) * cos(radians(Longitude) - radians({lon_c})) + 
                            sin(radians({lat_c})) * sin(radians(Latitude))
                        )) as dist_metros
                        FROM read_parquet('base_itbi_limpa_*.parquet')
                        WHERE Latitude IS NOT NULL {condicao_extra}
                    )
                    SELECT * FROM base_distancia WHERE dist_metros <= {raio}
                    """
                    df_bruto = duckdb.query(query).df()
                else:
                    st.error("Logradouro não encontrado no mapa.")
                    st.stop()
            else:
                bairro_sql = bairro_alvo.replace("'", "''")
                query = f"""
                SELECT * FROM read_parquet('base_itbi_limpa_*.parquet')
                WHERE Bairro = '{bairro_sql}' {condicao_extra}
                """
                df_bruto = duckdb.query(query).df()
                
        except Exception as e:
            st.error(f"Falha técnica na consulta: {e}")
            st.stop()
            
        if not df_bruto.empty:
            df = df_bruto.copy()
            col_val = 'Valor de Transação (declarado pelo contribuinte)'
            col_area = 'Área Construída (m2)'
            col_terr = 'Área do Terreno (m2)'
            col_ano = 'Ano_Construcao_Geo'
            
            df[col_val] = pd.to_numeric(df[col_val], errors='coerce')
            df[col_area] = pd.to_numeric(df[col_area], errors='coerce')
            df[col_terr] = pd.to_numeric(df[col_terr], errors='coerce')
            df[col_ano] = pd.to_numeric(df[col_ano], errors='coerce')
            df['Ano_Transacao'] = df['Data de Transação'].astype(str).str.extract(r'(\d{4})').astype(float)
            
            df = df.dropna(subset=[col_val, 'Ano_Transacao', col_area])

            # Filtro Estatístico de Outliers
            if remover_outliers and not df.empty:
                df['Preco_m2_Construido'] = df[col_val] / df[col_area]
                Q1 = df['Preco_m2_Construido'].quantile(0.25)
                Q3 = df['Preco_m2_Construido'].quantile(0.75)
                IQR = Q3 - Q1
                df = df[(df['Preco_m2_Construido'] >= (Q1 - 1.5 * IQR)) & (df['Preco_m2_Construido'] <= (Q3 + 1.5 * IQR))]
                df = df.drop(columns=['Preco_m2_Construido'])

            if not df.empty:
                # Inteligência Vetorizada de Retrofit
                df['Chave_Imovel'] = df['N° do Cadastro (SQL)'].fillna(df['Nome do Logradouro'] + df['Número'].fillna(''))
                df['Max_Area_Historica'] = df.groupby('Chave_Imovel')[col_area].transform('max')
                df['Min_Area_Historica'] = df.groupby('Chave_Imovel')[col_area].transform('min')
                df['Qtd_Areas_Unicas'] = df.groupby('Chave_Imovel')[col_area].transform('nunique')
                df['Houve_Expansao'] = (df['Qtd_Areas_Unicas'] > 1) & ((df['Max_Area_Historica'] - df['Min_Area_Historica']) > 10)

                condicoes = [
                    (df[col_ano] >= 2018),
                    (df['Houve_Expansao'] & (df[col_area] == df['Max_Area_Historica']))
                ]
                escolhas = ['Modernizado (≥ 2018)', 'Modernizado (Retrofit)']
                df['Categoria'] = np.select(condicoes, escolhas, default='Antigo')
                df = df.drop(columns=['Max_Area_Historica', 'Min_Area_Historica', 'Qtd_Areas_Unicas', 'Houve_Expansao'])
                
                df = df[(df['Ano_Transacao'] >= ano_min) & (df['Ano_Transacao'] <= ano_max)]

                if not df.empty:
                    ano_atual = date.today().year
                    anos_validos = df[col_ano][(df[col_ano] > 1800) & (df[col_ano] <= ano_atual)]
                    idade_media = ano_atual - anos_validos.median() if not anos_validos.empty else None
                    texto_idade = f"{int(idade_media)} anos" if pd.notna(idade_media) else "Sem Registro"

                    # --- INTERFACE ---
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Amostras no Período", len(df))
                    col2.metric("Valor Mediano", formata_moeda(df[col_val].median()))
                    col3.metric("Mediana / m² Construído", formata_moeda((df[col_val] / df[col_area]).median()))
                    col4.metric("Idade Mediana Predial", texto_idade)
                    
                    st.markdown("---")
                    
                    # Gráfico
                    st.subheader("📊 Ágio de Mercado: Modernizadas vs Antigas (Por m² de Terreno)")
                    df_grafico = df.dropna(subset=[col_area, col_terr])
                    df_grafico = df_grafico[df_grafico[col_terr] > 0].copy()
                    
                    if not df_grafico.empty:
                        df_grafico['R$/m² Terreno'] = df_grafico[col_val] / df_grafico[col_terr]
                        
                        def classificar_faixa_area(area):
                            if area < 300: return '<300m²'
                            elif area < 400: return '300 a 399m²'
                            elif area < 500: return '400 a 499m²'
                            elif area < 600: return '500 a 599m²'
                            elif area < 700: return '600 a 699m²'
                            elif area < 800: return '700 a 799m²'
                            elif area < 900: return '800 a 899m²'
                            else: return '≥900m²'
                        
                        df_grafico['Faixa de Área'] = df_grafico[col_area].apply(classificar_faixa_area)
                        df_grafico['Status'] = df_grafico['Categoria'].apply(lambda x: 'Modernizado' if 'Modernizado' in x else 'Antigo')
                        
                        resumo_grafico = df_grafico.groupby(['Faixa de Área', 'Status'])['R$/m² Terreno'].median().reset_index()
                        resumo_grafico['Texto_Valor'] = resumo_grafico['R$/m² Terreno'].apply(lambda x: f"R$ {int(x):,}".replace(',', '.') + "/m²")
                        ordem_faixas = ['<300m²', '300 a 399m²', '400 a 499m²', '500 a 599m²', '600 a 699m²', '700 a 799m²', '800 a 899m²', '≥900m²']
                        
                        base = alt.Chart(resumo_grafico).encode(
                            x=alt.X('Status:N', title=None, axis=alt.Axis(labels=False, ticks=False)),
                            y=alt.Y('R$/m² Terreno:Q', title=None, axis=alt.Axis(labels=False, grid=False, ticks=False)),
                            color=alt.Color('Status:N', scale=alt.Scale(domain=['Antigo', 'Modernizado'], range=['#bdc3c7', '#27ae60']), legend=alt.Legend(title=None, orient="top", labelFontSize=12)),
                            tooltip=[alt.Tooltip('Faixa de Área:N', title='Área'), alt.Tooltip('Status:N'), alt.Tooltip('Texto_Valor:N', title='Valor Mediano')]
                        ).properties(width=65, height=350)
                        
                        bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=24)
                        text = base.mark_text(align='left', baseline='middle', dy=-5, angle=270, fontSize=11, fontWeight='bold', color='#2c3e50').encode(text='Texto_Valor:N')
                        
                        grafico = alt.layer(bars, text).facet(column=alt.Column('Faixa de Área:N', title=None, sort=ordem_faixas, header=alt.Header(labelFontSize=12, labelFontWeight='bold', labelOrient='bottom'))).configure_view(stroke='transparent')
                        st.altair_chart(grafico, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Mapa (Google Maps)
                    st.subheader("📍 Distribuição Espacial")
                    if rua and lat_c and lon_c:
                        m = folium.Map([lat_c, lon_c], zoom_start=15, tiles=None)
                    else:
                        m = folium.Map([df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=14, tiles=None)
                    
                    folium.TileLayer(tiles='http://mt1.google.com/vt/lyrs=m&x={x}&y={y}&z={z}', attr='Google Maps', name='Google Maps').add_to(m)
                    
                    if rua and lat_c and lon_c:
                        folium.Marker([lat_c, lon_c], popup=f"Alvo: {rua}", icon=folIcon(color="red", icon="home")).add_to(m)
                        folium.Circle([lat_c, lon_c], radius=raio, color="blue", fill=True, fill_opacity=0.1).add_to(m)
                        
                    for _, r in df.iterrows():
                        if pd.notna(r['Latitude']) and pd.notna(r['Longitude']):
                            cor_pino = "green" if 'Modernizado' in r.get('Categoria', '') else "gray"
                            popup_txt = f"<b>{r.get('Categoria', '')}</b><br>{r.get('Nome do Logradouro', '')}<br>Valor: {formata_moeda(r[col_val])}<br>Área: {r.get(col_area)}m²"
                            folium.CircleMarker([r['Latitude'], r['Longitude']], radius=6, color=cor_pino, fill_color=cor_pino, fill_opacity=0.8, popup=popup_txt).add_to(m)
                    
                    folium_static(m, width=1200, height=500)
                    
                    # Tabelas
                    st.markdown("---")
                    st.subheader("🔝 Top 5 (Valor / m² Terreno)")
                    df_terreno_valido = df[df[col_terr] > 0].copy()
                    if not df_terreno_valido.empty:
                        df_terreno_valido['Valor/m² Terreno'] = df_terreno_valido[col_val] / df_terreno_valido[col_terr]
                        df_top5 = df_terreno_valido.sort_values(by='Valor/m² Terreno', ascending=False).head(5).copy()
                        df_top5['Valor/m² Terreno'] = df_top5['Valor/m² Terreno'].apply(formata_moeda)
                        df_top5[col_val] = df_top5[col_val].apply(formata_moeda)
                        st.dataframe(df_top5.drop(columns=['dist_metros', 'Chave_Imovel'], errors='ignore'), use_container_width=True)

                    st.subheader("📋 Planilha Detalhada")
                    df_visual = df.drop(columns=['dist_metros', 'Chave_Imovel'], errors='ignore').copy()
                    df_visual[col_val] = df_visual[col_val].apply(formata_moeda)
                    st.dataframe(df_visual, use_container_width=True)
                else:
                    st.warning("Sem transações na janela de tempo selecionada.")
            else:
                st.warning("Nenhum imóvel restou após o filtro de outliers.")
        else:
            st.warning("Nenhum comparável localizado.")
else:
    st.info("👈 Indique um Logradouro ou selecione um Bairro para iniciar.")
