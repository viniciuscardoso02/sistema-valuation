import streamlit as st
import pandas as pd
import duckdb
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from datetime import date
import altair as alt
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
    with st.spinner("A mapear o perímetro, analisar expansões históricas e calcular Valuation..."):
        geolocator = Nominatim(user_agent="h2i_valuation_pro_v9")
        
        endereco_busca = f"{rua}, {num}, São Paulo, SP" if num else f"{rua}, São Paulo, SP"
        loc = geolocator.geocode(endereco_busca, timeout=10)
        
        if loc:
            lat_c, lon_c = loc.latitude, loc.longitude
            
            # Construção dos filtros SQL
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
                df['Ano_Transacao'] = df['Data de Transação'].astype(str).str.extract(r'(\d{4})').astype(float)
                
                df = df.dropna(subset=[col_val, 'Ano_Transacao'])

                if not df.empty:
                    # INTELIGÊNCIA DE RETROFIT: Histórico de Variação de Área
                    df['Chave_Imovel'] = df['N° do Cadastro (SQL)'].fillna(df['Nome do Logradouro'] + df['Número'])
                    
                    def classificar_retrofit(grupo):
                        areas_unicas = grupo[col_area].dropna().unique()
                        houve_expansao = len(areas_unicas) > 1 and (max(areas_unicas) - min(areas_unicas)) > 10
                        
                        categorias = []
                        for _, row in grupo.iterrows():
                            if pd.notna(row[col_ano]) and row[col_ano] >= 2018:
                                categorias.append('Modernizado (≥ 2018)')
                            elif houve_expansao and row[col_area] == max(areas_unicas):
                                categorias.append('Modernizado (Retrofit)')
                            else:
                                categorias.append('Antigo')
                        grupo['Categoria'] = categorias
                        return grupo
                        
                    df = df.groupby('Chave_Imovel', group_keys=False).apply(classificar_retrofit)
                    df = df[(df['Ano_Transacao'] >= ano_min) & (df['Ano_Transacao'] <= ano_max)]

                    if not df.empty:
                        ano_atual = date.today().year
                        anos_validos = df[col_ano][(df[col_ano] > 1800) & (df[col_ano] <= ano_atual)]
                        idade_media = ano_atual - anos_validos.mean() if not anos_validos.empty else None
                        texto_idade = f"{int(idade_media)} anos" if idade_media else "Sem Registro"

                        # --- PAINEL DE MÉTRICAS ---
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Amostras no Período", len(df))
                        col2.metric("Valor Médio", formata_moeda(df[col_val].mean()))
                        col3.metric("Média / m² Construído", formata_moeda((df[col_val] / df[col_area]).mean()))
                        col4.metric("Idade Média Predial", texto_idade)
                        
                        st.markdown("---")
                        
                        # --- GRÁFICO: ÁGIO DE MERCADO ---
                        st.subheader("📊 Ágio de Mercado: Modernizadas vs Antigas (Por m² de Terreno)")
                        
                        df_grafico = df.copy()
                        df_grafico = df_grafico.dropna(subset=[col_area, col_terr])
                        df_grafico = df_grafico[df_grafico[col_terr] > 0]
                        
                        if not df_grafico.empty:
                            df_grafico['R$/m² Terreno'] = df_grafico[col_val] / df_grafico[col_terr]
                            
                            # CLASSIFICAÇÃO EXATA DO EIXO X
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
                            
                            resumo_grafico = df_grafico.groupby(['Faixa de Área', 'Status'])['R$/m² Terreno'].mean().reset_index()
                            # FORMATAÇÃO DO TEXTO NA COLUNA
                            resumo_grafico['Texto_Valor'] = resumo_grafico['R$/m² Terreno'].apply(lambda x: f"R$ {int(x):,}".replace(',', '.') + "/m²")
                            
                            ordem_faixas = ['<300m²', '300 a 399m²', '400 a 499m²', '500 a 599m²', '600 a 699m²', '700 a 799m²', '800 a 899m²', '≥900m²']
                            
                            # BASE DO GRÁFICO
                            base = alt.Chart(resumo_grafico).encode(
                                x=alt.X('Status:N', title=None, axis=alt.Axis(labels=False, ticks=False)),
                                y=alt.Y('R$/m² Terreno:Q', title=None, axis=alt.Axis(labels=False, grid=False, ticks=False)),
                                color=alt.Color('Status:N', scale=alt.Scale(domain=['Antigo', 'Modernizado'], range=['#bdc3c7', '#27ae60']), legend=alt.Legend(title=None, orient="top", labelFontSize=12)),
                                tooltip=[alt.Tooltip('Faixa de Área:N', title='Área'), alt.Tooltip('Status:N'), alt.Tooltip('Texto_Valor:N', title='Valor Médio')]
                            ).properties(width=65, height=350)
                            
                            # COLUNAS AFINADAS E ARREDONDADAS
                            bars = base.mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4, size=24)
                            
                            # RÓTULOS P/M² NA VERTICAL
                            text = base.mark_text(
                                align='left',
                                baseline='middle',
                                dy=-5,
                                angle=270,
                                fontSize=11,
                                fontWeight='bold',
                                color='#2c3e50'
                            ).encode(
                                text='Texto_Valor:N'
                            )
                            
                            # AGRUPAMENTO FINAL LADO A LADO
                            grafico = alt.layer(bars, text).facet(
                                column=alt.Column('Faixa de Área:N', title=None, sort=ordem_faixas, header=alt.Header(labelFontSize=12, labelFontWeight='bold', labelOrient='bottom'))
                            ).configure_view(stroke='transparent')
                            
                            st.altair_chart(grafico, use_container_width=True)
                        else:
                            st.info("Não há dados com área de terreno e construção preenchidas suficientes para gerar o gráfico.")
                        
                        st.markdown("---")
                        
                        # --- MAPA DE LOCALIZAÇÃO ---
                        st.subheader("📍 Distribuição Espacial")
                        m = folium.Map([lat_c, lon_c], zoom_start=15)
                        
                        txt_alvo = f"Alvo: {rua}, {num}" if num else f"Alvo: {rua} (Centro)"
                        folium.Marker([lat_c, lon_c], popup=txt_alvo, icon=folium.Icon(color="red", icon="home")).add_to(m)
                        folium.Circle([lat_c, lon_c], radius=raio, color="blue", fill=True, fill_opacity=0.1).add_to(m)
                        
                        for _, r in df.iterrows():
                            if pd.notna(r['Latitude']) and pd.notna(r['Longitude']):
                                cor_pino = "green" if 'Modernizado' in r.get('Categoria', '') else "gray"
                                popup_txt = f"<b>{r.get('Categoria', '')}</b><br>{r.get('Nome do Logradouro', '')}<br>Valor: {formata_moeda(r[col_val])}<br>Área: {r.get(col_area)}m²"
                                folium.CircleMarker([r['Latitude'], r['Longitude']], radius=6, color=cor_pino, fill_color=cor_pino, fill_opacity=0.8, popup=popup_txt).add_to(m)
                        
                        folium_static(m, width=1200, height=500)
                        
                        st.markdown("---")
                        
                        # --- TABELAS ---
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
                        st.warning(f"Existem transações, mas nenhuma dentro da janela de anos ({ano_min}-{ano_max}).")
                else:
                    st.warning("Imóveis no perímetro não possuem registros financeiros.")
            else:
                st.warning(f"Nenhum comparável localizado no raio de {raio}m.")
        else:
            st.error("Não foi possível geocodificar o logradouro.")
else:
    st.info("👈 Indique o Logradouro para iniciar a pesquisa espacial.")
