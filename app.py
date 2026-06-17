import streamlit as st
import pandas as pd
import duckdb
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from datetime import date

st.set_page_config(page_title="Valuation por Raio - Home 2 Invest", layout="wide")
st.title("🏢 Sistema de Valuation Inteligente por Raio Geográfico")
st.markdown("---")

# Função em cache para extrair filtros rápidos
@st.cache_data
def obter_filtros_macro():
    query = """
    SELECT DISTINCT "Descrição do uso (IPTU)" 
    FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true) 
    """
    try:
        df_filtros = duckdb.query(query).df()
        tipos = ["Todos"] + sorted([t for t in df_filtros['Descrição do uso (IPTU)'].unique() if t and str(t) != '-'])
        return tipos
    except:
        return ["Todos"]

tipos_disp = obter_filtros_macro()

# --- BARRA LATERAL DE FILTROS ---
st.sidebar.header("📍 Parâmetros da Busca")

# Entradas separadas de endereço para garantir precisão na busca geográfica
logradouro_alvo = st.sidebar.text_input("Nome do Logradouro (Ex: Rua Tamanas)")
numero_alvo = st.sidebar.text_input("Número (Ex: 100)")
cidade_alvo = "São Paulo, SP, Brasil"

raio_km = st.sidebar.slider("Raio de Busca (em metros)", min_value=100, max_value=2500, value=500, step=50) / 1000.0

st.sidebar.subheader("🎯 Filtros do Ativo")
tipo_selecionado = st.sidebar.selectbox("Uso do Imóvel (IPTU)", tipos_disp)
mod_filtro = st.sidebar.selectbox("Estado de Conservação", ["Ambos", "Apenas Modernizadas", "Apenas Antigas"])

# Função para geocodificar o endereço digitado usando a API do OpenStreetMap
def buscar_coordenadas_alvo(rua, numero, cidade):
    if not rua:
        return None
    geolocator = Nominatim(user_agent="home2invest_valuation_system")
    endereco_completo = f"{rua}, {numero}, {cidade}" if numero else f"{rua}, {cidade}"
    try:
        location = geolocator.geocode(endereco_completo, timeout=10)
        if location:
            return location.latitude, location.longitude
        return None
    except GeocoderTimedOut:
        return None

# --- PROCESSAMENTO PRINCIPAL ---
if logradouro_alvo:
    coordenadas_centro = buscar_coordenadas_alvo(logradouro_alvo, numero_alvo, city_alvo:=cidade_alvo)
    
    if coordenadas_centro:
        lat_c, lon_c = coordenadas_centro
        
        with st.spinner("Varrendo Datalake e aplicando filtros espaciais em tempo real..."):
            
            # Condições base de filtragem de atributos
            condicoes_filtros = []
            if tipo_selecionado != "Todos":
                condicoes_filtros.append(f"\"Descrição do uso (IPTU)\" = '{tipo_selecionado}'")
            if mod_filtro == "Apenas Modernizadas":
                condicoes_filtros.append("Modernizada = true")
            elif mod_filtro == "Apenas Antigas":
                condicoes_filtros.append("Modernizada = false")
                
            clausula_filtros_adicionais = " AND ".join(condicoes_filtros) if condicoes_filtros else "1=1"
            
            # Query executando a Fórmula de Haversine diretamente em SQL para máxima velocidade (Performance DuckDB)
            query_raio = f"""
            SELECT *,
                   (6371 * acos(
                       cos(radians({lat_c})) * cos(radians(Latitude)) * cos(radians(Longitude) - radians({lon_c})) + 
                       sin(radians({lat_c})) * sin(radians(Latitude))
                   )) as Distancia_KM
            FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true)
            WHERE Latitude IS NOT NULL 
              AND Longitude IS NOT NULL
              AND {clausula_filtros_adicionais}
            """
            
            try:
                # Executa e filtra apenas o que está contido no raio determinado
                dados_com_distancia = duckdb.query(query_raio).df()
                dados_filtrados = dados_com_distancia[dados_com_distancia['Distancia_KM'] <= raio_km].copy()
                
                if not dados_filtrados.empty:
                    # --- CÁLCULO DAS MÉTRICAS DE VALUATION ---
                    coluna_valor = 'Valor de Transação (declarado pelo contribuinte)'
                    col_area_terreno = 'Área do Terreno (m2)'
                    col_area_const = 'Área Construída (m2)'
                    
                    v_num = pd.to_numeric(dados_filtrados[coluna_valor], errors='coerce')
                    a_terreno = pd.to_numeric(dados_filtrados[col_area_terreno], errors='coerce')
                    a_const = pd.to_numeric(dados_filtrados[col_area_const], errors='coerce')
                    
                    df_calc = pd.DataFrame({'Valor': v_num, 'Area_T': a_terreno, 'Area_C': a_const})
                    
                    media_absoluta = df_calc['Valor'].mean()
                    
                    df_terreno_valido = df_calc[df_calc['Area_T'] > 0]
                    media_m2_terreno = (df_terreno_valido['Valor'] / df_terreno_valido['Area_T']).mean() if not df_terreno_valido.empty else 0
                    
                    df_const_valido = df_calc[df_calc['Area_C'] > 0]
                    media_m2_const = (df_const_valido['Valor'] / df_const_valido['Area_C']).mean() if not df_const_valido.empty else 0
                    
                    # Cálculo da Idade Média Real Oficial puxada do GeoSampa
                    idade_media_texto = "Sem cadastro"
                    if 'Ano_Construcao_Geo' in dados_filtrados.columns:
                        anos_validos = pd.to_numeric(dados_filtrados['Ano_Construcao_Geo'], errors='coerce')
                        anos_validos = anos_validos[anos_validos > 1500]
                        if not anos_validos.empty:
                            idade_media = date.today().year - anos_validos.mean()
                            idade_media_texto = f"{int(idade_media)} anos"

                    # --- EXIBIÇÃO DASHBOARD PRINCIPAL ---
                    st.subheader(f"📊 Métricas de Mercado no Raio de {int(raio_km * 1000)}m ({len(dados_filtrados)} transações encontradas)")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    def formata_moeda(valor):
                        return f"R$ {valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                    
                    col1.metric("Valor Médio de Venda", formata_moeda(media_absoluta))
                    col2.metric("Média / m² (Terreno)", formata_moeda(media_m2_terreno) if media_m2_terreno > 0 else "-")
                    col3.metric("Média / m² (Construído)", formata_moeda(media_m2_const) if media_m2_const > 0 else "-")
                    col4.metric("Idade Média das Edificações", idade_media_texto)
                    
                    st.markdown("---")
                    
                    # --- SEÇÃO DO MAPA INTERATIVO ---
                    st.subheader("🗺️ Mapa Espacial das Transações")
                    
                    # Cria o mapa centrado no endereço digitado
                    m = folium.Map(location=[lat_c, lon_c], zoom_start=15, control_scale=True)
                    
                    # Desenha o círculo do raio definido na barra lateral
                    folium.Circle(
                        radius=int(raio_km * 1000),
                        location=[lat_c, lon_c],
                        color="blue",
                        fill=True,
                        fill_color="blue",
                        fill_opacity=0.1,
                        popup=f"Raio de {int(raio_km * 1000)}m"
                    ).add_to(m)
                    
                    # Pino vermelho marcando o imóvel alvo digitado
                    folium.Marker(
                        location=[lat_c, lon_c],
                        popup=f"📍 ALVO: {logradouro_alvo}, {numero_alvo}",
                        icon=folium.Icon(color="red", icon="home")
                    ).add_to(m)
                    
                    # Agrupa e plota os pinos azuis de todas as transações históricas encontradas em volta
                    for _, row in dados_filtrados.iterrows():
                        v_formatado = formata_moeda(float(row[coluna_valor])) if pd.notna(row[coluna_valor]) else "N/A"
                        popup_texto = f"""
                        <b>Logradouro:</b> {row['Nome do Logradouro']}, {row['Número']}<br>
                        <b>Bairro:</b> {row['Bairro']}<br>
                        <b>Valor:</b> {v_formatado}<br>
                        <b>Uso:</b> {row['Descrição do uso (IPTU)']}<br>
                        <b>Distância:</b> {int(row['Distancia_KM'] * 1000)} metros
                        """
                        folium.CircleMarker(
                            location=[row['Latitude'], row['Longitude']],
                            radius=6,
                            color="darkblue",
                            fill=True,
                            fill_color="lightblue",
                            fill_opacity=0.8,
                            popup=folium.Popup(popup_texto, max_width=300)
                        ).add_to(m)
                    
                    # Renderiza o mapa na tela do Streamlit
                    folium_static(m, width=1200, height=500)
                    
                    st.markdown("---")
                    
                    # --- TABELA DE DADOS ENXUTA ---
                    st.subheader("📋 Detalhamento das Amostras Encontradas")
                    df_visual = dados_filtrados.copy()
                    
                    # Formata coluna financeira para legibilidade corporativa
                    df_visual[coluna_valor] = df_visual[coluna_valor].apply(
                        lambda v: formata_moeda(float(v)) if pd.notna(v) and str(v) != '-' else v
                    )
                    
                    # Remove a exibição de colunas internas de coordenadas para manter o relatório limpo
                    df_visual = df_visual.drop(columns=['Latitude', 'Longitude'], errors='ignore')
                    
                    # Ordena as transações mais próximas do imóvel primeiro
                    df_visual = df_visual.sort_values(by='Distancia_KM')
                    df_visual['Distância'] = df_visual['Distancia_KM'].apply(lambda d: f"{int(d*1000)}m")
                    df_visual = df_visual.drop(columns=['Distancia_KM'])
                    
                    st.dataframe(df_visual, use_container_width=True)
                    
                else:
                    st.warning("Nenhum imóvel transacionado foi encontrado neste raio com os filtros selecionados.")
                    
            except Exception as e:
                st.error(f"Erro na execução do motor de busca espacial: {e}")
    else:
        st.error("❌ Não conseguimos localizar as coordenadas GPS desse endereço. Verifique a grafia ou o número.")
else:
    st.info("👈 Digite o Nome do Logradouro e o Número na barra lateral para traçar o raio e carregar o Valuation.")
