import streamlit as st
import pandas as pd
import duckdb

# Configurações de layout
st.set_page_config(page_title="Valuation ITBI SP", layout="wide")
st.title("🏢 Ferramenta de Análise Home 2 Invest")

# 1. Puxa apenas a lista de tipos de imóveis para o filtro lateral
@st.cache_data
def obter_tipos():
    # O DuckDB lê direto do parquet sem estourar a memória
    query = """
    SELECT DISTINCT "Descrição do uso (IPTU)" 
    FROM 'base_itbi_parte_*.parquet' 
    WHERE "Descrição do uso (IPTU)" IS NOT NULL
    """
    df_tipos = duckdb.query(query).df()
    return ["Todos"] + sorted(df_tipos['Descrição do uso (IPTU)'].tolist())

# --- BARRA LATERAL COM FILTROS ---
st.sidebar.header("Filtros de Busca")
rua = st.sidebar.text_input("Nome da Rua / Logradouro (Obrigatório)")

tipos = obter_tipos()
tipo_selecionado = st.sidebar.selectbox("Tipo de Imóvel", tipos)

mod_filtro = st.sidebar.selectbox("Estado de Conservação", ["Ambos", "Apenas Modernizadas", "Apenas Antigas"])

# --- LÓGICA DE BUSCA OTIMIZADA ---
# O sistema agora aguarda você digitar uma rua para não carregar 2 milhões de linhas à toa
if rua:
    with st.spinner(f"Buscando histórico de transações na rua '{rua}'..."):
        
        # Monta a regra de busca (SQL)
        condicoes = [f"LOWER(\"Nome do Logradouro\") LIKE '%{rua.lower()}%'"]
        
        if tipo_selecionado != "Todos":
            condicoes.append(f"\"Descrição do uso (IPTU)\" = '{tipo_selecionado}'")
            
        if mod_filtro == "Apenas Modernizadas":
            condicoes.append("Modernizada = true")
        elif mod_filtro == "Apenas Antigas":
            condicoes.append("Modernizada = false")
            
        clausula_where = " AND ".join(condicoes)
        
        # O DuckDB vai direto nos arquivos Parquet buscar apenas as linhas que dão "match" com a rua
        query_final = f"SELECT * FROM 'base_itbi_parte_*.parquet' WHERE {clausula_where}"
        
        try:
            dados_filtrados = duckdb.query(query_final).df()
            
            st.subheader(f"📊 Resultados: {len(dados_filtrados)} transações encontradas")
            
            if len(dados_filtrados) > 0:
                coluna_valor = 'Valor de Transação (declarado pelo contribuinte)'
                v_num = pd.to_numeric(dados_filtrados[coluna_valor], errors='coerce')
                media = v_num.mean()
                st.metric("Preço Médio na Região", f"R$ {media:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
            
            # Mostra a tabela apenas com a rua filtrada (super leve)
            st.dataframe(dados_filtrados, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao buscar os dados: {e}")
else:
    st.info("👈 Digite o nome de uma rua no menu lateral para começar a análise.")
