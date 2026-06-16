import streamlit as st
import pandas as pd
import duckdb

# Configurações de layout
st.set_page_config(page_title="Valuation ITBI SP", layout="wide")
st.title("🏢 Ferramenta de Análise Home 2 Invest")

@st.cache_data
def obter_tipos():
    query = """
    SELECT DISTINCT "Descrição do uso (IPTU)" 
    FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true) 
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
if rua:
    with st.spinner(f"A procurar histórico de transações na rua '{rua}'..."):
        
        condicoes = [f"LOWER(\"Nome do Logradouro\") LIKE '%{rua.lower()}%'"]
        
        if tipo_selecionado != "Todos":
            condicoes.append(f"\"Descrição do uso (IPTU)\" = '{tipo_selecionado}'")
            
        if mod_filtro == "Apenas Modernizadas":
            condicoes.append("Modernizada = true")
        elif mod_filtro == "Apenas Antigas":
            condicoes.append("Modernizada = false")
            
        clausula_where = " AND ".join(condicoes)
        
        query_final = f"SELECT * FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true) WHERE {clausula_where}"
        
        try:
            dados_filtrados = duckdb.query(query_final).df()
            
            st.subheader(f"📊 Resultados: {len(dados_filtrados)} transações encontradas")
            
            if len(dados_filtrados) > 0:
                coluna_valor = 'Valor de Transação (declarado pelo contribuinte)'
                v_num = pd.to_numeric(dados_filtrados[coluna_valor], errors='coerce')
                media = v_num.mean()
                st.metric("Preço Médio na Região", f"R$ {media:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))
                
                # --- INÍCIO DAS CORREÇÕES VISUAIS DA TABELA ---
                df_visual = dados_filtrados.copy()
                
                # Correção 1: Ajustar o Bairro (Se estiver 'nan', puxa o dado da coluna Referência)
                df_visual['Bairro'] = df_visual.apply(
                    lambda row: row['Referência'] if str(row['Bairro']).strip().lower() == 'nan' else row['Bairro'], 
                    axis=1
                )
                
                # Limpar a palavra 'nan' que sobrou na coluna Referência
                df_visual['Referência'] = df_visual['Referência'].astype(str).replace('nan', '-')
                
                # Correção 2: Formatar o Valor de Transação no padrão 1.500.000,00
                def formatar_moeda(valor):
                    try:
                        v = float(valor)
                        if pd.isna(v): return "-"
                        # Formata com separador de milhar e decimal do Brasil
                        return f"R$ {v:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                    except:
                        return valor
                        
                df_visual[coluna_valor] = df_visual[coluna_valor].apply(formatar_moeda)
                # --- FIM DAS CORREÇÕES ---
                
                st.dataframe(df_visual, use_container_width=True)
            else:
                st.warning("Nenhuma transação encontrada com estes filtros.")
            
        except Exception as e:
            st.error(f"Erro ao procurar os dados: {e}")
else:
    st.info("👈 Digite o nome de uma rua no menu lateral para começar a análise.")
