import streamlit as st
import pandas as pd
import duckdb

st.set_page_config(page_title="Valuation ITBI SP", layout="wide")
st.title("🏢 Ferramenta de Análise Home 2 Invest")

@st.cache_data
def obter_filtros_dinamicos():
    query = """
    SELECT DISTINCT "Descrição do uso (IPTU)", Bairro 
    FROM read_parquet('base_itbi_parte_*.parquet', union_by_name=true) 
    """
    df_filtros = duckdb.query(query).df()
    
    tipos = ["Todos"] + sorted([t for t in df_filtros['Descrição do uso (IPTU)'].unique() if t and t != '-'])
    bairros = ["Todos"] + sorted([b for b in df_filtros['Bairro'].unique() if b and b != '-'])
    
    return tipos, bairros

tipos_disp, bairros_disp = obter_filtros_dinamicos()

# --- BARRA LATERAL ---
st.sidebar.header("Filtros de Busca")

rua = st.sidebar.text_input("Nome da Rua / Logradouro")
bairro_selecionado = st.sidebar.selectbox("Bairro", bairros_disp)
tipo_selecionado = st.sidebar.selectbox("Tipo de Imóvel", tipos_disp)
mod_filtro = st.sidebar.selectbox("Estado de Conservação", ["Ambos", "Apenas Modernizadas", "Apenas Antigas"])

# --- LÓGICA DE BUSCA ---
if rua or bairro_selecionado != "Todos":
    with st.spinner("A cruzar dados históricos da região..."):
        
        condicoes = []
        if rua:
            condicoes.append(f"LOWER(\"Nome do Logradouro\") LIKE '%{rua.lower()}%'")
        if bairro_selecionado != "Todos":
            b_safe = bairro_selecionado.replace("'", "''")
            condicoes.append(f"Bairro = '{b_safe}'")
            
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
                col_area_terreno = 'Área do Terreno (m2)'
                col_area_const = 'Área Construída (m2)'
                
                # Conversão segura para números, ignorando textos acidentais da prefeitura
                v_num = pd.to_numeric(dados_filtrados[coluna_valor], errors='coerce')
                a_terreno = pd.to_numeric(dados_filtrados[col_area_terreno], errors='coerce')
                a_const = pd.to_numeric(dados_filtrados[col_area_const], errors='coerce')
                
                # Agrupa em um mini banco de dados para os cálculos de metro quadrado
                df_calc = pd.DataFrame({'Valor': v_num, 'Area_T': a_terreno, 'Area_C': a_const})
                
                media_absoluta = df_calc['Valor'].mean()
                
                # Calcula o valor do m² linha por linha, apenas onde a área é maior que zero
                df_terreno_valido = df_calc[df_calc['Area_T'] > 0]
                media_m2_terreno = (df_terreno_valido['Valor'] / df_terreno_valido['Area_T']).mean() if not df_terreno_valido.empty else 0
                
                df_const_valido = df_calc[df_calc['Area_C'] > 0]
                media_m2_const = (df_const_valido['Valor'] / df_const_valido['Area_C']).mean() if not df_const_valido.empty else 0
                
                # --- EXIBIÇÃO EM 3 COLUNAS VISUAIS ---
                col1, col2, col3 = st.columns(3)
                
                def formata_moeda(valor):
                    return f"R$ {valor:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
                
                col1.metric("Valor Médio (Absoluto)", formata_moeda(media_absoluta))
                
                if media_m2_terreno > 0:
                    col2.metric("Valor Médio / m² (Terreno)", formata_moeda(media_m2_terreno))
                else:
                    col2.metric("Valor Médio / m² (Terreno)", "Sem dados de área")
                    
                if media_m2_const > 0:
                    col3.metric("Valor Médio / m² (Construída)", formata_moeda(media_m2_const))
                else:
                    col3.metric("Valor Médio / m² (Construída)", "Sem dados de área")
                
                st.markdown("---") # Linha divisória
                
                # --- TABELA DE DADOS ---
                df_visual = dados_filtrados.copy()
                
                # Aplica a formatação de Reais na tabela
                def formatar_tabela(valor):
                    try:
                        v = float(valor)
                        if pd.isna(v): return "-"
                        return formata_moeda(v)
                    except:
                        return valor
                        
                df_visual[coluna_valor] = df_visual[coluna_valor].apply(formatar_tabela)
                st.dataframe(df_visual, use_container_width=True)
            else:
                st.warning("Nenhuma transação atende a todos os filtros simultaneamente.")
                
        except Exception as e:
            st.error(f"Erro ao processar consulta: {e}")
else:
    st.info("👈 Digite uma rua ou selecione um bairro no menu lateral para iniciar o Valuation.")
