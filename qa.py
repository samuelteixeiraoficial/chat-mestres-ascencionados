import streamlit as st
from functions import carregar_dados, gerar_resposta

st.set_page_config(
    page_title="Chat com a Sabedoria dos Mestres Ascencionados",
    page_icon="🧘‍♂️",
    layout="centered",
)

st.title("Chat com a Sabedoria dos Mestres Ascencionados")

# Carregar os dados da base (Google Sheets via CSV ou local)
df = carregar_dados()

if df is None or df.empty:
    st.error("Erro ao carregar os dados. Recarregue a página ou tente mais tarde.")
    st.stop()

# Interface do Chat
with st.form("chat_form"):
    pergunta_usuario = st.text_input("Digite sua pergunta:", placeholder="Ex: O que é a onda SHAM?")
    submitted = st.form_submit_button("Perguntar")

if submitted and pergunta_usuario:
    resposta = gerar_resposta(pergunta_usuario, df)
    if resposta:
        st.markdown("**Resposta:**")
        st.write(resposta)
    else:
        st.warning("Nenhuma resposta encontrada para essa pergunta.")

# Rodapé
st.markdown("---")
st.caption("🌟 Baseado nos ensinamentos canalizados por Laércio Fonseca e outros mestres.")
