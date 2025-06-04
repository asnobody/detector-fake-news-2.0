import streamlit as st
import joblib
import re
import string
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Configuração da página
st.set_page_config(page_title="Detector de Fake News", page_icon="📰", layout="wide")

# Função de pré-processamento de texto
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Carregar o modelo e o vetorizador
@st.cache_resource

def load_model():
    model = joblib.load("modelo_treinado.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# Configuração da página
#st.set_page_config(page_title="Detector de Fake News", page_icon="📰", layout="wide")

# Título e descrição
st.title("📰 Detector de Fake News")
st.markdown("""
Este aplicativo utiliza um modelo de aprendizado de máquina para classificar notícias como **Fake** ou **Real**.
O modelo foi treinado usando o algoritmo Gradient Boosting em um conjunto de dados de notícias verdadeiras e falsas.
""")

# Opções de entrada
option = st.radio("Como deseja inserir o texto?", ('Digitar texto', 'Carregar arquivo'))

input_text = ""

if option == 'Digitar texto':
    input_text = st.text_area("Cole o texto da notícia aqui:", height=200)
else:
    uploaded_file = st.file_uploader("Carregue um arquivo de texto (.txt)", type="txt")
    if uploaded_file is not None:
        input_text = uploaded_file.read().decode("utf-8")

# Botão para classificação
if st.button("Classificar Notícia") and input_text:
    # Pré-processamento
    processed_text = preprocess_text(input_text)

    # Vetorização
    text_vec = vectorizer.transform([processed_text])

    # Predição
    prediction = model.predict(text_vec)
    prediction_proba = model.predict_proba(text_vec)

    # Exibir resultados
    st.subheader("Resultado da Classificação")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Classificação", "Fake" if prediction[0] == 0 else "Real")

    with col2:
        fake_prob = prediction_proba[0][0] * 100
        real_prob = prediction_proba[0][1] * 100
        st.metric("Probabilidade", f"{real_prob:.1f}% Real" if prediction[0] == 1 else f"{fake_prob:.1f}% Fake")

    # Gráfico de probabilidades
    prob_df = pd.DataFrame({
        'Categoria': ['Fake', 'Real'],
        'Probabilidade': [fake_prob, real_prob]
    })

    st.bar_chart(prob_df.set_index('Categoria'), use_container_width=True)

    # Exibir texto processado (opcional)
    with st.expander("Ver texto processado (como o modelo vê)"):
        st.text(processed_text)

# Seção sobre o modelo
st.sidebar.title("Sobre o Modelo")
st.sidebar.markdown("""
- **Algoritmo:** Gradient Boosting
- **Acurácia:** ~99.5% (validação cruzada)
- **Pré-processamento:** Remoção de pontuação, URLs, números, etc.
- **Vetorizador:** TF-IDF
""")

st.sidebar.title("Como Usar")
st.sidebar.markdown("""
1. Insira o texto da notícia digitando ou carregando um arquivo
2. Clique no botão "Classificar Notícia"
3. Veja o resultado e a confiança do modelo
""")

st.sidebar.title("Limitações")
st.sidebar.markdown("""
- O modelo foi treinado em um conjunto específico de dados
- Pode não generalizar bem para todos os tipos de notícias
- Sempre verifique informações com fontes confiáveis
""")
