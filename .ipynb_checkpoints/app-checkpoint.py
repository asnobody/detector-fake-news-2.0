
import streamlit as st
import pandas as pd
import pickle
import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Detector de Fake News", layout="centered")
st.title("📰 Detector de Fake News")

# Carregamento dos modelos e artefatos
@st.cache_resource
def load_all():
    # Modelo de Gradient Boosting
    with open("modelo_treinado.pkl", "rb") as f:
        gb_model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

    # Modelo de Rede Neural
    nn_model = load_model("rede_neural_model.h5")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    return gb_model, tfidf, nn_model, tokenizer

gb_model, tfidf, nn_model, tokenizer = load_all()

# Seletor de modo
mode = st.radio("Escolha o modo de uso:", ["Entrada Manual", "Upload de CSV"])

# Função de predição
def predict_news(texts, model_name):
    if model_name == "Gradient Boosting":
        X = tfidf.transform(texts)
        preds = gb_model.predict(X)
        probs = gb_model.predict_proba(X)
    else:
        seqs = tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(seqs, maxlen=300)
        probs = nn_model.predict(padded)
        preds = (probs > 0.5).astype(int).flatten()
    return preds, probs

model_option = st.selectbox(
    "Escolha o modelo para análise:",
    ("Gradient Boosting", "Rede Neural")
)

if mode == "Entrada Manual":
    text_input = st.text_area("Digite a notícia que deseja verificar:", height=200)
    if st.button("Verificar"):
        if not text_input.strip():
            st.warning("Por favor, insira o texto da notícia.")
        else:
            pred, prob = predict_news([text_input], model_option)
            pred = pred[0]
            prob = prob[0]
            if pred == 0:
                st.error("🛑 Esta notícia é possivelmente **FALSA**.")
            else:
                st.success("✅ Esta notícia é possivelmente **VERDADEIRA**.")
            st.write("**Confiança do modelo:**", f"{prob[pred]*100:.2f}%")
else:
    uploaded_file = st.file_uploader("Envie um arquivo CSV com uma coluna chamada 'text'", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("O CSV deve conter uma coluna chamada 'text'.")
        else:
            preds, probs = predict_news(df["text"].tolist(), model_option)
            labels = ["FALSA" if p == 0 else "VERDADEIRA" for p in preds]
            confs = [f"{prob[p]*100:.2f}%" for p, prob in zip(preds, probs)]

            df["Classificação"] = labels
            df["Confiança"] = confs

            st.success("Análise concluída!")
            st.dataframe(df[["text", "Classificação", "Confiança"]], use_container_width=True)

            csv_download = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Baixar resultados", csv_download, "resultados_fake_news.csv", "text/csv")
