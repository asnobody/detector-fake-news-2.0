# Autor: Carlos II

# Estudante de Ciênicas da Computção

# Universidade Mandume Ya Ndemufayo


# 📰 Detector de Fake News

Este projeto é um aplicativo interativo desenvolvido com [Streamlit](https://streamlit.io/) que permite classificar notícias como **Fake** (falsas) ou **Real** (verdadeiras) com base em um modelo de aprendizado de máquina treinado com Gradient Boosting.

## Funcionalidades

- Classificação de notícias como Fake ou Real
- Entrada de texto manual ou via upload de arquivos `.txt`
- Exibição da probabilidade da predição
- Visualização do texto processado
- Interface amigável com Streamlit
- Modelo treinado com validação cruzada e vetorização TF-IDF

## Tecnologias utilizadas

- Python 3.10+
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn
- Streamlit
- Joblib (para salvar e carregar o modelo e o vetorizador)

## Instalação

### 1. Clone o repositório

```bash
git clone https://github.com/seu-usuario/detector-de-fake-news.git
cd detector-de-fake-news
2. Crie um ambiente virtual (opcional, mas recomendado)
bash
Copy
Edit
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate
3. Instale as dependências
bash
Copy
Edit
pip install -r requirements.txt
4. Execute o app Streamlit
bash
Copy
Edit
streamlit run streamlit_app.py

📁 Estrutura do projeto
graphql
Copy
Edit
.
├── modelo_treinado.pkl         # Modelo Gradient Boosting treinado
├── tfidf_vectorizer.pkl        # Vetorizador TF-IDF treinado
├── test_data.pkl               # Dados de teste para avaliação posterior
├── training_history.pkl        # Histórico de treinamento (curvas de aprendizado, AUC, etc.)
├── streamlit_app.py            # Aplicativo principal em Streamlit
├── requirements.txt            # Dependências do projeto
├── README.md                   # Este arquivo
└── data/
    ├── Fake.csv                # Notícias falsas
    └── True.csv                # Notícias reais
🤖 Modelo de Machine Learning
Algoritmo: Gradient Boosting Classifier

Vetorização: TF-IDF

Pré-processamento: remoção de pontuação, URLs, números, stopwords, etc.

Validação: Validação cruzada estratificada (5 folds)

Acurácia média: ~99.5%

Curvas geradas: Confusion Matrix, ROC, Learning Curve

Como usar
Acesse o app via navegador após executar streamlit run.

Escolha o modo de entrada de texto (manual ou arquivo).

Clique em "Classificar Notícia".

Veja o resultado da predição e a probabilidade associada.

Limitações
O modelo foi treinado em um conjunto específico de notícias.

Pode não generalizar perfeitamente para outros contextos (ex: redes sociais).

Verificações humanas e fontes confiáveis continuam essenciais.

Licença
Este projeto é livre para uso educacional e experimental. Para uso comercial, entre em contato com o autor.
