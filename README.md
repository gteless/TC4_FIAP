# Preditor de Obesidade – Painel Analítico e Aplicação Preditiva

Este projeto reúne duas partes complementares:  
1) **Um painel analítico** utilizando dados de hábitos e condições de saúde.  
2) **Uma aplicação preditiva** que utiliza um modelo de Machine Learning treinado para prever o nível de obesidade do usuário com base em informações autocadastradas.

---

## 🎥 Vídeo de Apresentação
Assista ao vídeo completo aqui:  
[Assista ao vídeo completo clicando aqui:](https://drive.google.com/file/d/1fqlRBZK00pw3F55rytOujjdR-GqwMoUL/view)

---

## 🖼️ Interface do Aplicativo (Streamlit)
Assista ao vídeo completo aqui:  
[Veja o app completo clicando aqui:](https://ferramenta-diagnostico-de-obesidade-v4.streamlit.app/)

---
## 🚀 Funcionalidades

### **🔹 Painel Analítico**
- Leitura e tratamento dos dados (CSV)
- Estatísticas descritivas
- Gráficos interativos de distribuição
- Correlação entre variáveis
- Visualização da relação entre hábitos e nível de obesidade

### **🔹 Aplicação Preditiva (Streamlit)**
- Interface amigável para entrada de dados do usuário
- Carregamento de modelo `.joblib`
- Normalização dos dados com `StandardScaler`
- Exibição do resultado preditivo
- Interface simples e responsiva

---

## 📂 Estrutura do Projeto
```
├── app.py # Aplicação Streamlit
├── TC4_v1.ipynb # Notebook responsável pelo treinamento do modelo
├── modelo_obesidade_pipeline_COMPLETO.joblib # Arquivo gerado pelo notebook (após rodar)
└── requirements.txt # Dependências do projeto
```

---

## ⚙️ Como Executar o Projeto

### 🧠 1. Gerar o arquivo .joblib (modelo treinado)

Antes de rodar a aplicação Streamlit, é necessário executar o notebook responsável pelo treinamento do modelo.

Abra o arquivo TC4_v1.ipynb

Execute todas as células

O notebook irá gerar o arquivo model.joblib automaticamente na raiz

⚠️ Caso o arquivo modelo_obesidade_pipeline_COMPLETO.joblib já exista, você pode pular esta etapa.

### 🚀 2. Executar a aplicação Streamlit

Após gerar o arquivo .joblib, execute:
```
streamlit run app.py
```

 Isso abrirá a interface web onde você poderá interagir com o modelo.

## ⚙️ Pré-requisitos

Certifique-se de ter instalado:

- **Python 3.8+**
- **pip** atualizado  
- (Opcional) Ambiente virtual como `venv` ou `conda`

---

## 📦 Instalação das Dependências

Na raiz do projeto, execute:

```
pip install -r requirements.txt
```

## 🛠️ Tecnologias Utilizadas

- Python
- Pandas
- Scikit-learn
- Joblib
- Streamlit
- Jupyter Notebook

