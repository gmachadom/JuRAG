# ⚖️ JuRAG - Assistente Jurídico com RAG

JuRAG é um **assistente jurídico** baseado em **RAG (Retrieval-Augmented Generation)** que utiliza **LangChain**, **FAISS** e **Groq LLM** para responder perguntas em linguagem natural sobre a **Constituição Federal**, o **Código de Defesa do Consumidor**, o **Código Civil** e o **Código Penal**.  

Ele permite carregar documentos em PDF, criar uma base vetorial com embeddings, e responder perguntas com fundamentação nos textos jurídicos originais.  

---

## 🚀 Funcionalidades
- 📚 Carrega automaticamente os documentos jurídicos em PDF.  
- ✂️ Quebra os documentos em chunks para melhor recuperação de contexto.  
- 🔎 Indexa os textos usando **FAISS** e embeddings do modelo `all-MiniLM-L6-v2`.  
- 🤖 Usa o modelo **LLaMA-3.3-70B (Groq)** para responder perguntas.  
- 🖥️ Interface simples em **Streamlit**.  
- 📂 Exibe as fontes/documentos consultados para cada resposta.  

---

## 📂 Estrutura do projeto

```
JuRAG
├── data/ # PDFs jurídicos
│ ├── cdc-portugues-2013.pdf
│ ├── cf.pdf
│ ├── codigo_penal_1ed.pdf
│ └── Código Civil 2 ed.pdf
├── app.py
├── requirements.txt
├── .env
├── template.env
├── LICENSE
└── README.md
```

---

## ⚙️ Pré-requisitos
- Python 3.10+  
- Conta no [Groq](https://console.groq.com) e uma **chave de API** válida.  

---

## 🛠️ Instalação e uso

1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-usuario/JuRAG.git
   cd JuRAG

2. Crie um ambiente virtual (opcional, porém recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows

3. Instale as dependências:
   ```bash
   pip install -r requirements.txt

4. Configure sua chave Groq API:
   - Copie o arquivo template.env para .env: 
      ```bash
      cp template.env .env

   - Edite o arquivo .env e insira sua chave: 
      ```
      GROQ_API_KEY=coloque_sua_chave_aqui

5. Execute a aplicação:
   ```bash
   streamlit run app.py

6. Caso o navegador não inicialize automaticamente, acesse:
   ```
   http://localhost:8501

## 📖 Como usar

Clique em "Carregar PDFs" para indexar os documentos.

Digite sua pergunta no campo de texto (exemplo: "Quais são os direitos sociais da Constituição?").

Veja a resposta gerada e consulte as fontes (página e documento PDF).

## 📜 Licença

Este projeto está licenciado sob a MIT License.