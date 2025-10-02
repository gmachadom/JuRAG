import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

pdfs_path = r"data"
index_path = r"docs_index"
groq_api_key = os.getenv("GROQ_API_KEY")


# ====================================
# Load the PDF files
# ====================================
def load_pdfs(path: str):
    """Carrega todos os PDFs de uma pasta."""
    loader = DirectoryLoader(
        path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    docs = loader.load()
    return docs

# ====================================
# Break the PDFs in small chunks
# ====================================
def chunkification(docs, chunk_size=1000, overlap=150):
    """Divide os documentos em pedaços menores (chunks)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)

# ====================================
# Vectorization
# ====================================
def create_vector_db(chunks):
    """
    Cria o banco vetorial FAISS a partir dos chunks
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_documents(chunks, embeddings)

def load_vector_db(index_path):
    """
    Carrega o Banco vetorial FAISS para melhorar performance.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ====================================
# Create the chain QA with retriever
# ====================================
def create_chain(vectordb, groq_api_key: str):
    """Cria a chain de QA com RAG."""
    retriever = vectordb.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.3-70b-versatile"
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=True,
    )


# ====================================
# Front-end
# ====================================
def main():
    st.set_page_config(page_title="JuRAG", layout="wide")
    st.title("⚖️ JuRAG - Assistente Jurídico com RAG")

    st.sidebar.header("Configurações")

    if st.sidebar.button("Carregar PDFs"):
        with st.spinner("Carregando documentos..."):
            if os.path.exists(index_path):
                vectordb = load_vector_db(index_path)
                st.session_state.qa_chain = create_chain(vectordb, groq_api_key)
                st.success("✅ Banco vetorial e LLM prontos!")
            else:
                docs = load_pdfs(pdfs_path)
                st.success(f"✅ PDFs carregados: {len(docs)} páginas no total.")

                chunks = chunkification(docs)
                st.info(f"📑 Chunks gerados: {len(chunks)}")

                vectordb = create_vector_db(chunks)
                st.session_state.qa_chain = create_chain(vectordb, groq_api_key)
                st.success("✅ Banco vetorial e LLM prontos!")

    if "qa_chain" in st.session_state:
        st.subheader("💬 Faça uma pergunta:")
        pergunta = st.text_input("Digite sua pergunta:")

        if st.button("Perguntar") and pergunta.strip():
            with st.spinner("Gerando resposta..."):
                result = st.session_state.qa_chain.invoke({"query": pergunta})
                st.write("🔎 Resposta:")
                st.write(result["result"])

                # Mostrar fontes
                with st.expander("📂 Fontes consultadas"):
                    for doc in result["source_documents"]:
                        st.markdown(f"- **Fonte:** {doc.metadata.get("source")}")
                        st.markdown(f"Pág: {doc.metadata.get("page", "N/A")}")


if __name__ == "__main__":
    main()