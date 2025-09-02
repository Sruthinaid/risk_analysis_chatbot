import streamlit as st
import os
import asyncio
import pandas as pd
import tempfile
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType

# ---- Ensure asyncio loop exists ----
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---- Google API Key ----
api_key = "AIzaSyD2ijWmRFSSyPwTcMOD6zPb0kE514nmozo"

st.title("📄 Supply Chain Risk Analysis ")

uploaded_file = st.file_uploader(
    "Upload a document (PDF, TXT, CSV, DOCX)", type=["pdf", "txt", "csv", "docx"]
)

if uploaded_file:
    file_ext = uploaded_file.name.split(".")[-1].lower()
    documents, df = [], None

    # ---- PDF (OCR fallback) ----
    if file_ext == "pdf":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            pdf_path = tmp.name

        doc = fitz.open(pdf_path)
        text_data = ""
        for page in doc:
            text_data += page.get_text("text")

        if not text_data.strip():
            for page_num in range(len(doc)):
                pix = doc[page_num].get_pixmap(dpi=200)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                text_data += pytesseract.image_to_string(img)

        documents = [Document(page_content=text_data)]

    # ---- TXT ----
    elif file_ext == "txt":
        text_data = uploaded_file.read().decode("utf-8")
        documents = [Document(page_content=text_data)]

    # ---- CSV ----
    elif file_ext == "csv":
        df = pd.read_csv(uploaded_file)

    # ---- DOCX ----
    elif file_ext == "docx":
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(uploaded_file.getbuffer())
            loader = UnstructuredWordDocumentLoader(tmp.name)
            documents = loader.load()

    # ---- Case 1: CSV → AI + Pandas agent ----
    if df is not None:
        st.subheader("💬 Ask Questions About Your CSV")
        question = st.text_input("Enter your question:")

        if question:
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                google_api_key=api_key
            )

            agent = create_pandas_dataframe_agent(
                llm,
                df,
                agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=False,
                allow_dangerous_code=True,
            )

            with st.spinner("Analyzing your data..."):
                try:
                    result = agent.run(question)
                    st.success(result)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

    # ---- Case 2: Other docs → embeddings + QA ----
    elif documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(documents)

        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_key
        )
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        prompt_template = """
        You are a Supply Chain Risk Analysis assistant.
        Use the provided document context to answer the question.
        If the answer is not in the document, say "I couldn't find this in the document."

        Context: {context}
        Question: {question}
        Answer:
        """
        PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)

        qa = RetrievalQA.from_chain_type(
            llm=llm, retriever=retriever, chain_type_kwargs={"prompt": PROMPT}
        )

        st.subheader("💬 Ask Questions About Your Document")
        question = st.text_input("Enter your question:")

        if question:
            with st.spinner("Thinking..."):
                answer = qa.run(question)
            st.write("**Answer:**", answer)

