import pickle
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback


def store_pdf_to_vector_store(pdf_file: str):
    pdf_reader = PdfReader(pdf_file)
    pdf_text = ""
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    # split the text into trunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(pdf_text)

    # store or query the vector store
    store_name = pdf_file.name[:-4]

    if os.path.exists(f"resource/{store_name}.pkl"):
        with open(f"resource/{store_name}.pkl", "rb") as f:
            VectorStore = pickle.load(f)
            print("Embeddings loaded from Disk")
            return VectorStore
    else:
        # embeddings
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
        with open(f"resource/{store_name}.pkl", "wb") as f:
            pickle.dump(VectorStore, f)
            print("Embeddings saved to Disk")
            return VectorStore
