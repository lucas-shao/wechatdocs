import streamlit as st
import pickle
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

# Sidebar contents
with st.sidebar:
    st.title("LLM Chat With Paper")
    st.markdown(
        """
    
    ## About
    This is an app to chat with paper.

    """
    )


def main():
    st.header("Chat with Paper")

    # upload a PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    # print the text of the PDF
    if pdf_file is not None:
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

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            # st.write("Embeddings loaded from Disk")
        else:
            # embeddings
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embedding=embeddings)
            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            # st.write("Embeddings saved to Disk")

        # let user input a question
        query = st.text_input("Ask a question about the paper:")

        if query:
            # search the vector store
            docs = VectorStore.similarity_search(query, k=2)

            # load the LLM
            llm = OpenAI(model_name="gpt-3.5-turbo")

            # QA
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb)
            st.write(response)


if __name__ == "__main__":
    main()
