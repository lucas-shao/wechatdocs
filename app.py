import streamlit as st
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from wechatdocs.callbacks.streamlit import StreamlitCallbackHandler
from wechatdocs.store.store_pdf import store_pdf_to_vector_store
from wechatdocs.search.search import similarity_search


# Sidebar contents
with st.sidebar:
    st.title("Let's Chat With Documents")
    st.markdown(
        """
    
    ## About
    This is an app to chat with docs.

    """
    )


def main():
    st.header("We Chat Docs")

    # upload a PDF file
    pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if pdf_file is not None:
        # store the PDF to a vector store
        VectorStore = store_pdf_to_vector_store(pdf_file)

        # let user input a question
        query = st.text_input("Ask a question about the document:")

        if query:
            similarity_search(VectorStore, query)


if __name__ == "__main__":
    main()
