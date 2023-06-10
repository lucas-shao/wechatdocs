import streamlit as st
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from wechatdocs.callbacks.streamlit import StreamlitCallbackHandler
from wechatdocs.store.store_pdf import store_pdf_to_vector_store
from wechatdocs.search.search import similarity_search
from streamlit_chat import message


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
            # load the LLM
            message("I'm thinking...")

            # load the LLM
            # 重点：放在这里初始化LLM，为了能够在答案通过流式返回的时候，直接将内容在这个地方进行展示
            llm = OpenAI(
                model_name="gpt-3.5-turbo",
                streaming=True,
                callbacks=[StreamlitCallbackHandler()],
            )

            # 先将历史问答展示出来
            message(query, is_user=True)

            # 最后再发起询问
            similarity_search(llm, VectorStore, query)


if __name__ == "__main__":
    main()
