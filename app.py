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
    # init session state
    _init_st_session_state()

    st.header("We Chat Docs 📚")

    # upload a PDF file
    pdf_file = st.file_uploader("Upload a PDF file 📄", type=["pdf"])

    if pdf_file is not None:
        if st.session_state["pdf"] != pdf_file:
            st.session_state["pdf"] = pdf_file
            st.session_state["generated"] = []
            st.session_state["past"] = []
            _clear_query()

    if pdf_file is not None:
        # store the PDF to a vector store
        VectorStore = store_pdf_to_vector_store(pdf_file)

        # let user input a question
        # 这里的key=query,映射为st.session_state["query"]的值
        query = st.text_input("Ask a question about the document:", key="query")

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

            if st.session_state["generated"]:
                for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(
                        st.session_state["past"][i], is_user=True, key=str(i) + "_user"
                    )

            # 最后再发起询问
            resp = similarity_search(llm, VectorStore, query)

            st.session_state.past.append(query)
            st.session_state.generated.append(resp)


def _init_st_session_state():
    if "generated" not in st.session_state:
        st.session_state["generated"] = []

    if "past" not in st.session_state:
        st.session_state["past"] = []

    if "pdf" not in st.session_state:
        st.session_state["pdf"] = None

    if "query" not in st.session_state:
        st.session_state["query"] = ""


def _clear_query():
    st.session_state["query"] = ""


if __name__ == "__main__":
    main()
