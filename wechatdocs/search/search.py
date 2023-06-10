from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from wechatdocs.callbacks.streamlit import StreamlitCallbackHandler


def similarity_search(llm, vectorStore, query):
    # search the vector store
    docs = vectorStore.similarity_search(query, k=2)

    # QA
    chain = load_qa_chain(llm, chain_type="stuff")
    with get_openai_callback() as cb:
        response = chain.run(input_documents=docs, question=query)
        return response
        print(cb)
