from langchain.chains.summarize import load_summarize_chain
from langchain import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.docstore.document import Document

LLM_MAX_TOKENS = 3600


## 将文本做限定字数的总结 summaryTokensLen最好小于200
def summary_text(text: str, summaryTokensLen: int = 100) -> str:
    llm = OpenAI(
        temperature=0,
        model_name="gpt-3.5-turbo",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler()],
        max_tokens=summaryTokensLen,
    )

    if len(text) < (LLM_MAX_TOKENS - summaryTokensLen):
        # 小于LLM MAX Context token数则直接总结
        output_summary = _summary_sub_text(text, llm)
        print("output_summary :::: ")
        print(output_summary)
        return output_summary
    else:
        # 大于LLM MAX Context token则分段总结
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=(LLM_MAX_TOKENS - summaryTokensLen),
            chunk_overlap=20,
            length_function=len,
        )
        texts = text_splitter.split_text(text)
        output_summary = ""
        for t in texts:
            print("t :::: ")
            print(t)
            output_summary = output_summary + _summary_sub_text(t, llm)

        return summary_text(output_summary, summaryTokensLen)


def _summary_sub_text(text: str, llm) -> str:
    docs = [Document(page_content=text)]
    summary_chain = load_summarize_chain(llm, chain_type="stuff", verbose=True)
    output_summary = summary_chain.run(docs)
    return output_summary
