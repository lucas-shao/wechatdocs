from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import textwrap

llm = OpenAI(
    temperature=0,
    model_name="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0, length_function=len
)
with open("../resource/contract.txt") as f:
    contract = f.read()
texts = text_splitter.split_text(contract)
docs = [Document(page_content=t) for t in texts[:1]]

print(docs[0].page_content)

summary_chain = load_summarize_chain(llm, chain_type="map_reduce", verbose=True)
output_summary = summary_chain.run(docs)
print("output_summary :::: ")
print(output_summary)
