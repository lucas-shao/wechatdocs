from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import AnalyzeDocumentChain
import sys

llm = OpenAI(
    model_name="gpt-3.5-turbo",
    streaming=True,
)

text_splitter = CharacterTextSplitter()
with open("../resource/contract.txt") as f:
    contract = f.read()

summary_chain = load_summarize_chain(llm, chain_type="map_reduce")
summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)
response = summarize_document_chain.run(contract)
print(response)
