import streamlit as st
from streamlit_chat import message
import requests
from langchain.llms import OpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


st.set_page_config(page_title="Streamlit Chat - Demo", page_icon=":robot:")


st.header("Streamlit Chat - Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("You: ", "Hello, how are you?", key="input")
    return input_text


user_input = get_text()

# load the LLM
llm = OpenAI(
    model_name="gpt-3.5-turbo",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

if user_input:
    st.session_state.past.append(user_input)
    st.session_state.generated.append(llm(user_input))

if st.session_state["generated"]:
    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
