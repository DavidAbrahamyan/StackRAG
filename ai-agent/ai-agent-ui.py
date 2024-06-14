import streamlit as st
from utils import generate_full_response
import tiktoken
import asyncio


MAX_TOKEN_LIMIT = 50
st.title(":orange[ai-agent]")
user_prompt = st.chat_input("Please enter your question")


def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()


loop = get_or_create_eventloop()
asyncio.set_event_loop(loop)


def get_token_count(chat_history: list):
    result_string = ""
    for elem in chat_history:
        result_string = " ".join(elem) + ""
    encoder = tiktoken.encoding_for_model("gpt-4")
    token_count = len(encoder.encode(result_string))
    return token_count


if "user_input_history" not in st.session_state:
    st.session_state["user_input_history"] = []
if "answer_history" not in st.session_state:
    st.session_state["answer_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if user_prompt:
    with st.spinner("Please wait while we generate the answer to your question, this may take a few seconds..."):
        if get_token_count(st.session_state["chat_history"]) > MAX_TOKEN_LIMIT:
            if len(st.session_state["chat_history"]) > 1:
                removed_chat_history = st.session_state["chat_history"].pop(0)

        get_token_count(st.session_state["chat_history"])
        response = generate_full_response(question=user_prompt)

        answer = response
        st.session_state["user_input_history"].append(user_prompt)
        st.session_state["answer_history"].append(answer)
        st.session_state["chat_history"].append((user_prompt, answer))
        get_token_count(st.session_state["chat_history"])


if st.session_state.get("chat_history"):
    for i in range(len(st.session_state["answer_history"])):
        st.chat_message("user").write(st.session_state["user_input_history"][i])
        st.chat_message("assistant").write(st.session_state["answer_history"][i])

