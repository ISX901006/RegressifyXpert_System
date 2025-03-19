import streamlit as st
import pandas as pd
import openai

st.header("Welcome to RegressifyXpert")
st.write("""
    We're dedicated to empowering your data-driven decisions through advanced regression analysis. Whether you're a seasoned analyst or just beginning your journey into data science, RegressifyXpert is here to support you every step of the way.
    """)

# Always show the image
st.image(r"C:\Users\SHI\Desktop\統計專題\analysis.jpg", use_column_width=True)

pages = st.container()

with pages:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col5:
        if st.button("next page ▶️"):
            st.switch_page("pages/1_1️⃣_data_preprocessing.py")
            
st.sidebar.title("Regressitant 🤖")

openai.api_key = st.secrets["OPENAI_API_KEY"]

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    # 容器用於顯示訊息
    messages_container = st.container()

    # 在容器中顯示歷史訊息
    with messages_container:
        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["content"])
    
    # 保持輸入框在側邊欄最下方
    prompt = st.chat_input("Any questions?")
    
    if prompt:
        # 顯示使用者訊息
        with messages_container:
            st.chat_message("user").markdown(prompt)
        
        # 保存使用者訊息
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # 生成 AI 助手回應
        with messages_container:
            message_placeholder = st.empty()
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True
            ):
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response)
            st.chat_message("assistant").markdown(full_response)
        
        # 保存 AI 助手回應
        st.session_state.messages.append({"role": "assistant", "content": full_response})
        
        # 清空輸入框
        st.experimental_rerun()  # 重新加載頁面以清空輸入框




