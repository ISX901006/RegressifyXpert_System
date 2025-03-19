import streamlit as st
import pandas as pd
import openai

import os
from dotenv import load_dotenv

# 讀取 .env 檔案
load_dotenv()

def chat_gpt():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # openai.api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = OPENAI_API_KEY

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.sidebar.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.sidebar.chat_input("Any questions?")
    if prompt:
        with st.sidebar.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.sidebar.chat_message("assistant"):
            message_placeholder = st.sidebar.empty()   
            full_response = ""
            for response in openai.ChatCompletion.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role":m["role"], "content": m["content"]}
                        for m in st.session_state.messages], stream=True):  
                full_response += response.choices[0].delta.get("content", "")
                message_placeholder.markdown(full_response+"")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

chat_gpt()

if "final_data" not in st.session_state:
    st.session_state.final_data = None
if "bootstrap_results" not in st.session_state:
    st.session_state.bootstrap_results = None
if "wls_mean_function" not in st.session_state:
    st.session_state.wls_mean_function = None

if st.session_state.final_data is not None:
    st.header("Regression Model Result")
    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; The estimated regression function :</div>", unsafe_allow_html=True)
    if st.session_state.bootstrap_results is  None and st.session_state.wls_mean_function is None:
        st.markdown(st.session_state.est_function)
        st.markdown(st.session_state.mean_est_function)
        with st.expander("See explanation"):
            st.write(st.session_state.ols_function_interpre)
        coeff_table = pd.DataFrame(st.session_state.ols_table_coefficients)
    elif st.session_state.bootstrap_results is not None and st.session_state.wls_mean_function is None:
        st.markdown(st.session_state.est_function)
        st.markdown(st.session_state.mean_est_function)
        with st.expander("See explanation"):
            st.write(st.session_state.ols_function_interpre)
        coeff_table = st.session_state.bootstrap_results


    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Regression Statistics :</div>", unsafe_allow_html=True)
    
    ols_col1, ols_col2 = st.columns(2)
    ols_col1.table(pd.DataFrame.from_dict(st.session_state.ols_table1, orient="index", columns=["Value"]))
    ols_col2.table(pd.DataFrame.from_dict(st.session_state.ols_table2, orient="index", columns=["Value"]))
    
    with st.expander("See explanation"):
        st.write("text")

    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Coefficients :</div>", unsafe_allow_html=True)
    
    st.dataframe(coeff_table)
    with st.expander("See explanation"):
        st.write("text")
        


    


else:
    st.error("Please back to model fitting page and select a model.")



# 報表分析
    # analysis of variance
        # test of regression relation p.244
        # coefficients of multiple determination
        # 檢定個別變數

    # estimation and inference of regression parameters
    # 根據常態檢定選擇使用方法 中央極限定理或broostraping

# estimation of mean response

# prediction for new observation

pages = st.container(border=False  ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/5__5️⃣residual_analysis.py")
