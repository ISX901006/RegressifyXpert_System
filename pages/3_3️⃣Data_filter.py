import streamlit as st
import pandas as pd
import numpy as np
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

if "data_convert" not in st.session_state:
    st.session_state.data_convert = None

if "df_filter" not in st.session_state:
    st.session_state.df_filter = None

df = st.session_state.data_convert


# 显示数据过滤界面
st.header('Data Filter')
if df is not None:
    # 添加滑块用于过滤数值变量
    with st.expander('Numerical Filters', expanded=True):
        numerical_vars = st.session_state.numerical_vars
        num_cols = 3
        if len(numerical_vars) % num_cols == 0:
            num_rows = len(numerical_vars) // num_cols
        else:
            num_rows = len(numerical_vars) // num_cols + 1

        var_ranges = {}
        for row in range(num_rows):
            cols = st.columns(num_cols)
            for col in range(num_cols):
                var_idx = row * num_cols + col
                if var_idx >= len(numerical_vars):
                    break
                var = numerical_vars[var_idx]
                min_val = df[var].min()
                max_val = df[var].max()
                min_val_select, max_val_select = cols[col].slider(f'Select {var} range', min_val, max_val, (min_val, max_val), key=var)
                var_ranges[var] = (min_val_select, max_val_select)

            
    # 添加复选框用于过滤分类变量
    with st.expander('Categorical Filters', expanded=True):
        categorical_vars = st.session_state.categorical_vars
        cat_cols = 2
        if len(categorical_vars) % cat_cols == 0:
            cat_rows = len(categorical_vars) // cat_cols
        else:
            cat_rows = len(categorical_vars) // cat_cols + 1

        selected_values = {}
        for row in range(cat_rows):
            cols = st.columns(cat_cols)
            for col in range(cat_cols):
                var_idx = row * cat_cols + col
                if var_idx >= len(categorical_vars):
                    break
                var = categorical_vars[var_idx]
                unique_values = df[var].unique()
                selected_values[var] = cols[col].multiselect(f'Select {var}', unique_values, default=unique_values, key=var+'_multiselect')

    # 根据过滤条件显示数据
    for var, (min_val, max_val) in var_ranges.items():
        df = df[(df[var] >= min_val) & (df[var] <= max_val)]

    for var in categorical_vars:
        df = df[df[var].isin(selected_values[var])]

    # 显示过滤后的数据
    st.subheader('Show Filtered Data')
    st.write(df)
    st.session_state.df_filter = df

else:
    st.error("Please back to 2_data_visualization page.")



pages = st.container(border=False) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/2_2️⃣data_visualization.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/4_4️⃣model_fitting.py")