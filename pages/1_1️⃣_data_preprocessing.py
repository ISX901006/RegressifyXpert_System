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

st.header("Data Preprocessing")
st.subheader("Upload a CSV file:")
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None
if 'nrows_raw' not in st.session_state:
    st.session_state.nrows_raw = None
if 'df_deleted' not in st.session_state:
    st.session_state.df_deleted = None
if 'df_changeNA' not in st.session_state:
    st.session_state.df_changeNA = None
if 'df_dropNA' not in st.session_state:
    st.session_state.df_dropNA = None

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if all(df[col].dtype == 'object' for col in df.columns):
        st.error("Please upload a dataset containing numeric variables.")
    else:
        st.session_state.df_raw = df
        st.session_state.nrows_raw = df.shape[0]

df = st.session_state.df_raw

# Retrieve the DataFrame from session state

if df is not None:
        # Show dataframe
        st.write(f"Preview of the uploaded dataset: `{df.shape[0]}`the numbers of rows ")
        st.dataframe(df)

        # Allow user to delete rows and columns
        st.subheader("Delete rows and columns:")

        # Create two columns for options
        col_delete_rows, col_delete_columns = st.columns(2)

        if 'delete_rows' not in st.session_state:
            st.session_state.delete_rows = None

        if 'delete_cols' not in st.session_state:
            st.session_state.delete_cols = None

        # Left column: Delete rows
        with col_delete_rows:
            st.write("Delete rows:")
            st.session_state.delete_rows = st.multiselect("Select rows to delete:", options=df.index.tolist(), default=st.session_state.delete_rows)

        # Right column: Delete columns
        with col_delete_columns:
            st.write("Delete columns:")
            st.session_state.delete_cols = st.multiselect("Select columns to delete:", options=df.columns.tolist(), default=st.session_state.delete_cols)

        left1, middle , right1 = st.columns([0.3,0.4,0.3])
        with middle:
            if st.button("Delete Selected Rows and Columns"):
                if st.session_state.delete_rows or st.session_state.delete_cols:
                    df = df.drop(index=st.session_state.delete_rows, columns=st.session_state.delete_cols)
                    st.session_state.df_deleted = df
                    
                else:
                    st.error("Please select rows or columns to delete.")

        if st.session_state.df_deleted is not None:
            st.write(":green[deleted successfully!] There is filtered data: the sample size is now", st.session_state.df_deleted.shape[0])
            st.dataframe(st.session_state.df_deleted)
            df = st.session_state.df_deleted        
        else:
            df = st.session_state.df_raw

        if 'missing_value'  not in st.session_state:
            st.session_state.missing_value = None
        # Show missing values information
        missing_values = df.isna().sum()
        missing_values_transposed = missing_values.to_frame().T  # Transpose the DataFrame
        st.subheader("Missing values information:")
        st.dataframe(missing_values_transposed)
        if missing_values.sum() == 0:
            st.write(":green[No missing values found in the dataset.]")
        st.session_state.missing_value = missing_values.sum()

        # Additional functionality for handling missing values
        st.subheader("Handle potential missing values:")
        st.write("If missing values are represented by values other than NA, please enter the following information: In the following variables,")

        if 'var_missing_values' not in st.session_state:
            st.session_state.var_missing_values = None

        if 'missing_values_representation' not in st.session_state:
            st.session_state.missing_values_representation = '...'

        # Allow user to select variables with potential missing values
        st.session_state.var_missing_values = st.multiselect("Select variables with potential missing values:", options=df.columns.tolist(),default=st.session_state.var_missing_values)

        # Display text input for missing value representation
        st.session_state.missing_values_representation = st.text_input("Enter missing value representation:", value=st.session_state.missing_values_representation)

        # Show missing value representation
        st.write(f"Missing values are represented as'{st.session_state.missing_values_representation}'")
    

        # Update missing value representation in DataFrame
        if st.button("Update Missing Value Representation"):
            if st.session_state.var_missing_values is not None and st.session_state.missing_values_representation != '...':
                try:
                    missing_value_representation = float(st.session_state.missing_values_representation)
                except ValueError:
                    # 如果無法轉換為整數，則保留為字串型態
                    missing_value_representation = st.session_state.missing_values_representation
                for variable in st.session_state.var_missing_values:
                    df.loc[:, variable].replace(missing_value_representation, np.nan, inplace=True)
                
                st.write("Missing value representation updated successfully!")
                st.session_state.df_changeNA = df
                
            else:
                st.warning("Please select variables with potential missing values.")

        # Show missing values information again after updates
        if st.session_state.df_changeNA is not None:
            missing_values = st.session_state.df_changeNA.isna().sum()
            missing_values_transposed = missing_values.to_frame().T
            st.caption("Missing values information after transformation:")
            st.dataframe(missing_values_transposed)
            #st.dataframe(st.session_state.df_changeNA)
            st.session_state.missing_value = missing_values.sum()

        if st.session_state.missing_value != 0:
        
            st.subheader("Delete rows with missing values")

            if st.session_state.df_changeNA is not None:
                df = st.session_state.df_changeNA
            else:
                if st.session_state.df_deleted is not None:
                    df = st.session_state.df_deleted
                else:
                    df = st.session_state.df_raw


            if st.button("Delete missing values"):
                df.dropna(inplace=True)
                st.write(f"Successfully deleted all rows with missing values！ The number of rows in the dataset is now: :red{df.shape[0]}")
                st.dataframe(df)
                st.session_state.df_dropNA = df
                st.session_state.missing_value = 0


else:
    st.error("Please upload a CSV file.")

import os
print(os.getcwd())



pages = st.container(border=False) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("🏠main_page.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/2_2️⃣data_visualization.py")








