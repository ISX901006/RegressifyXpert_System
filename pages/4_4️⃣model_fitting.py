import streamlit as st
import pandas as pd
from collections import defaultdict
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.metrics import PredictionErrorDisplay
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




if "df_filter" not in st.session_state:
    st.session_state.df_filter = None
if "data_convert" not in st.session_state:
    st.session_state.data_convert = None


if st.session_state.df_filter is None:
    df = st.session_state.data_convert
    if df is not None:
        df = df.reset_index(drop=True)
elif st.session_state.df_filter.shape[0]==0:
    df = None
    error_text = "the dataset is empty. Please back to data filter page"
else:
    df = st.session_state.df_filter
    if df is not None:
        df = df.reset_index(drop=True)



# title
st.header("Model Fitting")

if "user_choose_y" not in st.session_state:
    st.session_state.user_choose_y = None
if "user_choose_x_num" not in st.session_state:
    st.session_state.user_choose_x_num = []
if "user_choose_x_cat" not in st.session_state:
    st.session_state.user_choose_x_cat = []

if "dummy_varName" not in st.session_state:
    st.session_state.dummy_varName = []

if "model_dataset" not in st.session_state:
    st.session_state.model_dataset = None

if "est_function" not in st.session_state:
    st.session_state.est_function = ""

if "mean_est_function" not in st.session_state:
    st.session_state.mean_est_function = ""

if "ols_function_interpre" not in st.session_state:
    st.session_state.ols_function_interpre = None

if "final_data" not in st.session_state:
    st.session_state.final_data = None




def user_choose_model_vars(numerical_vars, categorical_vars):
    container_ModelFitting11 = st.container(border=True) 
    with container_ModelFitting11:
        st.write("<div style='padding-bottom: 0.5rem;'>Purpose of Data Analysis：</div>", unsafe_allow_html=True)
        container_ModelFitting1_2, container_ModelFitting1_3, \
        container_ModelFitting1_4, container_ModelFitting1_5 = st.columns([1.4,1,0.23,1.32])

        with container_ModelFitting1_2:
            st.write("Explore the Relationship between")

        with container_ModelFitting1_3:
            model_y = st.selectbox(options=numerical_vars, label="輸入y變數", index=None, placeholder="Y:Selecting a Dependent Variable",label_visibility="collapsed")
        with container_ModelFitting1_4:
            st.write("and")
        if model_y is not None:
            disable_x = False 
            varX = list(set(categorical_vars + numerical_vars) - {model_y})
        else:
            disable_x = True 
            varX = numerical_vars + categorical_vars  
        # 把model x存起來
        with container_ModelFitting1_5: 
            model_x = st.multiselect(options=varX, placeholder="X:Selecting Independent Variables",label="輸入x變數",label_visibility="collapsed",disabled=disable_x)

        return model_y, model_x

def scatter_explain(df):
    df_numeric = df[[st.session_state.user_choose_y]+ st.session_state.user_choose_x_num]
    if st.button("Scatterplot Matrix"):
        with st.spinner('Wait for it...'):
            scatter_matrix = sns.pairplot(df_numeric, diag_kind='kde')
            st.pyplot(scatter_matrix)
        # 解讀散步矩陣圖 描述是否有關係以集關係的強弱
        all_x = ", ".join(st.session_state.user_choose_x_num)
        text = ""
        for var in st.session_state.user_choose_x_num:
            text += f" Observe if there is a linear relationship between `{var}` and `{df_numeric.columns[0]}` . "

        st.markdown("Interpreting Scatterplot Matrix:")
        st.markdown(f"- Please note the linear relationship between the independent and dependent variables.{text} These relationships may suggest that `{all_x}` has a significant impact on predicting `{df_numeric.columns[0]}` ")
        if len(st.session_state.user_choose_x_num) > 1:
            st.markdown(f"- Please note the correlation between independent variables: observe if there is any relationship between the variables in `{all_x}` pair by pair.")
    
    return df_numeric

def handling_dummy(df):   
    need_to_dummy = []
    st.write("<div style='padding-bottom: 0.5rem;'>Dealing with Categorical Variables:</div>", unsafe_allow_html=True)
    for var in st.session_state.user_choose_x_cat:
        categories_level = df.loc[:, var].unique()
        categories_num = len(categories_level)
        if categories_num > 1:
            need_to_dummy.append(var)
            dummyvar = []
            for levels in categories_level[1:]:
                if '_' in levels:
                # 将 '_' 替换为空字符串
                    levels = levels.replace('_', '')
                
                new_var = f"{var}_{{{levels}}}"
                dummyvar.append(new_var)
                dummy_text = f" Converting them into dummy variables: ${'  '.join(dummyvar)}$."
        else:
            dummy_text = f"If there's only one category, the variable is not suitable for inclusion in the model."
                        
        st.write(f"- The number of categories for categorical variable `{var}` is `{categories_num}`, with values {categories_level}.{dummy_text}")
        if categories_num > 1:
            for dummy in dummyvar :
                parts = dummy.split("_")
                st.write(f"which ${dummy}$ is :")
                st.markdown(rf" $ {dummy} = \begin{{cases}} 1 & ,\;\; \text{{if }} {var} = {parts[1]} \\ 0 & ,\;\; \text{{otherwise}} \end{{cases}} $")

            st.session_state.dummy_varName.append(dummyvar)
            

    if len(need_to_dummy) > 0:
        st.session_state.dummy_varName = [item for sublist in st.session_state.dummy_varName for item in sublist]
        # create dummy variables in dataframe
        df_dummy = pd.get_dummies(df, columns=need_to_dummy, drop_first=False)
                    
        #只有dummy variables的dataframe
        only_dummy_df = df_dummy.loc[:, df_dummy.columns.difference(df.columns)]
    else:
        only_dummy_df = None
        need_to_dummy = None

    return only_dummy_df, need_to_dummy

if df is not None:
    
    categorical_vars = st.session_state.categorical_vars
    numerical_vars = st.session_state.numerical_vars
    
    model_y, model_x = user_choose_model_vars(numerical_vars, categorical_vars)
    
    if model_y is not None:
        st.session_state.user_choose_y  =  model_y   
     
    if len(model_x) > 0:
        # show the categorical variables, numerical variables
        numeric_x = list(set(model_x) & set(numerical_vars))
        category_x = list(set(model_x) & set(categorical_vars))
        st.session_state.user_choose_x_num = numeric_x
        st.session_state.user_choose_x_cat = category_x

    if len(st.session_state.user_choose_x_num)>0:
        text_value_num = ", ".join(st.session_state.user_choose_x_num)
    else:
        text_value_num = None
    
    if len(st.session_state.user_choose_x_cat)>0:
        text_value_cat = ", ".join(st.session_state.user_choose_x_cat)
    else:
        text_value_cat = None

    container_ModelFitting12 = st.container(border=True)
    with container_ModelFitting12:
        st.write("<div style='padding-bottom: 0.5rem;'>Selected Variable Categories：</div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.text_area( label="Y-Numerical Variable",value= st.session_state.user_choose_y)
        with col2:
            st.text_area(label="X-Numerical Variables",value=text_value_num)
        with col3:
            st.text_area(label="X-Categorical Variables",value=text_value_cat)

    
    # 確認變數間的關係
    # drawing scatter matrix plot for all selected numeric variables \
        st.write("<div style='padding-bottom: 0.5rem;'>Plot Scatterplot Matrix of Selected Continuous Variables：</div>", unsafe_allow_html=True)
        
        if st.session_state.user_choose_y is not None: #scatter dummy model_data

            st.session_state.dummy_varName = []
            if len(st.session_state.user_choose_x_cat)>0 and len(st.session_state.user_choose_x_num)>0:
                df_numeric = scatter_explain(df)
                df_dummy, need_to_dummy = handling_dummy(df)
                if df_dummy is not None:
                    model_data = pd.concat([df_numeric, df_dummy], axis=1)
                else:
                    model_data = df_numeric

                st.session_state.model_dataset = model_data


            elif len(st.session_state.user_choose_x_num)>0:
                df_numeric = scatter_explain(df)
                model_data = df_numeric
                # # for categorical variables, draw boxplot for each variable and scatter plot for x is order and group by the categorical variables
                # # 繪製箱形圖和散點圖 
                st.session_state.model_dataset = model_data

            elif len(st.session_state.user_choose_x_cat)>0:
                st.info("There is no numeric independent variable.")
                df_dummy, need_to_dummy = handling_dummy(df)
                if df_dummy is not None:
                    df_y = df.loc[:,st.session_state.user_choose_y]
                    model_data = pd.concat([df_y, df_dummy], axis=1)
                    st.session_state.model_dataset = model_data                
                else:
                    st.error("Please select more independent variables to fit the model.") 
                    st.session_state.model_dataset = None 
                
            else:
                st.info("Please select independent variables.")
        
        
        else:
            st.info("Please select variables.")
            
    # build the model  
    if st.session_state.model_dataset is not None:
        st.subheader("Model Selection")
        # var name
        ynam = model_data.columns[0]
        NumName = []
        if st.session_state.dummy_varName != []:
            dummy_flattened_list = st.session_state.dummy_varName
            non_bool_columns = model_data.select_dtypes(exclude='bool').columns.tolist()
            NumName = non_bool_columns[1:]
            xnam = NumName + dummy_flattened_list
            x_square = [f"{var}^2" for var in NumName]
            x_log = [f"log({var})" for var in NumName]
            

        else:
            xnam = model_data.columns[1:].to_list()
            NumName = xnam
            x_square = [f"{var}^2" for var in xnam]
            x_log = [f"log({var})" for var in xnam]
            
         
        x_exp = [f"exp({var})" for var in xnam]
        x_interaction = [] 
        if len(xnam) > 1:
            for i in range(len(xnam)):
                for j in range(i+1, len(xnam)):
                    x_interaction.append(f"{xnam[i]}*{xnam[j]}")


        # choose the model form
        yform_container, xform_container_0, button_all = st.columns([0.3,0.55, 0.15])
        yform_option = [ynam, f"log({ynam})", f"{ynam}^2"]
        if "boxcox_y" not in st.session_state:
            st.session_state.boxcox_y = None
        if st.session_state.boxcox_y is not None:
            yform_option.append(f"boxcox({ynam})")
        
        if "y_form" not in st.session_state:
            st.session_state.y_form = int(0)

        yform = yform_container.radio("Select the dependent variable form", options=yform_option, index=st.session_state.y_form)
        st.session_state.y_form = yform_option.index(yform)

        if "x_first_order_form" not in st.session_state:
            st.session_state.x_first_order_form = []

        if button_all.button("Select All"):
            st.session_state.x_first_order_form = xnam
        st.session_state.x_first_order_form = xform_container_0.multiselect("Select independent variables for the first-order form", options=xnam, default=st.session_state.x_first_order_form, key="x_form1")
        
        
        xform_container_1 ,xform_container_2 =st.columns(2)
        if "x_second_order_form" not in st.session_state:
            st.session_state.x_second_order_form = []
        #use multiselect to select the independent variables of the second-order form
        st.session_state.x_second_order_form = xform_container_1.multiselect("Select independent variables for the second-order form", options=x_square, key="x_form2", default=st.session_state.x_second_order_form)

        if "x_interaction_form" not in st.session_state:
            st.session_state.x_interaction_form = []
        #use multiselect to select the independent variables of the interaction form
        st.session_state.x_interaction_form = xform_container_2.multiselect("Select independent variables for the interaction form", options=x_interaction, key="x_form3",default=st.session_state.x_interaction_form)

        xform_container_3 ,xform_container_4 =st.columns(2)
        if "x_log_form" not in st.session_state:
            st.session_state.x_log_form = []
        #use multiselect to select the independent variables of the nature log form
        st.session_state.x_log_form = xform_container_3.multiselect("Select independent variables for the log form", options=x_log, key="x_form4",default=st.session_state.x_log_form)

        if "x_exp_form" not in st.session_state:
            st.session_state.x_exp_form = []
        #use multiselect to select the independent variables of the exp form
        st.session_state.x_exp_form = xform_container_4.multiselect("Select independent variables for the exp form", options=x_exp, key="x_form5",default=st.session_state.x_exp_form)

        
        xcusform_1 ,xcusform_2 =st.columns([0.4,0.6])
        #use multiselect to select the independent variables of the custom form
        if "x_custom_only_var" not in st.session_state:
            st.session_state.x_custom_only_var = []
        st.session_state.x_custom_only_var = xcusform_1.multiselect("Select independent variables for the custom form", options=NumName,default=st.session_state.x_custom_only_var, key="x_form6")
            
        if "x_custom_order" not in st.session_state:
            st.session_state.x_custom_order = -1.0
        st.session_state.x_custom_order = xcusform_1.number_input("Please input the order of the custom form",min_value=-3.0, max_value=3.0, value=st.session_state.x_custom_order, step=0.1, key="x_form7")
            
        x_custom_order = round(st.session_state.x_custom_order, 1)
        x_custom_option = [f"{var}^{{{x_custom_order}}}" for var in st.session_state.x_custom_only_var]
        
        x_custom_form = xcusform_2.multiselect("Select independent variables for the custom form", options=x_custom_option, default=x_custom_option, key="x_form8")
            
        all_x_form = st.session_state.x_first_order_form + st.session_state.x_second_order_form + st.session_state.x_interaction_form + st.session_state.x_log_form + st.session_state.x_exp_form +x_custom_form
        
        
        if len(all_x_form) > 0:
            #st.subheader("Model Form")
            st.write("The multiple regression equation with an intercept term can be written as:")
            equation_tab = f"$${yform} = β₀ + "
            for idx, var in enumerate(all_x_form, start=1):
                equation_tab += f"β_{{{idx}}} {var} + "
            equation_tab += f"ε $$"
            st.markdown(equation_tab)
            markdown_text = """
             **Assumptions of the error term $\\varepsilon $:**
            1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
            2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
            3. The error term $ \\varepsilon $ is normally distributed.
            4. The error terms are independent of each other.
            """
            st.markdown(markdown_text)

        
        
        # data for model fitting
        fit_container1, fit_container2 = st.columns([0.58,0.42])
        fit_container1.subheader("Use OLS Method to fit the model :")
        
        if fit_container2.button("Run Model"):
            if "boxcox_lambda" not in st.session_state:
                st.session_state.boxcox_lambda = None
            else:
                st.session_state.boxcox_lambda = None

            if st.session_state.y_form == 0 :
                y_data = model_data.iloc[:, 0]
                
            elif st.session_state.y_form == 1 :
                y_data = np.log(model_data.iloc[:, 0])
                
            elif st.session_state.y_form == 2 :
                y_data = model_data.iloc[:, 0]
                y_data = y_data**2
                
            else :
                y_data = pd.Series(st.session_state.boxcox_y, name=yform)
            
            y_data.name = yform
            bool_columns = model_data.select_dtypes(include=bool).columns
            model_data[bool_columns] = model_data[bool_columns].astype(int)
            model_data_to_trans = model_data.copy()
            

            for var in st.session_state.x_first_order_form: 
                var_name = var.replace('{', '').replace('}', '')
                model_data_to_trans[var] = model_data[var_name]
            x_firstOrder_data = model_data_to_trans[st.session_state.x_first_order_form]
            final_data = pd.concat([y_data, x_firstOrder_data], axis=1)
            
            if st.session_state.x_second_order_form != []:
                for var in st.session_state.x_second_order_form:
                    var_name = var.split("^")[0]
                    model_data_to_trans[var] = model_data[var_name]**2
                x_secondOrder_data = model_data_to_trans[st.session_state.x_second_order_form]
                final_data = pd.concat([final_data, x_secondOrder_data], axis=1)

            if st.session_state.x_interaction_form != []:
                for var in st.session_state.x_interaction_form:
                    var_name1 = var.split("*")[0].replace('{', '').replace('}', '')
                    var_name2 = var.split("*")[1].replace('{', '').replace('}', '')
                    model_data_to_trans[var] = model_data[var_name1].multiply(model_data[var_name2])
                x_interaction_data = model_data_to_trans[st.session_state.x_interaction_form]
                final_data = pd.concat([final_data, x_interaction_data], axis=1)
            
            if st.session_state.x_log_form != []:
                for var in st.session_state.x_log_form:
                    var_name = var.split("(")[1].split(")")[0]
                    model_data_to_trans[var] = np.log(model_data[var_name])
                x_log_data = model_data_to_trans[st.session_state.x_log_form]
                final_data = pd.concat([final_data, x_log_data], axis=1)
            
            if st.session_state.x_exp_form != []:
                for var in st.session_state.x_exp_form:
                    var_name = var.split("(")[1].split(")")[0].replace('{', '').replace('}', '')
                    model_data_to_trans[var] = np.exp(model_data[var_name])
                x_exp_data = model_data_to_trans[st.session_state.x_exp_form]
                final_data = pd.concat([final_data, x_exp_data], axis=1)

            if x_custom_form != []:
                for var in x_custom_form:
                    var_name = var.split("^")[0]
                    var_order = st.session_state.x_custom_order
                    model_data_to_trans[var] = model_data[var_name]**var_order
                x_custom_data = model_data_to_trans[x_custom_form]
                final_data = pd.concat([final_data, x_custom_data], axis=1)


            st.session_state.final_data = final_data

        if st.session_state.final_data is not None:
            model_fitting = LinearRegression()
            X = st.session_state.final_data.iloc[:, 1:]
            Y = st.session_state.final_data.iloc[:, 0]
            model_fitting.fit(X, Y)
            beta_sklearn = np.insert(model_fitting.coef_, 0, model_fitting.intercept_)

            Y_varname = Y.name
            X_varname = X.columns

            # show estimated function and interpretation
            equation_est_mean = f"$E({Y_varname})$ = `{round(beta_sklearn[0], 2)}`"
            equation_est = f"${Y_varname}$ = `{round(beta_sklearn[0], 2)}`"
            func = ""
            interpretation = f"- This estimated regression function indicates that ：\n"
            for i, beta in enumerate(beta_sklearn[1:], start=1):
                func += f" + `{round(beta, 2)}`${X_varname[i-1]}$"
                interpretation += f"   - :green[ the mean of ${Y_varname}$] are expected to change by `{beta:.2f}` units when the :green[${X_varname[i-1]}$] increases by 1 unit, holding  other constant\n"

            st.markdown(equation_est_mean+func)
            st.session_state.mean_est_function = equation_est_mean+func
            st.session_state.ols_function_interpre = interpretation
            if st.button("interpretation"):
                st.markdown(interpretation)

            #keep the function to next page
            func += " + $residuals$"
            st.session_state.est_function = equation_est+func

            y_hat = model_fitting.predict(X)
            fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
            PredictionErrorDisplay.from_predictions(
                Y,
                y_pred= y_hat,
                kind="actual_vs_predicted",
                ax=axs[0],
                random_state=0,
            )
            axs[0].set_title("Actual vs. Predicted values")
            PredictionErrorDisplay.from_predictions(
                Y,
                y_pred=y_hat,
                kind="residual_vs_predicted",
                ax=axs[1],
                random_state=0,
            )
            axs[1].set_title("Residuals vs. Predicted Values")
            fig.suptitle("Plotting cross-validated predictions")
            plt.tight_layout()
            st.pyplot(fig)


    
    
    
    # ## tab1 first-order
    # with tab1:
    #     st.markdown("The multiple regression equation with an intercept term can be written as:")
    #     if not model_data.empty:
    #         y_var = model_data.columns[0]
    #         x_vars = model_data.columns[1:]
    #         equation = f"$$ {y_var} = β₀ + "
    #         for idx, var in enumerate(x_vars, start=1):
    #             equation += f"β_{idx} {var} + "
    #         equation += f"ε $$"
    #         st.markdown(equation)
    #         st.markdown(r"-  use least square method or MLE to fit the model : $\beta = (X^T X)^{-1}X^T Y$")
    #         # 創建一個線性回歸模型的實例
    #         model_firstorder = LinearRegression()
    #         # 使用X_matrix和Y_vector來擬合模型
    #         X1 = model_data.iloc[:, 1:]
    #         Y1 = model_data.iloc[:, 0]
    #         model_firstorder.fit(X1, Y1)
    #         # 取得估計的係數
    #         beta_sklearn = np.insert(model_firstorder.coef_, 0, model_firstorder.intercept_)
    #         equation_est = f"${y_var}$ = `{round(beta_sklearn[0], 2)}`"
    #         equation_est_mean = f"$E({y_var})$ = `{round(beta_sklearn[0], 2)}`"
    #         func = ""
    #         interpretation = f"- This estimated regression function indicates that ：\n"
    #         for i, beta in enumerate(beta_sklearn[1:], start=1):
    #             func += f" + `{round(beta, 2)}`${x_vars[i-1]}$"
    #             interpretation += f"   - :red[ the mean of {y_var}] are expected to change by {beta:.2f} units when the {x_vars[i-1]} increases by 1 unit, holding  other constant\n"
    #         #st.markdown( equation_est_mean+func)
    #         #st.markdown(interpretation)
    #         # 解讀係數
    #         #st.markdown("- 解讀係數：")
    #         #st.markdown("   - This estimated regression function indicates that mean y are expected to increase/decrease by beta1 單位")
    #         # when the x1 increases by 1 單位, holding  x2 constant, and that mean y are expected to increase/decrease 
    #         # by beta2 單位 when per x2 increases by 1 單位, holding the x1 constant
    #         func += " + $residuals$"
    #         st.markdown(equation_est+func)


            
    #     else:
    #         markdown_text = """

    # $$
    # Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\ldots + \\beta_n X_n + \\varepsilon
    # $$

    # Where:
    # - $Y$ is the dependent variable,
    # - $\\beta_0 $ is the intercept term,
    # - ${\\beta_1, \\beta_2, \\ldots, \\beta_n }$ are the regression coefficients corresponding to the independent variables ${ X_1, X_2, \ldots, X_n }$ respectively,
    # - ${X_1, X_2, \ldots, X_n }$ are the independent variables,
    # - $\\varepsilon $ is the error term.

    # **Assumptions of the error term $\\varepsilon $:**
    # 1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
    # 2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
    # 3. The error term $ \\varepsilon $ is normally distributed.
    # 4. The error terms are independent of each other.

    # These assumptions are important for making statistical inferences using regression analysis.
    # """  
    #         st.markdown(markdown_text)

    



    # ## tab2 second-order
    # with tab2:
    #     st.markdown("The multiple regression equation with an intercept term can be written as:")
    #     if not model_data.empty:
    #         model_data
    #     else:
    #         markdown_text = """

    # $$
    # Y = \\beta_0 + \\beta_1 X_1 + \\beta_2 X_2 + \\ldots + \\beta_n X_n + \\beta_{n+1} {X_1}^2 + \\beta_{n+2} {X_2}^2 + \\ldots + \\beta_{2n} {X_n}^2 + \\varepsilon
    # $$

    # Where:
    # - $Y$ is the dependent variable,
    # - $\\beta_0 $ is the intercept term,
    # - ${\\beta_1, \\beta_2, \\ldots, \\beta_{2n} }$ are the regression coefficients corresponding to the independent variables $X_1, X_2, \ldots, X_n ,{X_1}^2, {X_2}^2, \ldots, {X_n}^2$ respectively,
    # - $X_1, X_2, \ldots, X_n ,{X_1}^2, {X_2}^2, \ldots, {X_n}^2 $ are the independent variables,
    # - $\\varepsilon $ is the error term.

    # **Assumptions of the error term $\\varepsilon $:**
    # 1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
    # 2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
    # 3. The error term $ \\varepsilon $ is normally distributed.
    # 4. The error terms are independent of each other.

    # These assumptions are important for making statistical inferences using regression analysis.
    # """  
    #         st.markdown(markdown_text)

    
    # ## tab3 first-order with interaction
    # with tab3:
    #     st.markdown("The multiple regression equation with an intercept term can be written as:")
    #     if not model_data.empty:
    #         model_data
    #     else:
    #         markdown_text = """

    # $$
    # Y = \\beta_0 + \\beta_1 X_1 + \\ldots + \\beta_n X_n + \\beta_{n+1} X_1 X_2 + \\beta_{n+2} X_1 X_3 + \\ldots  + \\varepsilon
    # $$

    # Where:
    # - $Y$ is the dependent variable,
    # - $\\beta_0 $ is the intercept term,
    # - ${\\beta_1, \\beta_2, \\ldots, \\beta_n, \\beta_{n+1}, \\beta_{n+2} , \\ldots}$ are the regression coefficients corresponding to the independent variables ${ X_1, X_2, \ldots, X_n, X_1 X_2 , X_1 X_3 , \\ldots}$ respectively,
    # - ${X_1, X_2, \ldots, X_n }$ are the independent variables,
    # - $ X_1 X_2 , X_1 X_3 , \\ldots $ are the interaction terms,
    # - $\\varepsilon $ is the error term.

    # **Assumptions of the error term $\\varepsilon $:**
    # 1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
    # 2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
    # 3. The error term $ \\varepsilon $ is normally distributed.
    # 4. The error terms are independent of each other.

    # These assumptions are important for making statistical inferences using regression analysis.
    # """  
    #         st.markdown(markdown_text)
            

    # ## tab4 second-order with interaction
    # with tab4:
    #     st.markdown("The multiple regression equation with an intercept term can be written as:")
    #     if not model_data.empty:
    #         model_data
    #     else:
    #         markdown_text = """

    # $$
    # Y = \\beta_0 + \\beta_1 X_1 + \\ldots + \\beta_n X_n + \\beta_{n+1} {X_1}^2 + \\ldots + \\beta_{2n} {X_n}^2 + \\beta_{2n+1} X_1 X_2 + \\beta_{2n+2} X_1 X_3 + \\ldots  +\\varepsilon
    # $$

    # Where:
    # - $Y$ is the dependent variable,
    # - $\\beta_0 $ is the intercept term,
    # - ${\\beta_1, \\beta_2, \\ldots, \\beta_{2n+2} \\ldots }$ are the regression coefficients corresponding to the independent variables $X_1 \ldots X_n ,{X_1}^2 \ldots {X_n}^2, X_1 X_2 , X_1 X_3 \\ldots$ respectively,
    # - $X_1 \ldots X_n ,{X_1}^2 \ldots {X_n}^2 $ are the independent variables,
    # - $ X_1 X_2 , X_1 X_3 \\ldots $ are the interaction terms,
    # - $\\varepsilon $ is the error term.

    # **Assumptions of the error term $\\varepsilon $:**
    # 1. The error term $ \\varepsilon $ has a mean of zero, i.e., $ E(\\varepsilon) = 0 $.
    # 2. The error term $\\varepsilon $ has constant variance, i.e., $ Var(\\varepsilon) = \\sigma^2 $.
    # 3. The error term $ \\varepsilon $ is normally distributed.
    # 4. The error terms are independent of each other.

    # These assumptions are important for making statistical inferences using regression analysis.
    # """  
    #         st.markdown(markdown_text)

    
    # ## tab5 text model by user
    # with tab5:
    #     st.markdown("The multiple regression equation with an intercept term can be written as:")
    #     st.text_area( label="請依照格式輸入模型",value= "$Y = beta_0 + beta_1*X_1 + beta_2*X_2 + \\ldots + beta_n*X_n + \\varepsilon$")

else:
    if 'error_text' in locals():
        st.error(error_text)
    else:
        st.error("Please back to 2_data_visualization page.")




pages = st.container(border=False  ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/3_3️⃣data_filter.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/5__5️⃣residual_analysis.py")