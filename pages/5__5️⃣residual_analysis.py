import streamlit as st
import pandas as pd
from collections import defaultdict
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import durbin_watson
import math
from scipy.stats import f
from scipy.stats import boxcox
import statsmodels.api as sm
from sklearn.utils import resample
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


def time_series_plot(e):
        fig, ax = plt.subplots()
        ax.plot(e)
        ax.set_xlabel("Observations order")
        ax.set_ylabel("Residuals")
        ax.set_title("Time Series Plot of Residuals")
        st.pyplot(fig)
def normal_probability_plot(e):
        fig, ax = plt.subplots()
        stats.probplot(e, dist="norm", plot=ax)
        ax.set_title("Normal Probability (Q-Q) Plot of Residuals")
        st.pyplot(fig)
def residuals_plot(Y_pred, e):
        fig, ax = plt.subplots()
        ax.scatter(Y_pred, e)
        ax.axhline(y=0, color='red', linestyle='--')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals Plot")
        st.pyplot(fig)
def absolute_residuals_plot(Y_pred, e):
        fig, ax = plt.subplots()
        ax.scatter(Y_pred, np.abs(e))
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel("Absolute Residuals")
        ax.set_title("Absolute Residuals Plot")
        st.pyplot(fig)
def histogram_plot(e):
        mean = np.mean(e)
        std_dev = np.std(e)
        line_x = np.linspace(mean - 3*std_dev, mean + 3*std_dev, 100)
        normal_dist = (1/(std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((line_x - mean) / std_dev) ** 2)
        fig, ax = plt.subplots()
        ax.hist(e, bins=20,density=True)
        ax.plot(line_x, normal_dist, color='red',label='Normal Distribution')
        ax.set_xlabel("Residuals")
        ax.set_ylabel("Density")
        ax.set_title("Histogram of Residuals")
        ax.legend()
        st.pyplot(fig)

def residual_against_x_plot(X,residual_againstX_plot):
    x = X[residual_againstX_plot]
    fig, ax = plt.subplots()
    ax.scatter(x, e)
    ax.axhline(y=0, color='red', linestyle='--')
    xlabel = residual_againstX_plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Residuals")
    title = "Residuals Plot against " + xlabel
    ax.set_title(title)
    st.pyplot(fig)


def theLackOfFit (data,e,alpha):
    X_columns = data.columns[1:].to_list()
    Y_columns = data.columns[0]
    I = data.groupby(X_columns)[Y_columns].mean().reset_index().shape[0]
    k = len(X_columns)+1
    n = data.shape[0]
    mean_Y_by_X = data.groupby(X_columns)[Y_columns].transform('mean')
    data_lack = data.copy()
    data_lack['Y_mean'] = mean_Y_by_X
    data_lack['square_of_error'] = (data_lack[Y_columns] - data_lack['Y_mean']) ** 2
    ssef = data_lack['square_of_error'].sum()
    sser = np.sum(e ** 2)
    f_statistics = ((sser - ssef) / (I - k)) / (ssef / (n - I))
    lackOfFit_pValue = 1.0 - f.cdf(f_statistics, I-k, n-I)
    F_statistic = round(f_statistics,4)
    if lackOfFit_pValue < 0.0001:
        F_pvalue = "<0.0001"
    else:
        F_pvalue = math.floor(lackOfFit_pValue * 10**6) / 10**6
    st.write(f'F statistc: {F_statistic}')
    st.write(f'p-value: {F_pvalue}')
    # if lackOfFit_pValue < alpha:
    #     st.write('Reject the null hypothesis and the residuals are not linear tend.')
    # else:
    #     st.write('Fail to reject the null hypothesis and the residuals are linear tend.')
    return lackOfFit_pValue 



def Brown_Forsythe_Test(Y_pred,e, alpha):    
    # 計算 y_pred 的中位數
    median_y_pred = np.median(Y_pred)

    # 根據 y_pred 的中位數將 e 分成兩組
    #e_group1 = [e[i] for i in range(len(e)) if Y_pred[i] <= median_y_pred]
    e_group1 = [e[i] for i, pred in enumerate(Y_pred) if pred <= median_y_pred]
    e_group2 = [e[i] for i in range(len(e)) if Y_pred[i] > median_y_pred]

    BFresult = stats.levene(e_group1, e_group2, center='median')
    BFresult_statistic = round(BFresult.statistic,4)
    if BFresult.pvalue < 0.0001:
        BFresult_pvalue = "<0.0001"
    else:
        BFresult_pvalue = math.floor(BFresult.pvalue * 10**6) / 10**6
    st.write(f'Test statistc: {BFresult_statistic} ; p-value: {BFresult_pvalue}')
    #st.write(f'Brown-Forsythe Test p-value: {BFresult_pvalue}')
    if BFresult.pvalue < alpha:
        st.write('Reject the null hypothesis and the variances are unequal')
        check_box_heteroscedasticity = True
    else:
        st.write('Fail to reject the null hypothesis and the variances are equal')
        check_box_heteroscedasticity = False
    return check_box_heteroscedasticity

def Breusch_Pagan_test(e,alpha):
     x_copy = X.copy()   
     x_copy.insert(0, 'constant', 1.0)
     BPresult = sms.het_breuschpagan(e, x_copy)
     BPresult_statistic = round(BPresult[2],4)
     if BPresult[3] < 0.0001:
        BPresult_pvalue = "<0.0001"
     else:
        BPresult_pvalue = math.floor(BPresult[3] * 10**6) / 10**6
     st.write(f'F statistc: {BPresult_statistic} ; p-value: {BPresult_pvalue}')
     if BPresult[3] < alpha:
         st.write('Reject the null hypothesis and the variances are unequal')
         check_box_heteroscedasticity = True
     else:
         st.write('Fail to reject the null hypothesis and the variances are equal')
         check_box_heteroscedasticity = False
    
     return check_box_heteroscedasticity

def White_test(e,alpha):
    x_copy = X.copy()   
    x_copy.insert(0, 'constant', 1.0)
    White_result = sms.het_white(e, x_copy)
    White_result_statistic = round(White_result[2],4)
    if White_result[3] < 0.0001:
        White_result_pvalue = "<0.0001"
    else:
        White_result_pvalue = math.floor(White_result[3] * 10**6) / 10**6

    st.write(f'F statistc: {White_result_statistic} ; p-value: {White_result_pvalue}')
    if White_result[3] < alpha:
        st.write('Reject the null hypothesis and the variances are unequal')
        check_box_heteroscedasticity = True
    else:
        st.write('Fail to reject the null hypothesis and the variances are equal')
        check_box_heteroscedasticity = False
    return check_box_heteroscedasticity

def Shapiro_Wilk_Test(e,alpha_normal):
    shapiro_result = stats.shapiro(e)
    shapiro_result_statistic = round(shapiro_result.statistic,4)
    if shapiro_result.pvalue < 0.0001:
        shapiro_result_pvalue = "<0.0001"
    else:
        shapiro_result_pvalue = math.floor(shapiro_result.pvalue * 10**6) / 10**6
    st.write(f'Test statistc: {shapiro_result_statistic} ; p-value: {shapiro_result_pvalue}')
    if shapiro_result.pvalue < alpha_normal:
        st.write('Reject the null hypothesis and the residuals are not normally distributed')
        check_box_NonNormality = True
    else:
        st.write('Fail to reject the null hypothesis and the residuals are normally distributed')
        check_box_NonNormality = False
    return check_box_NonNormality

#perform Kolmogorov-Smirnov test for normality
def Kolmogorov_Smirnov_Test(e,alpha_normal):
    kstest_result = stats.kstest(e, 'norm')
    kstest_result_statistic = round(kstest_result.statistic,4)
    if kstest_result.pvalue < 0.0001:
        kstest_result_pvalue = "<0.0001"
    else:
        kstest_result_pvalue = math.floor(kstest_result.pvalue * 10**6) / 10**6
    st.write(f'Test statistc: {kstest_result_statistic} ; p-value: {kstest_result_pvalue}')
    if kstest_result.pvalue < alpha_normal:
        st.write('Reject the null hypothesis and the residuals are not normally distributed')
        check_box_NonNormality = True
    else:
        st.write('Fail to reject the null hypothesis and the residuals are normally distributed')
        check_box_NonNormality = False
    return check_box_NonNormality

def bootstrap_regression(data, n_bootstrap):
    coefficients = []
    standard_errors = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        X_boot = bootstrap_sample.iloc[:, 1:]
        y_boot = bootstrap_sample.iloc[:, 0]
        model_boot = LinearRegression()
        model_boot.fit(X_boot, y_boot)
        model_boot_coef = np.insert(model_boot.coef_, 0, model_boot.intercept_)
        coefficients.append(model_boot_coef)
        #standard_errors.append(np.sqrt(np.mean((model_boot.predict(X_boot) - y_boot) ** 2)))
    return np.array(coefficients)

def wls_bootstrap_regression(data, n_bootstrap , sd_function_indep):
    coefficients = []
    for _ in range(n_bootstrap):
        bootstrap_sample = resample(data, replace=True, n_samples=len(data))
        X_boot = bootstrap_sample.iloc[:, 1:]
        y_boot = bootstrap_sample.iloc[:, 0]
        model_origin = LinearRegression()
        model_origin.fit(X_boot, y_boot)
        Y_boot_pred = model_origin.predict(X_boot)
        e_boot = y_boot - Y_boot_pred
        e_boot = e_boot.rename('e')

        abs_e_boot = abs(e_boot).values.reshape(-1, 1)
        sd_boot_model = LinearRegression()
        if sd_function_indep == 'fitted value of Y':
            x_sdf = pd.Series(Y_boot_pred, name='Y_pred')
        else:
            x_sdf = X_boot[sd_function_indep]
        x_sdf = x_sdf.values.reshape(-1, 1)
        sd_boot_model.fit(x_sdf, abs_e_boot)
        s_i = sd_boot_model.predict(x_sdf)
        s_i_squared = s_i**2
        w_i = 1/s_i_squared
        W = np.diag(w_i.flatten())
                    
        # 計算 WLS 估計值
        wls_X = sm.add_constant(X_boot)
        wls_X_matrix = wls_X.values
        Y_matrix = y_boot.values
        X_transpose_X = wls_X_matrix.T @ W @ wls_X_matrix
        X_transpose_X_inv = np.linalg.inv(X_transpose_X)
        X_transpose_Y = wls_X_matrix.T @ W @ Y_matrix
        model_boot_coef = X_transpose_X_inv @ X_transpose_Y
        coefficients.append(model_boot_coef)
    return np.array(coefficients)
    

def OLS_sd(X,Y,e):
    # 計算殘差的方差（均方誤差）
    n = len(Y)  # 樣本數量
    p = X.shape[1]  # 自變數數量
    residual_sum_of_squares = np.sum(e**2)
    residual_variance = residual_sum_of_squares / (n - p - 1)

    # 構建設計矩陣 X，並在最前面加上一列全為1的常數項，以考慮截距
    X_with_intercept = np.column_stack([np.ones(X.shape[0]), X])

    # 計算設計矩陣的偽逆（(X^T * X)^(-1)）
    X_with_intercept_T_X_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)

    # 計算係數的標準誤
    standard_errors = np.sqrt(np.diag(residual_variance * X_with_intercept_T_X_inv))

    return standard_errors

def extract_ols_regression_stats(model, data):
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    Dep_var = data.columns[0]
    Model = "OLS Regression"
    Method = "Least Squares"
    Sample_size = data.shape[0]
    R_squared = model.score(X, Y)
    Adj_R_squared = 1 - (1 - R_squared) * (Sample_size - 1) / (Sample_size - X.shape[1] - 1)
    F_statistic = (R_squared / (1 - R_squared)) * ((Sample_size - X.shape[1] - 1) / X.shape[1])
    Prob_F_statistic = 1 - stats.f.cdf(F_statistic, X.shape[1], Sample_size - X.shape[1] - 1)
    regression_stats1 = {
        "Dependent Variable:": Dep_var,
        "Model:": Model,
        "Method:": Method,
        "Sample Size:": Sample_size }
    regression_stats2 = {
        "R-squared:": R_squared,
        "Adjusted R-squared:": Adj_R_squared,
        "F-statistic:": F_statistic,
        "Prob(F-statistic):": Prob_F_statistic}
    return regression_stats1, regression_stats2

def ols_coefficients_stats(beta, beta_sd, data ,coefficients_names):
    t_statistic = beta / beta_sd
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_statistic), data.shape[0] - data.shape[1]))
    upper_bound = beta + 1.96 * beta_sd
    lower_bound = beta - 1.96 * beta_sd
    coefficients_stats = {
        "Variable": coefficients_names,
        "Coefficient": beta.tolist(),
        "Standard Error": beta_sd.tolist(),
        "t-statistic": t_statistic.tolist(),
        "p-value": p_value.tolist(),
        "95% Confidence Interval Lower Bound": lower_bound.tolist(),
        "95% Confidence Interval Upper Bound": upper_bound.tolist()}
    return coefficients_stats
    


if "beta" not in st.session_state:
    st.session_state.beta = None
if "beta_sd" not in st.session_state:
    st.session_state.beta_sd = None
if "ols_table1" not in st.session_state:
    st.session_state.ols_table1 = None
if "ols_table2" not in st.session_state:
    st.session_state.ols_table2 = None
if "ols_table_coefficients" not in st.session_state:
    st.session_state.ols_table_coefficients = None


st.header("Residual Analysis")

if st.session_state.final_data is not None:
    data = st.session_state.final_data
    # bool_columns = data.select_dtypes(include=bool).columns
    # data[bool_columns] = data[bool_columns].astype(int)

    st.subheader("The model is :")
    st.markdown(st.session_state.est_function)
    model = LinearRegression()
    X = data.iloc[:, 1:]
    Y = data.iloc[:, 0]
    model.fit(X, Y)
    Y_pred = model.predict(X)
    e = Y - Y_pred
    e = e.rename('e')
    beta = np.insert(model.coef_, 0, model.intercept_)
    beta_sd = OLS_sd(X,Y,e)
    coefficients_names = ['Intercept'] + X.columns.tolist()

    st.session_state.ols_table1, st.session_state.ols_table2 = extract_ols_regression_stats(model, data)
    st.session_state.ols_table_coefficients = ols_coefficients_stats(beta, beta_sd, data ,coefficients_names)
    
    
    st.subheader("Diagnostics and Formal Tests for the Assumptions")
    #st.write("1. Independence")
    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'>1. &nbsp;&nbsp;Independence</div>", unsafe_allow_html=True)
    col1_1,col1_2 = st.columns(2)
    with col1_1:
        st.write("- Plot Diagnostic for Independence")
        time_series_plot(e)
    with col1_2:
        #st.write("To show the plot that do not violate the assumption of independence.")
        #st.link_button("Standard Time Series Plot", "https://streamlit.io/gallery")
        st.link_button("How to identify time series plot", "https://streamlit.io/gallery")
        st.write("- Test for Independence")
        st.markdown("$$H_0 : $$ There is no correlation among the residuals.")
        st.markdown("$$\;\;\;\;\;H_1 :$$ The residuals are autocorrelated.")
        dw_statistics = round(durbin_watson(e),4)
        st.write("- Durbin-Watson Test")
        st.write(f"Durbin-Watson statistics: {dw_statistics}")
        if dw_statistics < 1.5:
            st.write("It suggests that residuals may be positive serial correlation.")
            check_box_Nonindependence = True
        elif dw_statistics > 2.5:
            st.write("It suggests that residuals may be negative serial correlation.")
            check_box_Nonindependence = True
        else:
            st.write("It suggests that residuals are independent.")
            check_box_Nonindependence = False

    #st.write("2. Linearity")
    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'>2. &nbsp;&nbsp; Linearity</div>", unsafe_allow_html=True)
    col2_1,col2_2 = st.columns(2)
    with col2_1:
        st.write("- Plot Diagnostic for Linearity")
        residuals_plot(Y_pred, e)
    with col2_2:
        st.link_button("How to identify the residual plot", "https://streamlit.io/gallery")
        st.write("- Test for Ascertaining the Linear Function")
        st.markdown("$$\;\;\;\;\;H_0 : linear \;\;tendency$$")
        st.markdown("$$\;\;\;\;\;H_1 : non \;\; linear \;\; tendency$$")
        # f test for lack of fit
        col2_2_1,col2_2_2 = st.columns([0.6,0.4])
        with col2_2_2 :
            alpha = st.number_input("set the alpha value", 0.00, 0.15, 0.05, 0.01,key="alpha_linear")
        with col2_2_1 :
            st.write("- F Test for Lack of Fit")
            lackOfFit_pValue = theLackOfFit(data,e, alpha)
        if lackOfFit_pValue < alpha:
            st.write('Reject the null hypothesis and the residuals are not linear tend.')
            check_box_NonLinearity = True
        else:
            st.write('Fail to reject the null hypothesis and the residuals are linear tend.')
            check_box_NonLinearity = False




    #st.write("3. Equal Variance")
    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'>3. &nbsp;&nbsp; Equal Variance</div>", unsafe_allow_html=True)
    col3_1,col3_2 = st.columns(2)
    with col3_1:
        st.write("- Plot Diagnostic for Constant Error Variance")
        absolute_residuals_plot(Y_pred, e)
    with col3_2:
        st.write("- Test for Constancy of Error Variance")
        st.markdown("$$\;\;\;\;\;H_0 : Var(\epsilon_i)=\sigma^2$$")
        st.markdown(r"$$\;\;\;\;\;H_1 : Var(\epsilon_i)\neq\sigma^2$$")
        col3_2_1,col3_2_2 = st.columns([0.6,0.4])
        col3_2_1.selectbox("Select the test ", ['Brown-Forsythe Test',"Breusch-Pagan test", "White test"], key="var_test")
        with col3_2_2 :
            alpha = st.number_input("set the alpha value", 0.00, 0.15, 0.05, 0.01,key="alpha_variance")
        if st.session_state.var_test == "Brown-Forsythe Test":
            check_box_heteroscedasticity = Brown_Forsythe_Test(Y_pred,e, alpha)
        elif st.session_state.var_test == "Breusch-Pagan test":
            check_box_heteroscedasticity = Breusch_Pagan_test(e,alpha)   
        else:
            check_box_heteroscedasticity = White_test(e,alpha)        
    
    #st.write("4. Normality")
    st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'>4. &nbsp;&nbsp; Normality</div>", unsafe_allow_html=True)
    col4_1,col4_2 = st.columns(2)
    with col4_1:
        st.write("- Plot Diagnostic for Normality")
        tab1, tab2 = st.tabs(["Q-Q Plot", "Histogram"])
        with tab1:
            normal_probability_plot(e)
        with tab2:
            histogram_plot(e)
    with col4_2:
        st.link_button("How to identify the Q-Q plot", "https://streamlit.io/gallery")
        st.write("- Test for Normality")
        st.markdown("$$\;\;\;\;\;H_0 : \epsilon_i \sim Normal$$")
        st.markdown("$$\;\;\;\;\;H_1 : \epsilon_i \sim Non\;Normal$$")
        col4_2_1,col4_2_2 = st.columns([0.6,0.4])
        col4_2_1.selectbox("Select the test ", ["Shapiro-Wilk Test","Kolmogorov-Smirnov Test"], key="noraml_test")
        with col4_2_2 :
            alpha_normal = st.number_input("set the alpha value", 0.00, 0.15, 0.05, 0.01,key="alpha_normal")
        if st.session_state.noraml_test == "Shapiro-Wilk Test":
            check_box_NonNormality = Shapiro_Wilk_Test(e,alpha_normal)
        else:
            check_box_NonNormality = Kolmogorov_Smirnov_Test(e,alpha_normal)
    

    if len(e)<30 :
        st.info("文字提示以殘插圖為主")

    container_residual_test = st.container(border=True) 
    with container_residual_test:
        st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 5px;'> Residual Diagnostic Results : </div>", unsafe_allow_html=True)
        rd_result1, rd_result2, rd_result3, rd_result4 = st.columns(4)
        independence_result = rd_result1.checkbox("NonIndependence of residuals",value= check_box_Nonindependence ,key="check_independence")
        linearity_result = rd_result2.checkbox("NonLinearity of residuals",value= check_box_NonLinearity ,key="check_linearity")
        variance_result = rd_result3.checkbox("Heteroscedasticity of residuals",value= check_box_heteroscedasticity ,key="check_variance")
        normality_result = rd_result4.checkbox("NonNormality of residuals",value= check_box_NonNormality ,key="check_normality")


    if "wls_function" not in st.session_state:
        st.session_state.wls_function = None
    if "wls_mean_function" not in st.session_state:
        st.session_state.wls_mean_function = None
    if "wls_function_interpre" not in st.session_state:
        st.session_state.wls_function_interpre = None
    if "bootstrap_results" not in st.session_state:
        st.session_state.bootstrap_results = None

    sd_option = X.columns.tolist()
    sd_option.insert(0, 'fitted value of Y')
    if "sd_function_indep" not in st.session_state:
        st.session_state.sd_function_indep = None   
        sd_option_default = 0
    # else:
    #     sd_option_default = sd_option.index(st.session_state.sd_function_indep)
    
    if independence_result or linearity_result or variance_result or normality_result:
    
        st.subheader("Remedial Measures")
        if independence_result:
            st.error("Beacuse the error terms are not independent, please go back to the last page to select all correlated variables. Otherwise, please use the Boostrap method to estimate all parameter intervals.")
            if linearity_result:
                st.error("Because the residuals are not linear tend, please go back to the last page to reselect the independent form or add more correlated independent variables. Otherwise, the coefficients estimated by the model will be biased.")
            if variance_result:
                # WLS
                st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Unequal Error Variances Remedial Measures - Weighted Least Squares</div>", unsafe_allow_html=True)
                remedy_container2_1,remedy_container2_2 = st.columns([0.5,0.5])
                remedy_container2_1.link_button("What is Weighted Least Squares ?", "https://streamlit.io/gallery")
                remedy_container2_1.info("Use the residual plot against Xs or Y_hat to check the standard deviation function.")
                remedy_container2_1.write(r"1. standard deviation function : regress $|e|$ against $X$ or $\hat{Y}$")
                remedy_container2_1.write(r"2. the estimated weights : $w_i = \frac{1}{\hat{s_i}^2}$ where $\hat{s_i}$ is fitting value from  standard deviation function")
                remedy_container2_1.latex(r'\hat{\beta} = (X^T W X)^{-1} X^T W y')
                with remedy_container2_2 :
                    residual_againstX_plot = st.selectbox("Select X variable:", options=X.columns, key="residual_againstX",label_visibility='collapsed')
                    residual_against_x_plot(X,residual_againstX_plot)

                remedy_container2_3,remedy_container2_4 = st.columns([0.5,0.5])
                with remedy_container2_3:   
                    sd_function_indep = st.selectbox("Select indep. variable of standard deviation function", options=sd_option, key="sd_function",index = sd_option_default)
                    st.session_state.sd_function_indep = sd_function_indep

                if remedy_container2_4.button("Estimating model coefficients by WLS"):
                    abs_e = abs(e).values.reshape(-1, 1)
                    sd_model = LinearRegression()
                    if sd_function_indep == 'fitted value of Y':
                        x_sdf = pd.Series(Y_pred, name='Y_pred')
                    else:
                        x_sdf = X[sd_function_indep]
                    x_sdf = x_sdf.values.reshape(-1, 1)
                    sd_model.fit(x_sdf, abs_e)
                    s_i = sd_model.predict(x_sdf)
                    s_i_squared = s_i**2
                    w_i = 1/s_i_squared
                    W = np.diag(w_i.flatten())

                    wls_X = sm.add_constant(X)
                    wls_X_matrix = wls_X.values
                    Y_matrix = Y.values
                    X_transpose_X = wls_X_matrix.T @ W @ wls_X_matrix
                    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
                    X_transpose_Y = wls_X_matrix.T @ W @ Y_matrix
                    beta = X_transpose_X_inv @ X_transpose_Y
                    # 計算殘差
                    residuals = Y_matrix - wls_X_matrix @ beta

                    # 計算加權殘差的方差
                    n = len(Y)
                    p = wls_X.shape[1]
                    residual_sum_of_squares = np.sum(w_i.flatten() * residuals**2)
                    residual_variance = residual_sum_of_squares / (n - p)

                    # 計算係數的標準誤
                    beta_sd = np.sqrt(np.diag(residual_variance * X_transpose_X_inv))

                    
                    st.session_state.beta = beta
                    st.session_state.beta_sd = beta_sd

                    # write the estimated function
                    Y_varname = Y.name
                    X_varname = X.columns
                    # show estimated function and interpretation
                    equation_est_mean = f"$E({Y_varname})$ = `{round(beta[0], 2)}`"
                    equation_est = f"${Y_varname}$ = `{round(beta[0], 2)}`"
                    func = ""
                    interpretation = f"- This estimated regression function indicates that ：\n"
                    for i, beta in enumerate(beta[1:], start=1):
                        func += f" + `{round(beta, 2)}`${X_varname[i-1]}$"
                        interpretation += f"   - :green[ the mean of ${Y_varname}$] are expected to change by `{beta:.2f}` units when the :green[${X_varname[i-1]}$] increases by 1 unit, holding  other constant\n"

                    st.session_state.wls_mean_function = equation_est_mean+func
                    
                    st.session_state.wls_function_interpre = interpretation

                    #keep the function to next page
                    func += " + $residuals$"
                    st.session_state.wls_function = equation_est+func

                if st.session_state.wls_mean_function is not None:
                    st.write("The WLS estimated function of the mean response is as follows:")
                    st.markdown(st.session_state.wls_mean_function)

            # broostraping
            st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Boostraping for estimating parameter interval</div>", unsafe_allow_html=True)
            
            st.write(r"The $95\%$ confidence intervals for the coefficients are as follows:")

            if st.button("Show the Boostraping Results"):
                if st.session_state.wls_mean_function is not None and st.session_state.sd_function_indep is not None:
                    bootstrap_coefficients = wls_bootstrap_regression(data, n_bootstrap=1000 , sd_function_indep=st.session_state.sd_function_indep)
                    Estimate_name = "WLS Estimate"
                    Standard_Error = "WLS Standard Error"
                else:
                    bootstrap_coefficients = bootstrap_regression(data, n_bootstrap=1000)
                    Estimate_name = "OLS Estimate"
                    Standard_Error = "OLS Standard Error"
                
                if st.session_state.beta is None:
                    st.session_state.beta = beta
                if st.session_state.beta_sd is None:
                    st.session_state.beta_sd = beta_sd

                st.write(r"The $95\%$ confidence intervals for the coefficients are as follows:")
                confidence_interval_lower = np.percentile(bootstrap_coefficients, 2.5, axis=0)
                confidence_interval_upper = np.percentile(bootstrap_coefficients, 97.5, axis=0)
                
                bootstrap_results = pd.DataFrame({
                    'Coefficient': coefficients_names,
                    Estimate_name : st.session_state.beta,
                    Standard_Error : st.session_state.beta_sd,
                    'Bootstrap Lower Bound': confidence_interval_lower,
                    'Bootstrap Upper Bound': confidence_interval_upper
                })
                st.session_state.bootstrap_results = bootstrap_results
            
            if st.session_state.bootstrap_results is not None:
                st.write(st.session_state.bootstrap_results)


        
        else:
            if linearity_result:
                st.error("Because the residuals are not linear tend, please go back to the last page to reselect the independent form or add more correlated independent variables. Otherwise, the coefficients estimated by the model will be biased.")
            if variance_result or normality_result:
                # boxcox transformation
                remedy_container1_1,remedy_container1_2 = st.columns([0.45,0.55])
                remedy_container1_1.write("<div style='font-size: 1.3rem; font-weight: 600; padding-bottom: 5px;'>&bull; &nbsp;&nbsp; Box-Cox transformation of &nbsp; Y</div>", unsafe_allow_html=True)
                remedy_container1_2.link_button("What is Box-Cox method ?","https://streamlit.io/gallery")

                remedy_container1_3,remedy_container1_4 = st.columns([0.35,0.65])  
                if remedy_container1_3.button("Apply box-cox transformation"):
                    transformed_data, best_lambda = boxcox(Y)
                    st.session_state.boxcox_y = transformed_data
                    st.session_state.boxcox_lambda = best_lambda
                if st.session_state.boxcox_lambda is not None :
                    remedy_container1_4.write(f"the best lambda : {st.session_state.boxcox_lambda}") 
                    st.error("Please go back to last page to reselect the dependent form by BoxCox(Y)")
            if variance_result:
                # WLS
                st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Unequal Error Variances Remedial Measures - Weighted Least Squares</div>", unsafe_allow_html=True)
                remedy_container2_1,remedy_container2_2 = st.columns([0.5,0.5])
                remedy_container2_1.link_button("What is Weighted Least Squares ?", "https://streamlit.io/gallery")
                remedy_container2_1.info("Use the residual plot against Xs or Y_hat to check the standard deviation function.")
                remedy_container2_1.write(r"1. standard deviation function : regress $|e|$ against $X$ or $\hat{Y}$")
                remedy_container2_1.write(r"2. the estimated weights : $w_i = \frac{1}{\hat{s_i}^2}$ where $\hat{s_i}$ is fitting value from  standard deviation function")
                remedy_container2_1.latex(r'\hat{\beta} = (X^T W X)^{-1} X^T W y')
                with remedy_container2_2 :
                    residual_againstX_plot = st.selectbox("Select X variable:", options=X.columns, key="residual_againstX",label_visibility='collapsed')
                    residual_against_x_plot(X,residual_againstX_plot)

                remedy_container2_3,remedy_container2_4 = st.columns([0.5,0.5])
                with remedy_container2_3:   
                    sd_function_indep = st.selectbox("Select indep. variable of standard deviation function", options=sd_option, key="sd_function",index = sd_option_default)
                    st.session_state.sd_function_indep = sd_function_indep
                
                if remedy_container2_4.button("Estimating model coefficients by WLS"):
                    abs_e = abs(e).values.reshape(-1, 1)
                    sd_model = LinearRegression()
                    if sd_function_indep == 'fitted value of Y':
                        x_sdf = pd.Series(Y_pred, name='Y_pred')
                    else:
                        x_sdf = X[sd_function_indep]
                    x_sdf = x_sdf.values.reshape(-1, 1)
                    sd_model.fit(x_sdf, abs_e)
                    s_i = sd_model.predict(x_sdf)
                    s_i_squared = s_i**2
                    w_i = 1/s_i_squared
                    W = np.diag(w_i.flatten())
                    
                    # 計算 WLS 估計值
                    wls_X = sm.add_constant(X)
                    wls_X_matrix = wls_X.values
                    Y_matrix = Y.values
                    X_transpose_X = wls_X_matrix.T @ W @ wls_X_matrix
                    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
                    X_transpose_Y = wls_X_matrix.T @ W @ Y_matrix
                    beta = X_transpose_X_inv @ X_transpose_Y

                    # 計算殘差
                    residuals = Y_matrix - wls_X_matrix @ beta

                    # 計算加權殘差的方差
                    n = len(Y)
                    p = wls_X.shape[1]
                    residual_sum_of_squares = np.sum(w_i.flatten() * residuals**2)
                    residual_variance = residual_sum_of_squares / (n - p)

                    # 計算係數的標準誤
                    beta_sd = np.sqrt(np.diag(residual_variance * X_transpose_X_inv))

                    
                    st.session_state.beta = beta
                    st.session_state.beta_sd = beta_sd

                    # write the estimated function
                    Y_varname = Y.name
                    X_varname = X.columns
                    # show estimated function and interpretation
                    equation_est_mean = f"$E({Y_varname})$ = `{round(beta[0], 2)}`"
                    equation_est = f"${Y_varname}$ = `{round(beta[0], 2)}`"
                    func = ""
                    interpretation = f"- This estimated regression function indicates that ：\n"
                    for i, beta in enumerate(beta[1:], start=1):
                        func += f" + `{round(beta, 2)}`${X_varname[i-1]}$"
                        interpretation += f"   - :green[ the mean of ${Y_varname}$] are expected to change by `{beta:.2f}` units when the :green[${X_varname[i-1]}$] increases by 1 unit, holding  other constant\n"

                    st.session_state.wls_mean_function = equation_est_mean+func
                    
                    st.session_state.wls_function_interpre = interpretation

                    #keep the function to next page
                    func += " + $residuals$"
                    st.session_state.wls_function = equation_est+func

                if st.session_state.wls_mean_function is not None:
                    st.write("The WLS estimated function of the mean response is as follows:")
                    st.markdown(st.session_state.wls_mean_function)



            if normality_result :
                # broostraping
                st.write("<div style='font-size: 1.3rem;;font-weight: 600;padding-bottom: 15px;'>&bull; &nbsp;&nbsp; Boostraping for estimating parameter interval</div>", unsafe_allow_html=True)
                
                st.write(r"The $95\%$ confidence intervals for the coefficients are as follows:")

                if st.button("Show the Boostraping Results"):
                
                    if st.session_state.wls_mean_function is not None and st.session_state.sd_function_indep is not None:
                        bootstrap_coefficients = wls_bootstrap_regression(data, n_bootstrap=1000, sd_function_indep=st.session_state.sd_function_indep)
                        Estimate_name = "WLS Estimate"
                        Standard_Error = "WLS Standard Error"
                    else:
                        bootstrap_coefficients = bootstrap_regression(data, n_bootstrap=1000)
                        Estimate_name = "OLS Estimate"
                        Standard_Error = "OLS Standard Error"


                    if st.session_state.beta is None:
                        st.session_state.beta = beta
                    if st.session_state.beta_sd is None:
                        st.session_state.beta_sd = beta_sd

                    
                    confidence_interval_lower = np.percentile(bootstrap_coefficients, 2.5, axis=0)
                    confidence_interval_upper = np.percentile(bootstrap_coefficients, 97.5, axis=0)
                    
                    bootstrap_results = pd.DataFrame({
                        'Coefficient': coefficients_names,
                        Estimate_name : st.session_state.beta,
                        Standard_Error : st.session_state.beta_sd,
                        'Bootstrap Lower Bound': confidence_interval_lower,
                        'Bootstrap Upper Bound': confidence_interval_upper
                    })

                    st.session_state.bootstrap_results = bootstrap_results

                if st.session_state.bootstrap_results is not None:    
                    st.write(st.session_state.bootstrap_results)
                
            

else:
    st.error("Please back to model fitting page and select a model.")

# analysis of appropriateness of the model

# show residual plot against fitted values, x variables,  x interaction terms 
# A systematic pattern in this plot would suggest that an interaction effect may be present


#plot of the absolute residuals against the fitted values
# use WLS to fit the model
# conclusion - There is no indication of nonconstancy of the error varianCe


#normal probability plot of the residuals. 
# broostraping 
# inference and conclusion 
#The pattern is moderately linear. 
#The coefficient of correlation between the ordered residuals and their expected values 
#under normality is .980. This high value (the interpolated critical value in Table B.6 
#for n = 21 and ex = .05 is .9525) helps to confirm the reasonableness of the conclusion 
#that the error terms are fairly normally distributed.


pages = st.container(border=False  ) 
with pages:
    col1,col2,col3,col4,col5 = st.columns(5)
    with col1:
        if st.button("◀️ last page"):
            st.switch_page("pages/4_4️⃣model_fitting.py")
    with col5:
        if st.button("next page ▶️"): 
            st.switch_page("pages/6_6️⃣model_result.py")

