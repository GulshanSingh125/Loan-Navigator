import streamlit as st
import pandas as pd
import pickle as pk

model = pk.load(open('model.pkl','rb'))
scaler = pk.load(open('scaler.pkl','rb'))


st.set_page_config(
    page_title="Loan Navigator", 
    page_icon="🏦"
    )
st.sidebar.title('Loan Approval Prediction System')
app_mode = st.sidebar.selectbox('select page',['Home','Loan Approval'])


if(app_mode=='Home'):
    st.title("🏦 Loan Approval Prediction System")

    from PIL import Image
    img = Image.open('loanpic.jpg') 
    st.image(img)

    st.header('Loan Navigator')


    st.write(
        "Welcome to the Loan Approval Prediction System! Our platform helps you determine the likelihood of your loan approval "
        "based on key financial and personal details. Using machine learning, we analyze your information and provide "
        "an instant prediction. This tool is designed to assist individuals and financial institutions in making informed "
        "loan decisions quickly and efficiently. Simply enter your details, and let our AI-powered model do the rest!"
    )

    st.write("---")

elif(app_mode=='Loan Approval'):
    no_of_dep = st.slider('Choose No of dependents', 0, 5)
    grad = st.selectbox('Choose Education',['Graduated','Not Graduated'])
    self_emp = st.selectbox('Self Emoployed ?',['Yes','No'])
    Annual_Income = st.slider('Choose Annual Income', 0, 10000000)
    Loan_Amount = st.slider('Choose Loan Amount', 0, 10000000)
    Loan_Dur = st.slider('Choose Loan Duration', 0, 20)
    Cibil = st.slider('Choose Cibil Score', 0, 1000)
    Assets = st.slider('Choose Assets', 0, 10000000)



    if grad =='Graduated':
        grad_s =0
    else:
        grad_s = 1

    if self_emp =='No':
        emp_s =0
    else:
        emp_s = 1


    if st.button("Predict"):
        pred_data = pd.DataFrame([[no_of_dep,grad_s,emp_s,Annual_Income,Loan_Amount,Loan_Dur,Cibil,Assets]],
                            columns=['no_of_dependents','education','self_employed','income_annum','loan_amount','loan_term','cibil_score','Assets'])
        pred_data = scaler.transform(pred_data)
        predict = model.predict(pred_data)
        if predict[0] == 1:
            st.snow()
            st.markdown('Loan Is Approved')
        else:
            st.markdown('Loan Is Rejected')
    
        
