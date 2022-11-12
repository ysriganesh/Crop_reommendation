from typing import List, Any

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
import requests




st.set_page_config(page_title='Crop Prediction')

st.header('CROP RECOMMENDATION SYSTEM WEB APP')
st.write('Welcome to this web app here you will find all the details of recommendation assistant')
st.write('Here this system will recommend the crop is based on the data sets and algorithms and the plantation is based on the farmer choice')



excel_file = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/crop.xlsx'
profile_pic = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-1.png'
profile_pic2 = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-2.png'
profile_pic3 = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-3.png'
profile_pic4 = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-4.png'
profile_pic5 = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-5.png'
profile_pic6 = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-6.png'
profile_pic7 = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-7.png'
profile_pic8 = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-8.png'
profile_pic9 = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-9.png'
profile_pic10 = 'C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/bar-10.png'



st.header('DATA SET USED FOR ANALYSIS OF THe CROPS RECOMMENDATION')
st.write('Here User can see sample of the data set which algorithm is used to recommend')
df = pd.read_excel(excel_file,


                   header=0,
                   nrows=100
                   )

st.dataframe(df)
st.markdown('Figure shows the data set involved')

st.header('VISUALIZATION OF THE DATA SET')

profile_pic = Image.open(profile_pic)
profile_pic2 = Image.open(profile_pic2)
profile_pic3 = Image.open(profile_pic3)
profile_pic4 = Image.open(profile_pic4)
profile_pic5 = Image.open(profile_pic5)
profile_pic6 = Image.open(profile_pic6)
profile_pic7 = Image.open(profile_pic7)
profile_pic8 = Image.open(profile_pic8)
profile_pic9 = Image.open(profile_pic9)
profile_pic10 = Image.open(profile_pic10)


col1,col2 = st.columns(2,gap="small")
with col1:
    st.image(profile_pic, width=350)
with col2:
    st.image(profile_pic2, width=400)

col1,col2 = st.columns(2,gap="small")
with col1:
    st.image(profile_pic3, width=350)
with col2:
    st.image(profile_pic4, width=400)

col1,col2 = st.columns(2,gap="small")
with col1:
    st.image(profile_pic5, width=350)
with col2:
    st.image(profile_pic6, width=400)

col1,col2 = st.columns(2,gap="small")
with col1:
    st.image(profile_pic7, width=350)

col1,col2 = st.columns(2,gap="small")

st.header('ACCURACY AND PERFORMANCE OF THE TRAINED MODEL')
st.image(profile_pic9,width=500)

st.image(profile_pic10, width=500)

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("C:/Users/Y.SRI GANESH REDDY/OneDrive/Desktop/MINI PROJECT/streamlitenv/style/style.css")

with st.container():
    st.write("-----")
    st.header("Any Query Related to Your Farm")
    st.subheader("Fill the form and our expert team will contact you with short time via mail")
    st.write("##")

    contact_form = """
    <form action="https://formsubmit.co/sriganeshreddy.y@gmail.com" method="POST">
    <input type="hidden" name="_captcha" value="false">
     <input type="text" name="name" placeholder="Your Name" required>
     <input type="email" name="email" placeholder="Enter Your Email" required>
     <textarea name="message" placeholder="Enter the query to be asked " required></textarea>
    <button type="submit">Send</button>
</form>
    """


left_column,right_column = st.columns(2)
with left_column:
    st.markdown(contact_form, unsafe_allow_html=True)
with right_column:
    st.empty()







