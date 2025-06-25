import pickle
import streamlit as st
import numpy as np
model=pickle.load(open("linear_regression_model.pkl","rb"))
## lets create a web app
st.title("linera regression model")
tv=st.text_input("enter tv")
radio=st.text_input("enter radio")
newpaper=st.text_input("enter newpaper")
if st.button("predict"):
    features=np.array([[tv,radio,newpaper]],dtype=np.float64)
    result=model.predict(features).reshape(1,-1)
    st.write("predicted sales:::::",result)