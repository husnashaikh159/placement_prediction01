import pickle
import pandas as pd
import numpy as np
import streamlit as st

def prediction(study_hours,attendance,sleep_hours,internet_usage,assignments_completed,previous_score,exam_score,scaler_path,model_path):
    try:
        # load the scaler
        with open(scaler_path,'rb') as file1:
            scaler = pickle.load(file1)
        
        # load the model
        with open(model_path,'rb') as file2:
            model=pickle.load(file2)
        
        # prepare input data
        dct ={
            'study_hours':[study_hours],
            'attendance':[attendance],
            'sleep_hours':[sleep_hours],
            'internet_usage':[internet_usage],
            'assignments_completed':[assignments_completed],
            'previous_score':[previous_score],
            'exam_score':[exam_score]
        }
        x_new= pd.DataFrame(dct)

        #Transform input data
        xnew_pre = scaler.transform(x_new)

        # make predictions
        pred = model.predict(xnew_pre)
        probs = model.predict_proba(xnew_pre)  # model.predict_probs
        max_prob = float(np.max(probs))

        return pred,max_prob
    except Exception as e:
        # log and Display Errors
        st.error(f"Error during Prediction: {str(e)}")
        return None, None 
    
    # Streamlit UI
st.title("🎓 Student Placement Predictor")

# input fields for the features
study_hours = st.number_input(

"Study Hours",

min_value=0,

max_value=24,

value=5

)



attendance = st.number_input(

"Attendance (%)",

min_value=0,

max_value=100,

value=75

)



sleep_hours = st.number_input(

"Sleep Hours",

min_value=0,

max_value=24,

value=7

)



internet_usage = st.number_input(

"Internet Usage (Hours)",

min_value=0,

max_value=24,

value=3

)



assignments_completed = st.number_input(

"Assignments Completed",

min_value=0,

value=5

)



previous_score = st.number_input(

"Previous Score",

min_value=0,

max_value=100,

value=60

)



exam_score = st.number_input(

"Exam Score",

min_value=0,

max_value=100,

value=70

)



# Prediction Button



if st.button("Predict Placement"):
    # file paths
    scaler_path = 'notebook/scaler.pkl'
    model_path = 'notebook/model.pkl'

    # call the predictions functions
    pred, max_prob = prediction(study_hours,attendance,sleep_hours,internet_usage,assignments_completed,previous_score,exam_score,scaler_path,model_path)

    # Display Results
    if pred is not None and max_prob is not None:
        st.subheader(f"Prediction:{pred[0]}")
        st.subheader(f"Prediction Probability:{max_prob:.4f}")   # st.subheader(f"Prediction Probability:{max_prob:.4f}")
        st.progress(max_prob)
    else:
        st.error("Prediction Failed. Check the input values are Correct") 
