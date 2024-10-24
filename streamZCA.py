import streamlit as st
import numpy as np
import pickle
import joblib
from  tensorflow.keras.models import load_model

import   streamlit  as st; from PIL import Image; import numpy  as np
import pandas  as pd; import pickle

import os

filename1 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture1.PNG'
filename2 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture2.PNG'

st.title('Predicting Zinc, Cadmium, and Arsenic Levels in European Soils')
with st.container():
    st.image(filename1)
    st.image(filename2)


# Arrange input boxes into three columns for input features
col1, col2, col3 = st.columns(3)

# First row of inputs
with col1:
    FA = st.number_input('Farming Area (FA, ha)', 0.0)
with col2:
    OM = st.number_input('Operating Mine (OM)', 0.0)
with col3:
    RL = st.number_input('Road Length (RL, km)', 0.0)

# Second row of inputs
col4, col5, col6 = st.columns(3)
with col4:
    AR = st.number_input('Annual Rainfall (AR, mm/year)', 0.0)
with col5:
    MT = st.number_input('Mean Temperature (MT, C)', 0.0)
with col6:
    AWC = st.number_input('Available Water Capacity (AWC)', 0.0)

# Third row of inputs for soil physical properties
col7, col8, col9 = st.columns(3)
with col7:
    Clay = st.number_input('Clay Content (%)', 0.0)
with col8:
    Silt = st.number_input('Silt Content (%)', 0.0)
with col9:
    Sand = st.number_input('Sand Content (%)', 0.0)

# Fourth row for soil physical properties
col10, col11, col12 = st.columns(3)
with col10:
    Cfrag = st.number_input('Coarse Fragments (Cfrag, %)', 0.0)
with col11:
    BD = st.number_input('Bulk Density (BD, Tm-3)', 0.0)

# Fifth row for soil chemical properties
col13, col14, col15 = st.columns(3)
with col13:
    Phosphorus = st.number_input('Phosphorus (mg/kg)', 0.0)
with col14:
    Potassium = st.number_input('Potassium (mg/kg)', 0.0)
with col15:
    Nitrogen = st.number_input('Nitrogen (g/kg)', 0.0)

# Sixth row for chemical properties
col16, col17, col18 = st.columns(3)
with col16:
    CEC = st.number_input('Cation Exchange Capacity (CEC)', 0.0)
with col17:
    CaCO3 = st.number_input('Calcium Carbonates (CaCO3, mg/kg)', 0.0)
with col18:
    pH_H2O = st.number_input('pH in H2O', 0.0)

# Final chemical property
col19, col20 = st.columns(2)
with col19:
    CN = st.number_input('Carbon to Nitrogen Ratio (CN)', 0.0)

# Gather all inputs into a list to check how many are zero
input_values = [FA, OM, RL, AR, MT, AWC, Clay, Silt, Sand, Cfrag, BD, Phosphorus,
                Potassium, Nitrogen, CEC, CaCO3, pH_H2O, CN]

# Normalize the input values based on min and max values
FA_tval = (2 * (FA - 0) / (20623310.0 - 0)) - 1
OM_tval = (2 * (OM - 0) / (1.0 - 0)) - 1
RL_tval = (2 * (RL - 0) / (629276.0 - 0)) - 1
AR_tval = (2 * (AR - 0) / (1750.0 - 0)) - 1
MT_tval = (2 * (MT - (-7.5)) / (22.5 - (-7.5))) - 1
AWC_tval = (2 * (AWC - 0) / (0.20 - 0)) - 1
Clay_tval = (2 * (Clay - 0) / (63.0 - 0)) - 1
Silt_tval = (2 * (Silt - 0) / (71.5 - 0)) - 1
Sand_tval = (2 * (Sand - 0) / (83.5 - 0)) - 1
Cfrag_tval = (2 * (Cfrag - 0) / (50.5 - 0)) - 1
BD_tval = (2 * (BD - 0) / (1.505 - 0)) - 1
Phosphorus_tval = (2 * (Phosphorus - 0) / (81.0 - 0)) - 1
Potassium_tval = (2 * (Potassium - 0) / (797.0 - 0)) - 1
Nitrogen_tval = (2 * (Nitrogen - 0) / (10.4 - 0)) - 1
CEC_tval = (2 * (CEC - 0) / (48.5 - 0)) - 1
CaCO3_tval = (2 * (CaCO3 - 0) / (400.0 - 0)) - 1
pH_H2O_tval = (2 * (pH_H2O - 0) / (8.0 - 0)) - 1
CN_tval = (2 * (CN - 0) / (28.015 - 0)) - 1

# Combine all normalized input values into a single array
inputvec = np.array([FA_tval, OM_tval, RL_tval, AR_tval, MT_tval, AWC_tval, Clay_tval, 
                     Silt_tval, Sand_tval, Cfrag_tval, BD_tval, Phosphorus_tval, 
                     Potassium_tval, Nitrogen_tval, CEC_tval, CaCO3_tval, pH_H2O_tval, 
                     CN_tval])


# Check for zeros
zero_count = sum(1 for value in input_values if value == 0)



# Load models and predict the outputs when the button is pressed
if st.button('Run'):

     ## Validation: If more than 5 inputs are zero, show a warning message
    if zero_count > 5:
        st.error(f"Error: More than five input values are zero. Please provide valid inputs for at least 13 features.")
    else:

        ## load model
        # model2 = joblib.load('SSE_RF.pkl')

        # load models from file
        def load_all_models(n_models, Cobj):
            all_models = list()
            for i in range(n_models):
                # define filename for this ensemble
                filenamemod = 'CNNmodel_'+str(i+1)+'.h5'
                ## load model
                loadedmodel = load_model(filenamemod, custom_objects=Cobj)  # , custom_objects=Cobj
                # add to list of members
                all_models.append(loadedmodel)
                # print('>loaded %s' % filename)
                return all_models

        # create stacked model input dataset as outputs from the ensemble
        def stacked_dataset(members, inputX):
            stackX = None
            for model in members:
                # make prediction
                yhat = model.predict(inputX, verbose=0)
                # stack predictions into [rows, members, probabilities]
                if stackX is None:
                    stackX = yhat
                else:
                    stackX = np.dstack((stackX, yhat))

                # flatten predictions to [rows, members x probabilities]
                newstackX = stackX.reshape((stackX.shape[0], stackX.shape[1]*stackX.shape[2]))

                # print("Stacked dataset shape:", stackX.shape)
                # print("Final stacked shape: ", newstackX.shape)
                return newstackX

        def stacked_prediction(members, inputX ):
            # create dataset using ensemble
            # stackedX = stacked_dataset(members, inputX)
            # print(f'Stacked dataset shape: {stackedX.shape}')  # Debugging line
            # print('stacked_prediction is called')  # Debugging line
            # make a prediction
            yhat = members[0].predict(inputX)
            yhatpd = pd.DataFrame(yhat)

            return yhat

        n_members = 5
        Cbij = {'mse': 'mean_squared_error'}
        members = load_all_models(n_members, Cbij)

        input_values = inputvec.reshape(1,inputvec.shape[0],1 ) # Assuming trainx is loaded
        # YY = stacked_prediction(members, model2, input_values)
        YY = stacked_prediction(members,  input_values)


    # Predict Zinc, Cadmium, and Arsenic
    st.write(YY.shape)
    yhat1 = YY[0]
    yhat2 = YY[1]
    yhat3 = YY[2]

    # Convert predictions back to the original scale
    Zinc_real = (yhat1 + 1) * (80.0 - 0.0) * 0.5 + 0.0  # min=0, max=80 for Zinc
    Cadmium_real = (yhat2 + 1) * (1.4 - 0.0) * 0.5 + 0.0  # min=0, max=1.4 for Cadmium
    Arsenic_real = (yhat3 + 1) * (10.0 - 0.0) * 0.5 + 0.0  # min=0, max=10 for Arsenic

    # Display predictions
    st.write("Zinc (mg/kg): ", np.round(Zinc_real, decimals=4))
    st.write("Cadmium (mg/kg): ", np.round(Cadmium_real, decimals=4))
    st.write("Arsenic (mg/kg): ", np.round(Arsenic_real, decimals=4))


filename7 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture3.PNG'
filename8 = 'https://raw.githubusercontent.com/imsb1371/ZCAprediction/refs/heads/main/Capture4.PNG'


with st.container():
    st.header("Developer:")

    st.image(filename8)
 


with st.container():
    st.header("Supervisor:")

    st.image(filename7)