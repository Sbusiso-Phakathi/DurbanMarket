import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the trained model
model_save_path = "ml_onion_mild_model.pkl"
with open(model_save_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Sidebar with input field descriptions
st.sidebar.header("Description of The Required Input Fields")
st.sidebar.markdown("**Province**: The provinces producing Onion brown.")   
st.sidebar.markdown("**Size_Grade**: The sizes of the brown onion packages.")
st.sidebar.markdown("**Weight_Kg**: The kilogram weight that the onion brown weigh.")
st.sidebar.markdown("**Low_Price**: The lowest price the onion brown cost.")
st.sidebar.markdown("**Sales_Total**: The total price purchase onion brown.")
st.sidebar.markdown("**Stock_On_Hand**: The onion brown stock currently available in the warehouse.")



# Streamlit interface
st.title("Onion Mild Average Price Prediction")
st.image("/Users/da-m1-09/Desktop/sales-hall-6259e56562ec0.jpg",width=600)

# Function to preprocess user inputs and make predictions
def predict_price(Province,Container,Size_Grade,Weight_Kg,Low_Price,Total_Kg_Sold,High_Price,Sales_Total,Stock_On_Hand,month):
    # Assuming label encoding mappings are known
    province_mapping = {'NORTH WEST':2, 'WESTERN CAPE - CERES':7, 'TRANSVAAL':6,'OTHER AREAS':4, 'WESTERN FREESTATE':8, 'NATAL':1,'KWAZULU NATAL':0,
                        'NORTHERN CAPE':3, 'SOURTHEN WESTERN FREESTATE':5} 
   # Replace with actual mappings
    size_grade_mapping = {'1M':1, '2L':6, '1R':2, '1L':0, '1Z':5, '1S':3, '1X':4, '2Z':11, '2R':8, '2M':7, '3Z':14,'4S':16,
       '3Z':15, '3L':12, '2X':10, '3M':13, '2S':9}
    Container_mapping={"AA100":0,"AC030":1,"AF070":2,"AG100":3,"AL200":4}
    # Convert categorical inputs to numerical using label encoding
    province_encoded = province_mapping.get(Province,-1)  # Use -1 for unknown categories
    size_grade_encoded = size_grade_mapping.get(Size_Grade,-1)  # Use -1 for unknown categories
    Container_encoded= Container_mapping.get(Container,-1)

    # Prepare input data as a DataFrame for prediction
    input_data = pd.DataFrame([[province_encoded,Container_encoded,size_grade_encoded,Weight_Kg,Low_Price,Total_Kg_Sold,High_Price,Sales_Total,Stock_On_Hand,month]],
                              columns=[Province,Container,Size_Grade,Weight_Kg,Low_Price,Total_Kg_Sold,High_Price,Sales_Total,Stock_On_Hand,month])
     # Rename columns to string names
     # Make sure the feature names match the model's expectations
    input_data.columns = ['Province','Container','Size_Grade','Weight_Kg','Sales_Total','Low_Price','High_Price','Total_Kg_Sold','month','Stock_On_Hand']

    # Make prediction
    predicted_price = loaded_model.predict(input_data)

    return predicted_price[0]

# Organize input fields into columns
col1, col2, col3 = st.columns(3)

with col1:
    Province = st.selectbox('Province', ['NORTHERN CAPE', 'WESTERN CAPE - CERES', 'WEST COAST','SOUTH WESTERN FREE STATE', 'WESTERN FREESTATE', 'NATAL',
                                    'KWAZULU NATAL', 'OTHER AREAS', 'TRANSVAAL'])
    Size_Grade = st.selectbox("Size Grade", ['1M', '2L', '1R', '1L', '1Z', '1S', '1X', '3L', '2R', '2M', '3S','3Z', '3M', '2Z', '3R', '2S'])
    Total_Kg_Sold = st.number_input('Total Kilos Sold', min_value=0)
    
with col2:
    Container = st.selectbox("Container", ["AA100","AC030","AF070","AG100","AL200"])
    Weight_Kg = st.number_input("Weight Per Kilo", min_value=0.0)
    Stock_On_Hand = st.number_input('Stock On Hand', step=1)
    month = st.slider("Month",1,12)

with col3:
    Low_Price = st.number_input("Low Price", min_value=0)
    High_Price = st.number_input("High Price", min_value=0)
    Sales_Total = st.number_input('Total Sale', min_value=0)
    
    

# Make prediction
if st.button("Predict"):
     # Call the prediction function
    prediction_price=predict_price(Province,Container,Size_Grade,Weight_Kg,Sales_Total,Low_Price,High_Price,Total_Kg_Sold,month,Stock_On_Hand)
    st.success(f'Predicted Average Price of Onion : R{prediction_price:.2f}')