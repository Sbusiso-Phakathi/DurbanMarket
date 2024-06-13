import streamlit as st
import pickle
import pandas as pd

background_image_url = "https://t3.ftcdn.net/jpg/02/05/56/54/360_F_205565499_waqtbwXp6KukIa7ilco7GfIErIjIK117.jpg"

#Custom CSS
background_css = f"""
<style>
    .stApp {{
        background: url("{background_image_url}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}

    .stTitle {{
        color: #FF6347;  /* Tomato color */
    }}

    .stHeader {{
        color: #4682B4;  /* Steel Blue color */
    }}
</style>
"""
link_url = "https://docs.google.com/presentation/d/15-kLfNXqNHWdRfkZqG7Ta4EVjLv3wx32txovkPFULc4/edit#slide=id.g2dc3f7c3705_0_40"
st.markdown(background_css, unsafe_allow_html=True)

# Load the Onion Brown model
model_path_brown = "price_predict_model.pkl"
with open(model_path_brown, 'rb') as file:
    loaded_model_brown = pickle.load(file)

# Load the Onion Mild model
model_path_mild = "ml_onion_mild_model.pkl"
with open(model_path_mild, 'rb') as file:
    loaded_model_mild = pickle.load(file)

# Load the Market Prediction model
def load_market_model():
    with open('mlr_1.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

# Load the trained model for POTATO(WASHED)MONDIAL
model_save_path = "dbnmarket.pkl"
with open(model_save_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Load the trained model for POTATO SIFRA (WASHED)
model_save_path_sifra = "potato_sifra__model.pkl"
with open(model_save_path_sifra, 'rb') as file:
    loaded_model_sifra = pickle.load(file)

# Function to preprocess user inputs and make predictions for Onion Brown
def predict_price_brown(Province, Size_Grade, Weight_Kg, Low_Price, Sales_Total, Stock_On_Hand, month, day):
    # Replace with actual mappings
    province_mapping = {'NORTHERN CAPE': 2, 'WESTERN CAPE - CERES': 7, 'WEST COAST': 6, 'SOUTH WESTERN FREE STATE': 4, 'WESTERN FREESTATE': 8, 'NATAL': 1, 'KWAZULU NATAL': 0,
                        'OTHER AREAS': 3, 'TRANSVAAL': 5} 
    size_grade_mapping = {'1M': 1, '2L': 6, '1R': 2, '1L': 0, '1Z': 5, '1S': 3, '1X': 4, '3L': 11, '2R': 8, '2M': 7, '3S': 14,
       '3Z': 15, '3M': 12, '2Z': 10, '3R': 13, '2S': 9}
    
    # Encode categorical inputs
    province_encoded = province_mapping.get(Province, -1)  
    size_grade_encoded = size_grade_mapping.get(Size_Grade, -1)  

    # Prepare input data for prediction
    input_data = pd.DataFrame([[province_encoded, size_grade_encoded, Weight_Kg, Low_Price, Sales_Total, Stock_On_Hand, month, day]],
                              columns=['Province', 'Size_Grade', 'Weight_Kg', 'Low_Price', 'Sales_Total', 'Stock_On_Hand', 'month', 'day'])

    # Make prediction
    predicted_price = loaded_model_brown.predict(input_data)

    return predicted_price[0]

# Function to preprocess user inputs and make predictions for Onion Mild
def predict_price_mild(Province,Container,Size_Grade,Weight_Kg,Sales_Total,Low_Price,High_Price,Total_Kg_Sold,month,Stock_On_Hand):
    # Replace with actual mappings
    province_mapping = {'NORTH WEST': 2, 'WESTERN CAPE - CERES': 7, 'TRANSVAAL': 6,'OTHER AREAS': 4, 'WESTERN FREESTATE': 8, 'NATAL': 1,'KWAZULU NATAL': 0,
                        'NORTHERN CAPE': 3, 'SOURTHEN WESTERN FREESTATE': 5} 
    size_grade_mapping = {'1M': 1, '2L': 6, '1R': 2, '1L': 0, '1Z': 5, '1S': 3, '1X': 4, '2Z': 11, '2R': 8, '2M': 7, '3Z': 14,'4S': 16,
       '3Z': 15, '3L': 12, '2X': 10, '3M': 13, '2S': 9}
    container_mapping = {"AA100": 0, "AC030": 1, "AF070": 2, "AG100": 3, "AL200": 4}
    
    # Encode categorical inputs
    province_encoded = province_mapping.get(Province, -1)  
    size_grade_encoded = size_grade_mapping.get(Size_Grade, -1)  
    container_encoded = container_mapping.get(Container, -1)

    # Prepare input data for prediction
    input_data = pd.DataFrame([[province_encoded, container_encoded, size_grade_encoded, Weight_Kg, Low_Price, Total_Kg_Sold, High_Price, Sales_Total, Stock_On_Hand, month]],
                              columns=['Province','Container','Size_Grade','Weight_Kg','Sales_Total','Low_Price','High_Price','Total_Kg_Sold','month','Stock_On_Hand'])

    # Make prediction
    predicted_price = loaded_model_mild.predict(input_data)

    return predicted_price[0]

# Function to preprocess user inputs and make predictions for POTATO(WASHED)MONDIAL
def predict_price(Province,Size_Grade,Weight_Kg,Sales_Total,Low_Price,High_Price,Total_Kg_Sold,Stock_On_Hand):
    # Assuming label encoding mappings are known
    province_mapping = {'NORTHERN CAPE': 5, 'WESTERN CAPE - CERES': 11, 'WEST COAST': 10, 'SOUTH WESTERN FREE STATE': 7, 'WESTERN FREESTATE': 12,
                        'KWAZULU NATAL': 1, 'OTHER AREAS': 6, 'TRANSVAAL': 9, 'EASTERN FREESTATE': 0, 'MPUMALANGA': 2,
                        'NORTH EASTERN CAPE': 3, 'NORTH WEST': 4, 'SOUTHERN CAPE': 8}
    # Replace with actual mappings
    size_grade_mapping = {'1L': 0, '1M': 1, '1R': 2, '1S': 3, '1U': 4, '1X': 5, '1Z': 6, '2L': 7, '2M': 8, '2S': 9, '2U': 10, '2X': 11, '2Z': 12,
                          '3L': 13, '3M': 14, '3R': 15, '3S': 16, '3U': 17, '3X': 18, '3X': 19, '3Z': 20, '4L': 21, '4M': 22, '4R': 23, '4S': 24, '4U': 25,
                          '4Z': 26}
    # Convert categorical inputs to numerical using label encoding
    province_encoded = province_mapping.get(Province, -1)  # Use -1 for unknown categories
    size_grade_encoded = size_grade_mapping.get(Size_Grade, -1)  # Use -1 for unknown categories

    # Prepare input data as a DataFrame for prediction
    input_data = pd.DataFrame([[province_encoded,size_grade_encoded,Weight_Kg,Sales_Total,Low_Price,High_Price,Total_Kg_Sold,Stock_On_Hand]])
     # Rename columns to string names
     # Make sure the feature names match the model's expectations
    input_data.columns = ['Province','Size_Grade','Weight_Kg','Sales_Total','Low_Price','High_Price','Total_Kg_Sold','Stock_On_Hand']

    # Make prediction
    predicted_price = loaded_model.predict(input_data)

    return predicted_price[0]

# Function to preprocess user inputs and make predictions for POTATO SIFRA (WASHED)
def predict_price_sifra(Province, Size_Grade, Weight_Kg, Low_Price, Sales_Total, Stock_On_Hand, month, day):
    # Replace with actual mappings
    province_mapping = {'NORTHERN CAPE': 2, 'WESTERN CAPE - CERES': 7, 'WEST COAST': 6, 'SOUTH WESTERN FREE STATE': 4, 'WESTERN FREESTATE': 8, 'NATAL': 1, 'KWAZULU NATAL': 0,
                        'OTHER AREAS': 3, 'TRANSVAAL': 5}
    size_grade_mapping = {'1R': 2, '1M': 1, '1Z': 5, '2L': 6, '2Z': 11, '3M': 13, '1S': 3, '3R': 14, '2M': 7, '1U': 4, '3Z': 17,
       '1L': 0, '2S': 9, '2R': 8, '4Z': 21, '3L': 12, '3U': 16, '4M': 19, '4L': 18, '3S': 15, '2U': 10, '4R': 20}

    # Encode categorical inputs
    province_encoded = province_mapping.get(Province, -1)
    size_grade_encoded = size_grade_mapping.get(Size_Grade, -1)

    # Prepare input data for prediction
    input_data = pd.DataFrame([[province_encoded, size_grade_encoded, Weight_Kg, Low_Price, Sales_Total, Stock_On_Hand, month, day]],
                              columns=['Province', 'Size_Grade', 'Weight_Kg', 'Low_Price', 'Sales_Total', 'Stock_On_Hand', 'month', 'day'])

    # Make prediction
    predicted_price = loaded_model_sifra.predict(input_data)

    return predicted_price[0]

# Main function to run the app
def main():
    st.title('Commodity Average Price Per Kg Prediction')

    # Sidebar for selecting model
    selected_model = st.sidebar.selectbox("Select Commodity", ["Onion Brown", "Onion Mild", "Tomato Long Life", "Potato Washed Mondial", "Potato SIFRA (WASHED)"])
    if st.sidebar.button("Click here, For More Info About The Analysis"):
        # st.sidebar.markdown(f'<a href="{link_url}" target="_blank"><button>Visit Market_Analysis.com</button></a>', unsafe_allow_html=True)
        st.sidebar.markdown(f'<a href="{link_url}" target="_blank"><button style="background-color: #4CAF50; color: white; padding: 10px 24px; cursor: pointer; border: none; border-radius: 4px;">Visit Market_Analysis.com</button></a>', unsafe_allow_html=True)
    if selected_model == "Onion Brown":
        # Display input fields for Onion Brown
        st.header("Onion Brown Features")
    
        col1, col2 = st.columns(2)
        with col1:
            Province_brown = st.selectbox('Province', ['NORTHERN CAPE', 'WESTERN CAPE - CERES', 'WEST COAST','SOUTH WESTERN FREE STATE', 'WESTERN FREESTATE', 'NATAL',
                                        'KWAZULU NATAL', 'OTHER AREAS', 'TRANSVAAL'])
            Size_Grade_brown = st.selectbox("Size Grade", ['1M', '2L', '1R', '1L', '1Z', '1S', '1X', '3L', '2R', '2M', '3S','3Z', '3M', '2Z', '3R', '2S'])
            Weight_Kg_brown = st.selectbox("Weight Per Kilograms", [1,2,3,5,7,10,20])
            if Province_brown == "WEST COAST" and Size_Grade_brown == "1M" and Weight_Kg_brown == 1:
                Low_Price_brown = 7.535714286
            elif Province_brown == "WESTERN CAPE - CERES" and Size_Grade_brown in("1M", "1R","1S","1Z") and Weight_Kg_brown == 1:
                Low_Price_brown = 6,657142857
        with col2:    
            Sales_Total_brown = st.number_input('Total Sales(R)', min_value=0)
            Stock_On_Hand_brown = st.number_input('Stock On Hand', step=1)
            month_brown = st.slider("Month", 1, 12)
            day_brown = st.slider("Day", 1, 31)

        # Button to predict Onion Brown price
        if st.button("Predict Onion Brown"):
            predicted_price_brown = predict_price_brown(Province_brown, Size_Grade_brown, Weight_Kg_brown, Low_Price_brown, Sales_Total_brown, Stock_On_Hand_brown, month_brown, day_brown)
            st.success(f'Predicted Average Price of Onion Brown: R{predicted_price_brown:.2f}')

    elif selected_model == "Onion Mild":
        # Display input fields for Onion Mild
        st.header("Onion Mild Features")
        
        col1, col2 = st.columns(2)
        with col1:
            Province_mild = st.selectbox('Province', ['NORTH WEST', 'WESTERN CAPE - CERES', 'TRANSVAAL','OTHER AREAS', ''])
            Size_Grade_mild = st.selectbox("Size Grade", ['1M', '2L', '1R', '1L', '1Z', '1S', '1X', '3L', '2R', '2M', '3S','3Z', '3M', '2Z', '3R', '2S'])
            Container_mild = st.selectbox("Container", ["AA100","AC030","AF070","AG100","AL200"])
            Weight_Kg_mild = st.selectbox("Weight Per Kilogram", [1,2,3,7,10,20])
            Low_Price_mild = st.number_input("Low Price(R)", min_value=0)
        with col2:
            Total_Kg_Sold_mild = st.number_input('Total Kilograms Sold', min_value=0)
            High_Price_mild = st.number_input("High Price(R)", min_value=0)
            Sales_Total_mild = st.number_input('Total Sales(R)', min_value=0)
            Stock_On_Hand_mild = st.number_input('Stock On Hand', step=1)
            month_mild = st.slider("Month", 1, 12)

        # Button to predict Onion Mild price
        if st.button("Predict Onion Mild"):
            predicted_price_mild = predict_price_mild(Province_mild,Container_mild,Size_Grade_mild,Weight_Kg_mild,Sales_Total_mild,Low_Price_mild,High_Price_mild,Total_Kg_Sold_mild,month_mild,Stock_On_Hand_mild)
            st.success(f'Predicted Average Price of Onion Mild: R{predicted_price_mild:.2f}')

    elif selected_model == "Potato Washed Mondial":
        # Display input fields for Potato Washed Mondial
        st.header("Potato Washed Mondial Features")
        
        col1, col2 = st.columns(2)
        with col1:
            Province_potato = st.selectbox('Province', ['NORTHERN CAPE', 'WESTERN CAPE - CERES', 'WEST COAST','SOUTH WESTERN FREE STATE', 'WESTERN FREESTATE', 'KWAZULU NATAL',
                                          'OTHER AREAS', 'TRANSVAAL', 'EASTERN FREESTATE', 'MPUMALANGA', 'NORTH EASTERN CAPE', 'NORTH WEST', 'SOUTHERN CAPE'])
            Size_Grade_potato = st.selectbox("Size Grade", ['1L', '1M', '1R', '1S', '1U', '1X', '1Z', '2L', '2M', '2S', '2U', '2X', '2Z',
                                              '3L', '3M', '3R', '3S', '3U', '3X', '3X', '3Z', '4L', '4M', '4R', '4S', '4U', '4Z'])
            Weight_Kg_potato = st.selectbox("Weight Per Kilogram", [7,10])
            Low_Price_potato = st.number_input("Low Price(R)", min_value=0)
        with col2:
            High_Price_potato = st.number_input("High Price(R)", min_value=0)
            Total_Kg_Sold_potato = st.number_input('Total Kilograms Sold(R)', min_value=0)
            Sales_Total_potato = st.number_input('Total Sales(R)', min_value=0)
            Stock_On_Hand_potato = st.number_input('Stock On Hand', step=1)

        # Button to predict Potato Washed Mondial price
        if st.button("Predict Potato Washed Mondial"):
            predicted_price_potato = predict_price(Province_potato,Size_Grade_potato,Weight_Kg_potato,Sales_Total_potato,Low_Price_potato,High_Price_potato,Total_Kg_Sold_potato,Stock_On_Hand_potato)
            st.success(f'Predicted Average Price of Potato Washed Mondial: R{predicted_price_potato:.2f}')
    elif selected_model == "Tomato Long Life":
        # Display input fields for Market Prediction
        st.header("Tomato Long Life Prediction Features")
        model = load_market_model()
        

        features = ['Weight in Kg', 'Low Price in Rands', 'High Price in Rands', 'Province', 'Container','Size Grade', 'Month']
        #features = ['Weight_Kg', 'Low_Price', 'High_Price','Month_encoded', 'Province_encoded', 'Container_encoded','Size_Grade_encoded']
        
        month = ["April","August","February","December","January","July","June","March","May","October","September"]
        province = ["NATAL","NORTH EASTERN CAPE","TRANSVAAL"]
        container = ["BM050","BS060","BT070"]
        sizegrade = ["1L", "1M","1R","1S","1U","1X","1Z","2L","2M","2R","2S","2X","2Z","3M","3R","3S","3Z"]
        
        user_inputs = {}
        for feature in features:
            if feature == "Province" or feature == "Container" or feature == "Month":
                if feature == "Province":
                    ss = province
                elif feature == "Container":
                    ss = container
                elif feature == "Size Grade":
                    ss = sizegrade
                elif feature == "Month":
                    ss = month
        
                display = (ss)


                options = list(range(len(display)))

                value = st.selectbox(feature, options, format_func=lambda x: display[x], key=feature)
                
                user_inputs[feature] = value
                
            else:
                user_input = st.text_input(f"Enter {feature}:")
                try:
                    user_inputs[feature] = float(user_input)
                except:
                    st.write("")

        # Prediction button
        if st.button('Predict'):
        # Convert user inputs into DataFrame
            user_inputs_df = pd.DataFrame([user_inputs])
        #st.write("User inputs:", user_inputs_df) # Debug statement
        
        # Predict using the loaded model
            try:
                prediction = model.predict(user_inputs_df)
                st.write(f'Predicted Average Price of Tomato Long Life: R{prediction[0]:.2f}')
            except Exception as e:
                st.write("Error occurred during prediction:", e)   

    elif selected_model == "Potato SIFRA (WASHED)":
        # Display input fields for Potato SIFRA (WASHED)
        st.header("Potato SIFRA (WASHED) Features")
       
        col1, col2 = st.columns(2)
        with col1:
            Province_sifra = st.selectbox('Province', ['NORTHERN CAPE', 'WESTERN CAPE - CERES', 'WEST COAST','SOUTH WESTERN FREE STATE', 'WESTERN FREESTATE', 'NATAL',
                                          'KWAZULU NATAL', 'OTHER AREAS', 'TRANSVAAL'])
            Size_Grade_sifra = st.selectbox("Size Grade", ['1R', '1M', '1Z', '2L', '2Z', '3M', '1S', '3R', '2M', '1U', '3Z', '1L', '2S', '2R', '4Z', '3L', '3U', '4M', '4L', '3S', '2U', '4R'])
            Weight_Kg_sifra = st.selectbox("Weight Per Kilogram", [2,4,7,10])
            Low_Price_sifra = st.number_input("Low Price(R)", min_value=0)
        with col2:
            Sales_Total_sifra = st.number_input('Total Sales(R)', min_value=0)
            Stock_On_Hand_sifra = st.number_input('Stock On Hand', step=1)
            month_sifra = st.slider("Month", 1, 12)
            day_sifra = st.slider("Day", 1, 31)

        # Button to predict Potato SIFRA (WASHED) price
        if st.button("Predict Potato SIFRA (WASHED)"):
            predicted_price_sifra = predict_price_sifra(Province_sifra, Size_Grade_sifra, Weight_Kg_sifra, Low_Price_sifra, Sales_Total_sifra, Stock_On_Hand_sifra, month_sifra, day_sifra)
            st.success(f'Predicted Average Price of Potato SIFRA (WASHED): R{predicted_price_sifra:.2f}')

# Run the main function
if __name__ == "__main__":
    main()
