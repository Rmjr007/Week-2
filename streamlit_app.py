import streamlit as st
import pandas as pd
import joblib
import os

# -------------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------------
st.set_page_config(
    page_title="EV Policy Simulator 📈",
    page_icon="🚗",
    layout="wide"
)

st.title("📈 EV Policy Simulator Dashboard")

# --- TABS FOR SIMULATOR AND CHATBOT ---
tab1, tab2 = st.tabs(["Policy Simulator", "App Assistant (Chatbot)"])

# -------------------------------------------------------
# TAB 1: POLICY SIMULATOR
# -------------------------------------------------------
with tab1:
    st.markdown("Use this app to predict an outcome based on electric vehicle policy and economic factors.")

    # -------------------------------------------------------
    # Load trained model and scaler
    # -------------------------------------------------------

    # Define file paths
    MODEL_PATH = "models/ev_policy_best_model.pkl"
    SCALER_PATH = "models/scaler.pkl"

    # Helper function to load files
    @st.cache_resource
    def load_model(model_path, scaler_path):
        # Check for model file
        if not os.path.exists(model_path):
            st.error(f"**Error:** Model file not found at `{model_path}`.")
            return None, None
            
        # Check for scaler file
        if not os.path.exists(scaler_path):
            st.error(f"**Error:** Scaler file not found at `{scaler_path}`.")
            return None, None
            
        # Load files
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            return model, scaler
        except Exception as e:
            st.error(f"Error loading model or scaler: {e}")
            return None, None

    model, scaler = load_model(MODEL_PATH, SCALER_PATH)

    # Define the exact feature names the model expects
    EXPECTED_FEATURES = [
        'total_vehicles_registered',
        'ev_percentage_share',
        'charging_stations_count',
        'avg_cost_ev',
        'avg_cost_gasoline_vehicle',
        'gov_incentive_amount',
        'co2_emissions_per_vehicle',
        'fuel_price_per_liter',
        'electricity_price_per_kwh'
    ]

    # -------------------------------------------------------
    # User Input Section
    # -------------------------------------------------------
    st.subheader("🔧 Enter Policy & Economic Data")

    if model and scaler:
        col1, col2, col3 = st.columns(3)

        with col1:
            total_vehicles = st.number_input("Total Vehicles Registered", value=None, format="%.2f", key="sim_total_vehicles", placeholder="Enter value")
            ev_share = st.number_input("EV Percentage Share (%)", value=None, format="%.2f", key="sim_ev_share", placeholder="Enter value")
            charging_stations = st.number_input("Charging Stations Count", value=None, format="%.2f", key="sim_charging_stations", placeholder="Enter value")

        with col2:
            avg_cost_ev = st.number_input("Avg. Cost of EV ($)", value=None, format="%.2f", key="sim_avg_cost_ev", placeholder="Enter value")
            avg_cost_gas = st.number_input("Avg. Cost of Gasoline Vehicle ($)", value=None, format="%.2f", key="sim_avg_cost_gas", placeholder="Enter value")
            incentive = st.number_input("Government Incentive ($)", value=None, format="%.2f", key="sim_incentive", placeholder="Enter value")

        with col3:
            co2_emissions = st.number_input("CO2 Emissions per Vehicle (g/km)", value=None, format="%.2f", key="sim_co2", placeholder="Enter value")
            fuel_price = st.number_input("Fuel Price (per Liter) ($)", value=None, format="%.2f", key="sim_fuel_price", placeholder="Enter value")
            electricity_price = st.number_input("Electricity Price (per kWh) ($)", value=None, format="%.2f", key="sim_electricity_price", placeholder="Enter value")

        # -------------------------------------------------------
        # Predict
        # -------------------------------------------------------
        if st.button("🔮 Run Simulation"):
            # Check if all fields are filled
            all_inputs = [
                total_vehicles, ev_share, charging_stations,
                avg_cost_ev, avg_cost_gas, incentive,
                co2_emissions, fuel_price, electricity_price
            ]
            
            if None in all_inputs:
                st.warning("Please enter a value for all 9 input fields.")
            else:
                try:
                    # Create DataFrame with the exact feature names
                    input_data = pd.DataFrame(
                        data=[[
                            float(total_vehicles),
                            float(ev_share),
                            float(charging_stations),
                            float(avg_cost_ev),
                            float(avg_cost_gas),
                            float(incentive),
                            float(co2_emissions),
                            float(fuel_price),
                            float(electricity_price)
                        ]],
                        columns=EXPECTED_FEATURES
                    )
                    
                    st.info("Calculating... (Scaling data and running prediction)")
                    
                    # Scale Data
                    scaled_data = scaler.transform(input_data)
                    
                    # Predict
                    prediction = model.predict(scaled_data)[0]
                    
                    st.success(f"**Predicted Outcome:** {prediction:,.2f}")
                    st.caption("(Note: This is the raw output from the linear regression model.)")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}. Please ensure all inputs are valid numbers.")

    else:
        st.error("Application cannot start. Please check file paths and ensure `models/ev_policy_best_model.pkl` and `models/scaler.pkl` are in your GitHub repository.")

# -------------------------------------------------------
# TAB 2: APP ASSISTANT (CHATBOT)
# -------------------------------------------------------
with tab2:
    st.header("App Assistant")
    st.markdown("Ask me about this app or its input fields.")

    # Simple rule-based response function
    def get_bot_response(user_query):
        """
        Parses the user query and returns a simple, rule-based response.
        """
        query = user_query.lower()
        
        # General Greetings / Info
        if "hello" in query or "hi" in query:
            return "Hello! How can I help you understand the EV Policy Simulator?"
        if "what does this app do" in query or "purpose" in query or "about this app" in query:
            return "This app uses a machine learning model to predict an outcome based on 9 different economic and policy inputs related to electric vehicles."
        if "model" in query:
            return "The app uses a Linear Regression model (`ev_policy_best_model.pkl`) that was trained on 9 features."
        if "scaler" in query:
            return "The app uses a `StandardScaler` (`scaler.pkl`) to normalize the inputs before sending them to the model. This is crucial for the model's accuracy."

        # Feature-specific questions
        if "total vehicles" in query:
            return "This is the 'Total Vehicles Registered' input. It represents the total number of vehicles (gasoline and EV) in the area you are simulating."
        if "ev percentage" in query or "ev share" in query:
            return "This is the 'EV Percentage Share (%)'. It's the percentage of total vehicles that are electric."
        if "charging stations" in query:
            return "This is the 'Charging Stations Count'. It represents the total number of public charging stations available."
        if "cost of ev" in query:
            return "This is the 'Avg. Cost of EV ($)'. It's the average purchase price for a new electric vehicle."
        if "cost of gasoline" in query or "cost of gas vehicle" in query:
            return "This is the 'Avg. Cost of Gasoline Vehicle ($)'. It's the average purchase price for a new gasoline-powered vehicle."
        if "incentive" in query or "government" in query:
            return "This is the 'Government Incentive ($)'. It refers to the average amount of money (e.g., tax credit, rebate) the government provides for buying an EV."
        if "co2" in query or "emissions" in query:
            return "This is the 'CO2 Emissions per Vehicle (g/km)'. It represents the average CO2 emissions from a typical vehicle."
        if "fuel price" in query:
            return "This is the 'Fuel Price (per Liter) ($)'. It's the average cost of one liter of gasoline."
        if "electricity price" in query:
            return "This is the 'Electricity Price (per kWh) ($)'. It's the average cost of one kilowatt-hour of electricity, which is used to charge EVs."

        # Fallback
        return "I'm sorry, I don't have information on that. I can only answer basic questions about the simulator's 9 input fields (like 'What is fuel price?' or 'What does this app do?')."

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! Ask me what this app does or what an input field means (e.g., 'What is gov_incentive_amount?')."}]

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display bot response
        bot_response = get_bot_response(prompt)
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        # Add bot response to chat history
        st.session_state.messages.append({"role": "assistant", "content": bot_response})

# -------------------------------------------------------
# Footer (Common to all tabs)
# -------------------------------------------------------
st.markdown("---")
st.caption("A Streamlit app built to work with `ev_policy_best_model.pkl`.")
