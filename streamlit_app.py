import altair as alt
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Set Streamlit page config
st.set_page_config(page_title="Effluent Prediction Dashboard", page_icon="🌊")
st.title("🌊 Effluent Prediction Dashboard")
st.write(
    """
    This app predicts chemical concentrations in industrial effluents based on historical data.
    It helps industries monitor environmental compliance and take necessary actions.
    Select parameters below to explore predictions!
    """
)

# Load and cache the data
@st.cache_data
def load_data():
    df = pd.read_csv("FINAL_SDI CSV FILE.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    return df

df = load_data()

# Encode categorical columns
categorical_cols = ['Industry', 'Chemical_Type', 'Weather_Condition', 'Production_Scale']
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Feature selection and scaling
features = ['Effluent_Volume_Liters', 'Chemical_Concentration_ppm', 'pH', 'TDS_ppm', 'Conductivity_uS', 'Production_Scale']
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(df_scaled, columns=features)

# Train-Test Split
train_size = int(len(df) * 0.8)
train, test = df_scaled[:train_size], df_scaled[train_size:]

# Prepare LSTM input data
X_train, y_train = train[:-1], train[1:]
X_test, y_test = test[:-1], test[1:]
X_train = np.array(X_train).reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = np.array(X_test).reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=16)

# Future Prediction Function
def predict_future(years=5):
    future_dates = pd.date_range(df.index[-1], periods=years * 12, freq='M')
    future_preds = model.predict(X_test[-len(future_dates):])
    future_preds = future_preds.reshape(-1, 6)
    future_preds_inversed = scaler.inverse_transform(future_preds)
    return pd.DataFrame({'Date': future_dates, 'Predicted_Concentration': future_preds.flatten()})

future_trends = predict_future(6)

# Streamlit UI Elements
industry = st.selectbox("Select Industry", df.index.unique())
year = st.slider("Select Year", 2025, 2030, 2027)

# Fetch prediction for selected year
predicted_value = future_trends[future_trends['Date'].dt.year == year]['Predicted_Concentration'].values[0]
effluent_volume = np.random.uniform(30000, 60000)
tds = np.random.uniform(500, 2000)
conductivity = np.random.uniform(800, 2000)
regulatory_limit = 100

# Generate Report
def generate_report(industry, year, concentration, reg_limit, effluent_volume, tds, conductivity):
    output = []
    if concentration > reg_limit:
        output.append(f"The chemical concentration for {industry} in {year} is {concentration:.2f} ppm, exceeding the limit of {reg_limit} ppm. Action required!")
    else:
        output.append(f"The chemical concentration for {industry} in {year} is {concentration:.2f} ppm, within safe limits.")
    if effluent_volume > 50000:
        output.append(f"Effluent volume is high ({effluent_volume:.2f} liters), indicating increased industrial output.")
    if tds > 1000:
        output.append(f"TDS level ({tds:.2f} ppm) suggests potential water quality issues.")
    if conductivity > 1500:
        output.append(f"Conductivity ({conductivity:.2f} µS) indicates high ion concentration.")
    return "\n".join(output)

st.write(generate_report(industry, year, predicted_value, regulatory_limit, effluent_volume, tds, conductivity))

# Data Visualization
df_chart = future_trends.copy()
df_chart['Year'] = df_chart['Date'].dt.year
chart = (
    alt.Chart(df_chart)
    .mark_line()
    .encode(
        x=alt.X("Year:N", title="Year"),
        y=alt.Y("Predicted_Concentration:Q", title="Predicted Chemical Concentration (ppm)"),
    )
    .properties(height=320)
)
st.altair_chart(chart, use_container_width=True)
