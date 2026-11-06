
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Page settings
st.set_page_config(
    page_title="Weather Prediction System",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Prediction System")
st.write("Analyze weather conditions and predict the weather.")

# Generate sample weather data
@st.cache_data
def create_data():
    np.random.seed(42)
    n = 500

    humidity = np.random.uniform(20, 100, n)
    rainfall = np.random.uniform(0, 20, n)
    wind = np.random.uniform(0, 40, n)
    cloud = np.random.uniform(0, 100, n)

    temperature = (
        38 - humidity * 0.12
        - cloud * 0.04
        - rainfall * 0.1
        + np.random.normal(0, 2, n)
    )

    temperature = np.clip(temperature, 5, 45)

    weather = []

    for i in range(n):
        if rainfall[i] > 10:
            weather.append("Rainy")
        elif cloud[i] > 50:
            weather.append("Cloudy")
        else:
            weather.append("Sunny")

    return pd.DataFrame({
        "Temperature": temperature,
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Wind": wind,
        "Cloud": cloud,
        "Weather": weather
    })


data = create_data()

# Prepare data
features = ["Humidity", "Rainfall", "Wind", "Cloud"]
X = data[features]

# Classification model
weather_model = DecisionTreeClassifier(random_state=42)
weather_model.fit(X, data["Weather"])

# Regression model
temperature_model = LinearRegression()
temperature_model.fit(X, data["Temperature"])

# Clustering model
scaler = StandardScaler()

scaled_data = scaler.fit_transform(
    data[["Temperature", "Humidity", "Rainfall", "Wind"]]
)

group_model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

data["Group"] = group_model.fit_predict(scaled_data)

# Weather overview
st.header("Weather Overview")

c1, c2, c3 = st.columns(3)

c1.metric("Total Records", len(data))

c2.metric(
    "Average Temperature",
    f"{data['Temperature'].mean():.1f} °C"
)

c3.metric(
    "Average Humidity",
    f"{data['Humidity'].mean():.1f} %"
)

# Weather analysis
st.header("Weather Analysis")

fig, ax = plt.subplots()

data["Weather"].value_counts().plot(
    kind="bar", ax=ax
)

ax.set_xlabel("Weather")
ax.set_ylabel("Number of Records")

st.pyplot(fig)
plt.close(fig)

# Temperature distribution
st.subheader("Temperature Distribution")

fig, ax = plt.subplots()

ax.hist(data["Temperature"], bins=20)

ax.set_xlabel("Temperature (°C)")
ax.set_ylabel("Frequency")

st.pyplot(fig)
plt.close(fig)

# Weather prediction
st.header("Check Weather")

humidity = st.slider("Humidity (%)", 20, 100, 60)
rainfall = st.slider("Rainfall (mm)", 0, 20, 5)
wind = st.slider("Wind Speed (km/h)", 0, 40, 10)
cloud = st.slider("Cloud Cover (%)", 0, 100, 40)

if st.button("Predict Weather"):

    new_data = pd.DataFrame([{
        "Humidity": humidity,
        "Rainfall": rainfall,
        "Wind": wind,
        "Cloud": cloud
    }])

    result = weather_model.predict(new_data)[0]
    temp = temperature_model.predict(new_data)[0]

    st.success(f"Weather Prediction: {result}")
    st.info(f"Estimated Temperature: {temp:.1f} °C")

# Weather patterns
st.header("Weather Patterns")

fig, ax = plt.subplots()

ax.scatter(
    data["Humidity"],
    data["Temperature"],
    c=data["Group"]
)

ax.set_xlabel("Humidity (%)")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Similar Weather Conditions")

st.pyplot(fig)
plt.close(fig)

st.caption("Weather Prediction System | Academic Project")
