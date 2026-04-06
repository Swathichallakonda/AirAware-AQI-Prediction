# import os
# import pickle
# import numpy as np
# import pandas as pd
# import streamlit as st
# import plotly.express as px
# import plotly.graph_objects as go
# from xgboost import XGBRegressor

# # ---------------------------
# # PAGE CONFIG
# # ---------------------------
# st.set_page_config(
#     page_title="AirAware Smart Air Quality Analytics",
#     page_icon="🌍",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # ---------------------------
# # PATHS
# # ---------------------------
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_PATH = os.path.join(BASE_DIR, "C:/Users/abc/Downloads/AQI_Project/globalAirQuality.csv")
# MODEL_PATH = os.path.join(BASE_DIR, "C:/Users/abc/Downloads/AQI_Project/aqi_xgboost_model.pkl")

# # ---------------------------
# # CUSTOM CSS
# # ---------------------------
# st.markdown("""
# <style>
# .main {
#     background: linear-gradient(135deg, #050816 0%, #0b1026 45%, #111c44 100%);
# }
# .block-container {
#     padding-top: 1.5rem;
#     padding-bottom: 2rem;
# }
# h1, h2, h3, h4 {
#     color: #ffffff !important;
# }
# [data-testid="stMetricValue"] {
#     color: #ffffff;
# }
# [data-testid="stMetricLabel"] {
#     color: #d0d7ff;
# }
# .glass-card {
#     background: rgba(255, 255, 255, 0.06);
#     border: 1px solid rgba(255,255,255,0.12);
#     border-radius: 20px;
#     padding: 18px;
#     box-shadow: 0 8px 30px rgba(0,0,0,0.35);
#     backdrop-filter: blur(10px);
# }
# .hero-title {
#     font-size: 3rem;
#     font-weight: 800;
#     color: white;
#     margin-bottom: 0.2rem;
# }
# .hero-sub {
#     color: #c8d2ff;
#     font-size: 1.05rem;
#     margin-bottom: 1rem;
# }
# .small-note {
#     color: #aeb8e8;
#     font-size: 0.95rem;
# }
# hr {
#     border: none;
#     height: 1px;
#     background: linear-gradient(to right, transparent, rgba(255,255,255,0.25), transparent);
#     margin: 1rem 0 1.5rem 0;
# }
# </style>
# """, unsafe_allow_html=True)

# # ---------------------------
# # HELPERS
# # ---------------------------
# FEATURES = [
#     "pm10", "pm25", "no2", "so2", "o3", "co",
#     "temperature", "humidity", "wind_speed"
# ]

# def aqi_category(aqi):
#     if aqi <= 50:
#         return "Good"
#     elif aqi <= 100:
#         return "Moderate"
#     elif aqi <= 150:
#         return "Unhealthy for Sensitive Groups"
#     elif aqi <= 200:
#         return "Unhealthy"
#     elif aqi <= 300:
#         return "Very Unhealthy"
#     return "Hazardous"

# def load_data():
#     df = pd.read_csv(DATA_PATH)
#     df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
#     df = df.drop_duplicates().sort_values("timestamp").ffill()
#     df["AQI_Category"] = df["aqi"].apply(aqi_category)
#     return df

# def train_and_save_model(df):
#     model = XGBRegressor(
#         n_estimators=300,
#         max_depth=6,
#         learning_rate=0.05,
#         random_state=42
#     )
#     X = df[FEATURES]
#     y = df["aqi"]
#     model.fit(X, y)
#     with open(MODEL_PATH, "wb") as f:
#         pickle.dump(model, f)
#     return model

# def load_model(df):
#     if os.path.exists(MODEL_PATH):
#         with open(MODEL_PATH, "rb") as f:
#             return pickle.load(f)
#     return train_and_save_model(df)

# def gauge_color(aqi):
#     if aqi <= 50:
#         return "#16c784"
#     elif aqi <= 100:
#         return "#f5b700"
#     elif aqi <= 150:
#         return "#ff8c42"
#     elif aqi <= 200:
#         return "#ff4d4f"
#     elif aqi <= 300:
#         return "#8f3fff"
#     return "#7a0019"

# # ---------------------------
# # LOAD
# # ---------------------------
# if not os.path.exists(DATA_PATH):
#     st.error("globalAirQuality.csv not found in the same folder as app.py")
#     st.stop()

# df = load_data()
# model = load_model(df)

# # ---------------------------
# # SIDEBAR
# # ---------------------------
# st.sidebar.title("⚙️ Control Panel")

# all_cities = sorted(df["city"].dropna().unique().tolist())
# selected_city = st.sidebar.selectbox("Select City", all_cities)

# city_df = df[df["city"] == selected_city].copy()

# min_date = city_df["timestamp"].min().date()
# max_date = city_df["timestamp"].max().date()

# date_range = st.sidebar.date_input(
#     "Select Date Range",
#     value=(min_date, max_date),
#     min_value=min_date,
#     max_value=max_date
# )

# if isinstance(date_range, tuple) and len(date_range) == 2:
#     start_date, end_date = date_range
#     city_df = city_df[
#         (city_df["timestamp"].dt.date >= start_date) &
#         (city_df["timestamp"].dt.date <= end_date)
#     ]

# show_raw = st.sidebar.checkbox("Show Raw Data Table", value=False)

# # ---------------------------
# # HEADER
# # ---------------------------
# st.markdown('<div class="hero-title">AirAware: Smart Air Quality Analytics</div>', unsafe_allow_html=True)
# st.markdown(
#     f'<div class="hero-sub">Advanced AQI monitoring, pollutant intelligence, prediction, alerts, and model retraining for <b>{selected_city}</b>.</div>',
#     unsafe_allow_html=True
# )
# st.markdown('<div class="small-note">This dashboard combines machine learning, visual analytics, and real-time-style air quality exploration.</div>', unsafe_allow_html=True)
# st.markdown("<hr>", unsafe_allow_html=True)

# if city_df.empty:
#     st.warning("No data available for the selected filters.")
#     st.stop()

# latest_row = city_df.iloc[-1]
# latest_aqi = float(latest_row["aqi"])
# avg_aqi = float(city_df["aqi"].mean())
# max_aqi = float(city_df["aqi"].max())
# latest_category = aqi_category(latest_aqi)

# # ---------------------------
# # KPI CARDS
# # ---------------------------
# c1, c2, c3, c4 = st.columns(4)
# with c1:
#     st.metric("Current AQI", f"{latest_aqi:.0f}")
# with c2:
#     st.metric("Average AQI", f"{avg_aqi:.2f}")
# with c3:
#     st.metric("Peak AQI", f"{max_aqi:.0f}")
# with c4:
#     st.metric("Air Quality Status", latest_category)

# # ---------------------------
# # ALERTS
# # ---------------------------
# st.subheader("🚨 AQI Alert Engine")
# if latest_aqi <= 50:
#     st.success("Air quality is good. Minimal health risk.")
# elif latest_aqi <= 100:
#     st.info("Air quality is moderate. Sensitive groups should stay cautious.")
# elif latest_aqi <= 150:
#     st.warning("Unhealthy for sensitive groups. Consider reducing prolonged outdoor exposure.")
# elif latest_aqi <= 200:
#     st.warning("Air quality is unhealthy. Outdoor activity should be limited.")
# else:
#     st.error("Air quality is very unhealthy or hazardous. Strong precautions are recommended.")

# # ---------------------------
# # TOP VISUALS
# # ---------------------------
# left, right = st.columns([1, 1])

# with left:
#     st.subheader("AQI Live Gauge")
#     gauge_fig = go.Figure(go.Indicator(
#         mode="gauge+number",
#         value=latest_aqi,
#         number={"font": {"size": 42, "color": "white"}},
#         title={"text": f"{selected_city} AQI", "font": {"size": 24, "color": "white"}},
#         gauge={
#             "axis": {"range": [0, 500], "tickcolor": "white"},
#             "bar": {"color": gauge_color(latest_aqi)},
#             "bgcolor": "rgba(0,0,0,0)",
#             "borderwidth": 0,
#             "steps": [
#                 {"range": [0, 50], "color": "#16c784"},
#                 {"range": [50, 100], "color": "#f5b700"},
#                 {"range": [100, 150], "color": "#ff8c42"},
#                 {"range": [150, 200], "color": "#ff4d4f"},
#                 {"range": [200, 300], "color": "#8f3fff"},
#                 {"range": [300, 500], "color": "#7a0019"}
#             ],
#         }
#     ))
#     gauge_fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         font={"color": "white"},
#         height=420
#     )
#     st.plotly_chart(gauge_fig, use_container_width=True)

# with right:
#     st.subheader("AQI Trend Over Time")
#     trend_fig = px.line(
#         city_df,
#         x="timestamp",
#         y="aqi",
#         title=f"AQI Trend - {selected_city}",
#         template="plotly_dark"
#     )
#     trend_fig.update_traces(line=dict(width=3))
#     trend_fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         xaxis_title="Timestamp",
#         yaxis_title="AQI",
#         height=420
#     )
#     st.plotly_chart(trend_fig, use_container_width=True)

# # ---------------------------
# # SECOND ROW
# # ---------------------------
# t1, t2 = st.columns([1, 1])

# with t1:
#     st.subheader("Pollutant Comparison")
#     latest_pollutants = pd.DataFrame({
#         "Pollutant": ["PM2.5", "PM10", "NO2", "SO2", "O3", "CO"],
#         "Value": [
#             latest_row["pm25"], latest_row["pm10"], latest_row["no2"],
#             latest_row["so2"], latest_row["o3"], latest_row["co"]
#         ]
#     })
#     pollutant_fig = px.bar(
#         latest_pollutants,
#         x="Pollutant",
#         y="Value",
#         color="Value",
#         template="plotly_dark",
#         title="Latest Pollutant Levels"
#     )
#     pollutant_fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         height=420
#     )
#     st.plotly_chart(pollutant_fig, use_container_width=True)

# with t2:
#     st.subheader("Weather Conditions")
#     weather_df = pd.DataFrame({
#         "Metric": ["Temperature", "Humidity", "Wind Speed"],
#         "Value": [latest_row["temperature"], latest_row["humidity"], latest_row["wind_speed"]]
#     })
#     weather_fig = px.pie(
#         weather_df,
#         names="Metric",
#         values="Value",
#         hole=0.55,
#         template="plotly_dark",
#         title="Current Weather Contribution View"
#     )
#     weather_fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         height=420
#     )
#     st.plotly_chart(weather_fig, use_container_width=True)

# # ---------------------------
# # TABS
# # ---------------------------
# tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
#     "📈 Trend Analysis",
#     "🧠 3D Analytics",
#     "🌍 Spatial View",
#     "📊 Feature Insights",
#     "🔮 AQI Prediction",
#     "🔁 Model Retraining"
# ])

# with tab1:
#     st.subheader("Detailed Trend Analysis")

#     sub1, sub2 = st.columns(2)

#     with sub1:
#         pm25_fig = px.line(
#             city_df, x="timestamp", y="pm25",
#             template="plotly_dark", title="PM2.5 Trend"
#         )
#         pm25_fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)",
#             height=400
#         )
#         st.plotly_chart(pm25_fig, use_container_width=True)

#     with sub2:
#         pm10_fig = px.line(
#             city_df, x="timestamp", y="pm10",
#             template="plotly_dark", title="PM10 Trend"
#         )
#         pm10_fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)",
#             height=400
#         )
#         st.plotly_chart(pm10_fig, use_container_width=True)

#     corr_df = city_df.select_dtypes(include=np.number)
#     corr = corr_df.corr()

#     heatmap_fig = px.imshow(
#         corr,
#         text_auto=True,
#         color_continuous_scale="RdBu_r",
#         template="plotly_dark",
#         title="Correlation Heatmap"
#     )
#     heatmap_fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         height=650
#     )
#     st.plotly_chart(heatmap_fig, use_container_width=True)

# with tab2:
#     st.subheader("3D Pollution Intelligence")

#     scatter3d_fig = px.scatter_3d(
#         city_df,
#         x="pm25",
#         y="no2",
#         z="o3",
#         color="aqi",
#         size="pm10",
#         hover_data=["timestamp", "city", "aqi"],
#         template="plotly_dark",
#         title="3D Relationship: PM2.5 vs NO2 vs O3"
#     )
#     scatter3d_fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         scene=dict(
#             xaxis_title="PM2.5",
#             yaxis_title="NO2",
#             zaxis_title="O3",
#             bgcolor="rgba(0,0,0,0)"
#         ),
#         height=700
#     )
#     st.plotly_chart(scatter3d_fig, use_container_width=True)

# with tab3:
#     st.subheader("Geographic Air Quality Map")

#     map_df = df.groupby(["city", "country", "latitude", "longitude"], as_index=False).agg({
#         "aqi": "mean",
#         "pm25": "mean",
#         "pm10": "mean"
#     })

#     map_fig = px.scatter_map(
#         map_df,
#         lat="latitude",
#         lon="longitude",
#         color="aqi",
#         size="pm25",
#         hover_name="city",
#         hover_data=["country", "pm10"],
#         zoom=1,
#         height=650,
#         title="Average AQI Across Cities",
#         color_continuous_scale="Turbo"
#     )
#     map_fig.update_layout(
#         map_style="open-street-map",
#         paper_bgcolor="rgba(0,0,0,0)",
#         margin={"r": 0, "t": 60, "l": 0, "b": 0}
#     )
#     st.plotly_chart(map_fig, use_container_width=True)

# with tab4:
#     st.subheader("Feature Importance & Distributions")

#     if hasattr(model, "feature_importances_"):
#         fi_df = pd.DataFrame({
#             "Feature": FEATURES,
#             "Importance": model.feature_importances_
#         }).sort_values("Importance", ascending=False)

#         fi_fig = px.bar(
#             fi_df,
#             x="Importance",
#             y="Feature",
#             orientation="h",
#             color="Importance",
#             template="plotly_dark",
#             title="XGBoost Feature Importance"
#         )
#         fi_fig.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)",
#             height=450
#         )
#         st.plotly_chart(fi_fig, use_container_width=True)

#     dist_fig = px.histogram(
#         city_df,
#         x="aqi",
#         nbins=40,
#         color="AQI_Category",
#         template="plotly_dark",
#         title="AQI Distribution"
#     )
#     dist_fig.update_layout(
#         paper_bgcolor="rgba(0,0,0,0)",
#         plot_bgcolor="rgba(0,0,0,0)",
#         height=450
#     )
#     st.plotly_chart(dist_fig, use_container_width=True)

# with tab5:
#     st.subheader("AQI Prediction Engine")

#     c1, c2, c3 = st.columns(3)
#     with c1:
#         pm25 = st.number_input("PM2.5", value=float(city_df["pm25"].mean()))
#         pm10 = st.number_input("PM10", value=float(city_df["pm10"].mean()))
#         no2 = st.number_input("NO2", value=float(city_df["no2"].mean()))
#     with c2:
#         so2 = st.number_input("SO2", value=float(city_df["so2"].mean()))
#         o3 = st.number_input("O3", value=float(city_df["o3"].mean()))
#         co = st.number_input("CO", value=float(city_df["co"].mean()))
#     with c3:
#         temperature = st.number_input("Temperature", value=float(city_df["temperature"].mean()))
#         humidity = st.number_input("Humidity", value=float(city_df["humidity"].mean()))
#         wind_speed = st.number_input("Wind Speed", value=float(city_df["wind_speed"].mean()))

#     if st.button("Predict AQI", use_container_width=True):
#         input_df = pd.DataFrame([{
#             "pm10": pm10,
#             "pm25": pm25,
#             "no2": no2,
#             "so2": so2,
#             "o3": o3,
#             "co": co,
#             "temperature": temperature,
#             "humidity": humidity,
#             "wind_speed": wind_speed
#         }])

#         pred = float(model.predict(input_df)[0])
#         pred_cat = aqi_category(pred)

#         st.success(f"Predicted AQI: {pred:.2f}")
#         st.info(f"Predicted Category: {pred_cat}")

#         pred_gauge = go.Figure(go.Indicator(
#             mode="gauge+number",
#             value=pred,
#             number={"font": {"size": 40, "color": "white"}},
#             title={"text": "Predicted AQI", "font": {"size": 24, "color": "white"}},
#             gauge={
#                 "axis": {"range": [0, 500], "tickcolor": "white"},
#                 "bar": {"color": gauge_color(pred)},
#                 "steps": [
#                     {"range": [0, 50], "color": "#16c784"},
#                     {"range": [50, 100], "color": "#f5b700"},
#                     {"range": [100, 150], "color": "#ff8c42"},
#                     {"range": [150, 200], "color": "#ff4d4f"},
#                     {"range": [200, 300], "color": "#8f3fff"},
#                     {"range": [300, 500], "color": "#7a0019"}
#                 ]
#             }
#         ))
#         pred_gauge.update_layout(
#             paper_bgcolor="rgba(0,0,0,0)",
#             plot_bgcolor="rgba(0,0,0,0)",
#             height=400
#         )
#         st.plotly_chart(pred_gauge, use_container_width=True)

# with tab6:
#     st.subheader("Automated Model Retraining")

#     uploaded_file = st.file_uploader("Upload a new CSV file for retraining", type=["csv"])

#     if uploaded_file is not None:
#         new_df = pd.read_csv(uploaded_file)
#         st.write("Uploaded Dataset Preview")
#         st.dataframe(new_df.head())

#         required_columns = set(FEATURES + ["aqi"])

#         if required_columns.issubset(set(new_df.columns)):
#             if st.button("Retrain and Save Model", use_container_width=True):
#                 new_df = new_df.drop_duplicates().ffill()

#                 X_new = new_df[FEATURES]
#                 y_new = new_df["aqi"]

#                 model.fit(X_new, y_new)

#                 with open(MODEL_PATH, "wb") as f:
#                     pickle.dump(model, f)

#                 st.success("Model retrained and saved successfully.")
#         else:
#             st.error("Uploaded CSV does not contain the required columns.")

# # ---------------------------
# # RAW DATA
# # ---------------------------
# if show_raw:
#     st.subheader("Raw Filtered Data")
#     st.dataframe(city_df)

# # ---------------------------
# # FOOTER
# # ---------------------------
# st.markdown("<hr>", unsafe_allow_html=True)
# st.markdown(
#     '<div class="small-note">AirAware Dashboard • Built with Streamlit, Plotly, XGBoost, and interactive visual analytics.</div>',
#     unsafe_allow_html=True
# )






import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="AirAware · Intelligent Air Quality",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# PATHS
# ---------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "C:/Users/abc/Downloads/AQI_Project/globalAirQuality.csv")
MODEL_PATH = os.path.join(BASE_DIR, "C:/Users/abc/Downloads/AQI_Project/aqi_xgboost_model.pkl")

# ---------------------------
# STUNNING CUSTOM CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

:root {
    --cyan:    #00f5ff;
    --magenta: #ff00cc;
    --green:   #00ff88;
    --orange:  #ff6b35;
    --purple:  #7b2fff;
    --bg-deep: #020408;
    --bg-mid:  #060d18;
    --bg-card: rgba(0,245,255,0.04);
    --border:  rgba(0,245,255,0.15);
    --text-hi: #e8f4ff;
    --text-lo: #7a9bbf;
}

html, body, [data-testid="stApp"] {
    background: var(--bg-deep) !important;
    font-family: 'Rajdhani', sans-serif;
    color: var(--text-hi);
}

[data-testid="stApp"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px);
    background-size: 60px 60px;
    animation: gridMove 20s linear infinite;
    pointer-events: none;
    z-index: 0;
}

@keyframes gridMove {
    0%   { background-position: 0 0; }
    100% { background-position: 60px 60px; }
}

[data-testid="stApp"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 600px 400px at 20% 20%, rgba(0,245,255,0.06) 0%, transparent 70%),
        radial-gradient(ellipse 500px 350px at 80% 70%, rgba(123,47,255,0.08) 0%, transparent 70%),
        radial-gradient(ellipse 400px 300px at 50% 90%, rgba(255,0,204,0.05) 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    animation: orbPulse 8s ease-in-out infinite alternate;
}

@keyframes orbPulse {
    0%   { opacity: 0.6; }
    100% { opacity: 1.2; }
}

.block-container {
    padding: 1rem 2rem 3rem !important;
    position: relative;
    z-index: 1;
}

.hero-wrapper {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 4px;
    color: var(--cyan);
    text-transform: uppercase;
    border: 1px solid rgba(0,245,255,0.3);
    padding: 4px 16px;
    border-radius: 20px;
    background: rgba(0,245,255,0.05);
    margin-bottom: 1rem;
    animation: fadeSlideDown 0.8s ease both;
}

.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2.2rem, 5vw, 4rem);
    font-weight: 900;
    line-height: 1.1;
    background: linear-gradient(135deg, #ffffff 0%, var(--cyan) 40%, var(--magenta) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.5rem;
    animation: fadeSlideDown 0.8s 0.1s ease both;
    filter: drop-shadow(0 0 30px rgba(0,245,255,0.3));
}

.hero-sub {
    font-size: 1.1rem;
    color: var(--text-lo);
    font-weight: 300;
    letter-spacing: 1px;
    animation: fadeSlideDown 0.8s 0.2s ease both;
    margin-bottom: 0.5rem;
}

.hero-city {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    color: var(--cyan);
    letter-spacing: 3px;
    animation: fadeSlideDown 0.8s 0.3s ease both;
}

@keyframes fadeSlideDown {
    from { opacity: 0; transform: translateY(-20px); }
    to   { opacity: 1; transform: translateY(0); }
}

.neon-divider {
    height: 1px;
    background: linear-gradient(to right,
        transparent,
        var(--cyan) 20%,
        var(--magenta) 50%,
        var(--cyan) 80%,
        transparent);
    margin: 1.5rem 0;
    box-shadow: 0 0 10px rgba(0,245,255,0.5);
    animation: dividerGlow 3s ease-in-out infinite alternate;
}

@keyframes dividerGlow {
    from { opacity: 0.5; }
    to   { opacity: 1; }
}

[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 16px !important;
    padding: 1.2rem 1.4rem !important;
    position: relative;
    overflow: hidden;
    transition: transform 0.3s ease, box-shadow 0.3s ease;
    backdrop-filter: blur(20px);
}

[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0,245,255,0.15), 0 0 0 1px rgba(0,245,255,0.3) !important;
    border-color: rgba(0,245,255,0.4) !important;
}

[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(to right, var(--cyan), var(--magenta));
}

[data-testid="stMetricValue"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 1.9rem !important;
    color: #ffffff !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: var(--cyan) !important;
    font-size: 0.7rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050d1a 0%, #020408 100%) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * { color: var(--text-hi) !important; }

[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(0,245,255,0.05) !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 10px !important;
}

[data-testid="stTabs"] [role="tablist"] {
    gap: 4px;
    border-bottom: 1px solid var(--border);
    background: transparent;
    padding-bottom: 0;
}

[data-testid="stTabs"] button[role="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1.5px !important;
    color: var(--text-lo) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px 10px 0 0 !important;
    padding: 8px 18px !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase;
}

[data-testid="stTabs"] button[role="tab"]:hover {
    color: var(--cyan) !important;
    background: rgba(0,245,255,0.05) !important;
    border-color: rgba(0,245,255,0.2) !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: var(--cyan) !important;
    background: rgba(0,245,255,0.08) !important;
    border-color: rgba(0,245,255,0.35) !important;
    border-bottom-color: transparent !important;
    box-shadow: 0 0 15px rgba(0,245,255,0.1) !important;
}

h1, h2, h3, h4 {
    font-family: 'Orbitron', monospace !important;
    color: #ffffff !important;
    letter-spacing: 1px;
}

h3 { font-size: 1rem !important; letter-spacing: 2px !important; }

[data-testid="stAlert"] {
    border-radius: 12px !important;
    border-width: 1px !important;
    font-family: 'Rajdhani', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
}

[data-testid="stButton"] > button {
    background: linear-gradient(135deg, rgba(0,245,255,0.15), rgba(123,47,255,0.15)) !important;
    border: 1px solid rgba(0,245,255,0.4) !important;
    color: var(--cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 2px !important;
    border-radius: 10px !important;
    padding: 0.7rem 2rem !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
}

[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, rgba(0,245,255,0.3), rgba(123,47,255,0.3)) !important;
    box-shadow: 0 0 25px rgba(0,245,255,0.3), 0 0 50px rgba(123,47,255,0.15) !important;
    transform: translateY(-2px) !important;
    border-color: var(--cyan) !important;
}

[data-testid="stNumberInput"] input {
    background: rgba(0,245,255,0.04) !important;
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 8px !important;
    color: white !important;
    font-family: 'Share Tech Mono', monospace !important;
}

[data-testid="stNumberInput"] label {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 1.5px !important;
    color: var(--text-lo) !important;
    text-transform: uppercase !important;
}

.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 4px;
    color: var(--cyan);
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    opacity: 0.8;
}

.section-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.05rem;
    font-weight: 700;
    color: white;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-title::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, var(--border), transparent);
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50%       { opacity: 0.4; transform: scale(0.6); }
}

.live-dot {
    display: inline-block;
    width: 7px; height: 7px;
    background: var(--green);
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse-dot 1.5s ease-in-out infinite;
    box-shadow: 0 0 8px var(--green);
}

[data-testid="stFileUploader"] {
    background: rgba(0,245,255,0.03) !important;
    border: 1px dashed rgba(0,245,255,0.25) !important;
    border-radius: 14px !important;
    padding: 1rem !important;
}

::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.3); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

.footer {
    text-align: center;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.68rem;
    color: rgba(122,155,191,0.5);
    letter-spacing: 3px;
    padding: 1.5rem 0 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# HELPERS
# ---------------------------
FEATURES = [
    "pm10", "pm25", "no2", "so2", "o3", "co",
    "temperature", "humidity", "wind_speed"
]

AQI_LEVELS = [
    (50,  "Good",                          "#00ff88", "rgba(0,255,136,0.12)"),
    (100, "Moderate",                      "#f5e642", "rgba(245,230,66,0.12)"),
    (150, "Unhealthy for Sensitive Groups","#ff8c42", "rgba(255,140,66,0.12)"),
    (200, "Unhealthy",                     "#ff4d4f", "rgba(255,77,79,0.12)"),
    (300, "Very Unhealthy",                "#b44fff", "rgba(180,79,255,0.12)"),
    (500, "Hazardous",                     "#ff2255", "rgba(255,34,85,0.12)"),
]

def aqi_meta(aqi):
    for threshold, label, color, bg in AQI_LEVELS:
        if aqi <= threshold:
            return label, color, bg
    return "Hazardous", "#ff2255", "rgba(255,34,85,0.12)"

def aqi_category(aqi):
    return aqi_meta(aqi)[0]

def gauge_color(aqi):
    return aqi_meta(aqi)[1]

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.drop_duplicates().sort_values("timestamp").ffill()
    df["AQI_Category"] = df["aqi"].apply(aqi_category)
    return df

def train_and_save_model(df):
    model = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05, random_state=42)
    model.fit(df[FEATURES], df["aqi"])
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    return model

def load_model(df):
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return train_and_save_model(df)

def dark_layout(height=420, margin=None):
    m = margin or dict(l=20, r=20, t=50, b=20)
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Rajdhani, sans-serif", color="#c8d8f0", size=13),
        height=height,
        margin=m,
        xaxis=dict(gridcolor="rgba(0,245,255,0.06)", zerolinecolor="rgba(0,245,255,0.1)"),
        yaxis=dict(gridcolor="rgba(0,245,255,0.06)", zerolinecolor="rgba(0,245,255,0.1)"),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.05)"),
        title_font=dict(family="Orbitron, monospace", size=14, color="white"),
    )

# ---------------------------
# LOAD DATA & MODEL
# ---------------------------
if not os.path.exists(DATA_PATH):
    st.error("⚠️  globalAirQuality.csv not found. Please check DATA_PATH in app.py")
    st.stop()

df = load_data()
model = load_model(df)

# ---------------------------
# SIDEBAR
# ---------------------------
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 1.5rem;">
        <div style="font-family:'Orbitron',monospace; font-size:1.1rem; font-weight:900;
                    background:linear-gradient(135deg,#00f5ff,#ff00cc);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            AIR<span style="-webkit-text-fill-color:#7b2fff">AWARE</span>
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem;
                    letter-spacing:3px; color:rgba(0,245,255,0.5); margin-top:4px;">
            CONTROL PANEL v2.0
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📍 Location</div>', unsafe_allow_html=True)
    all_cities = sorted(df["city"].dropna().unique().tolist())
    selected_city = st.selectbox("Select City", all_cities, label_visibility="collapsed")

    city_df = df[df["city"] == selected_city].copy()
    min_date = city_df["timestamp"].min().date()
    max_date = city_df["timestamp"].max().date()

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">📅 Date Range</div>', unsafe_allow_html=True)
    date_range = st.date_input("Date Range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date,
                               label_visibility="collapsed")

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
        city_df = city_df[
            (city_df["timestamp"].dt.date >= start_date) &
            (city_df["timestamp"].dt.date <= end_date)
        ]

    st.markdown('<div class="section-label" style="margin-top:1.2rem;">⚙️ Options</div>', unsafe_allow_html=True)
    show_raw = st.checkbox("Show Raw Data Table", value=False)

    if not city_df.empty:
        live_aqi = float(city_df.iloc[-1]["aqi"])
        live_cat, live_col, live_bg = aqi_meta(live_aqi)
        st.markdown(f"""
        <div style="margin-top:2rem; background:{live_bg}; border:1px solid {live_col}40;
                    border-radius:14px; padding:1rem; text-align:center;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem;
                        letter-spacing:3px; color:{live_col}; margin-bottom:6px;">
                <span class="live-dot"></span>LIVE AQI
            </div>
            <div style="font-family:'Orbitron',monospace; font-size:2.5rem;
                        font-weight:900; color:white; line-height:1;">
                {live_aqi:.0f}
            </div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:0.85rem;
                        color:{live_col}; margin-top:4px; font-weight:600;">
                {live_cat}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="footer">© AirAware Intelligence</div>', unsafe_allow_html=True)

# ---------------------------
# GUARD
# ---------------------------
if city_df.empty:
    st.warning("No data available for the selected filters.")
    st.stop()

latest_row   = city_df.iloc[-1]
latest_aqi   = float(latest_row["aqi"])
avg_aqi      = float(city_df["aqi"].mean())
max_aqi      = float(city_df["aqi"].max())
min_aqi      = float(city_df["aqi"].min())
cat, col, bg = aqi_meta(latest_aqi)

# ---------------------------
# HERO
# ---------------------------
st.markdown(f"""
<div class="hero-wrapper">
    <div class="hero-badge">🌍 &nbsp; REAL-TIME AIR QUALITY INTELLIGENCE PLATFORM</div>
    <div class="hero-title">AirAware</div>
    <div class="hero-sub">Advanced environmental monitoring · Predictive analytics · ML-powered insights</div>
    <div class="hero-city">▸ &nbsp; {selected_city.upper()} &nbsp; ◂</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

# ---------------------------
# KPI CARDS
# ---------------------------
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Current AQI", f"{latest_aqi:.0f}", delta=f"{latest_aqi - avg_aqi:+.1f} vs avg")
with c2:
    st.metric("Average AQI", f"{avg_aqi:.1f}")
with c3:
    st.metric("Peak AQI", f"{max_aqi:.0f}")
with c4:
    st.metric("Min AQI", f"{min_aqi:.0f}")

# ---------------------------
# ALERTS
# ---------------------------
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
st.markdown('<div class="section-title">🚨 AQI ALERT ENGINE</div>', unsafe_allow_html=True)

if latest_aqi <= 50:
    st.success(f"✅  Air quality is **Good** — minimal health risk. Enjoy outdoor activities freely.")
elif latest_aqi <= 100:
    st.info(f"🟡  Air quality is **Moderate** — acceptable; sensitive individuals should take caution.")
elif latest_aqi <= 150:
    st.warning(f"🟠  **Unhealthy for Sensitive Groups** — limit prolonged outdoor exertion if affected.")
elif latest_aqi <= 200:
    st.warning(f"🔴  Air quality is **Unhealthy** — everyone should reduce outdoor activity.")
elif latest_aqi <= 300:
    st.error(f"🟣  Air quality is **Very Unhealthy** — avoid outdoor exposure. Wear high-filtration mask.")
else:
    st.error(f"☠️  **HAZARDOUS** — health emergency. Stay indoors with air filtration.")

# ---------------------------
# GAUGE  +  TREND
# ---------------------------
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
left, right = st.columns([1, 1.3])

gauge_steps = [
    {"range": [0,   50],  "color": "rgba(0,255,136,0.12)"},
    {"range": [50,  100], "color": "rgba(245,230,66,0.12)"},
    {"range": [100, 150], "color": "rgba(255,140,66,0.12)"},
    {"range": [150, 200], "color": "rgba(255,77,79,0.12)"},
    {"range": [200, 300], "color": "rgba(180,79,255,0.12)"},
    {"range": [300, 500], "color": "rgba(255,34,85,0.12)"},
]

with left:
    st.markdown('<div class="section-title">🎯 LIVE AQI GAUGE</div>', unsafe_allow_html=True)
    gauge_fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=latest_aqi,
        delta={"reference": avg_aqi, "valueformat": ".0f",
               "font": {"size": 15, "color": "#7a9bbf"}},
        number={"font": {"size": 52, "color": "white", "family": "Orbitron"}},
        title={"text": f"<b>{cat}</b><br><span style='font-size:12px;color:{col}'>{selected_city}</span>",
               "font": {"size": 14, "color": "white", "family": "Orbitron"}},
        gauge={
            "axis": {"range": [0, 500], "tickcolor": "rgba(255,255,255,0.25)",
                     "tickwidth": 1, "tickfont": {"size": 9, "color": "#7a9bbf"}},
            "bar": {"color": col, "thickness": 0.25},
            "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
            "steps": gauge_steps,
            "threshold": {"line": {"color": "white", "width": 3}, "thickness": 0.8, "value": latest_aqi}
        }
    ))
    gauge_fig.update_layout(**dark_layout(height=440))
    gauge_fig.update_layout(annotations=[
        dict(x=0.15, y=0.05, text="<b>Good</b>", showarrow=False,
             font=dict(color="#00ff88", size=9, family="Share Tech Mono")),
        dict(x=0.5,  y=-0.06, text="<b>Moderate</b>", showarrow=False,
             font=dict(color="#f5e642", size=9, family="Share Tech Mono")),
        dict(x=0.85, y=0.05, text="<b>Hazardous</b>", showarrow=False,
             font=dict(color="#ff2255", size=9, family="Share Tech Mono")),
    ])
    st.plotly_chart(gauge_fig, use_container_width=True)

with right:
    st.markdown('<div class="section-title">📈 AQI TREND ANALYSIS</div>', unsafe_allow_html=True)
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=city_df["timestamp"], y=city_df["aqi"],
        fill='tozeroy', fillcolor="rgba(0,245,255,0.04)",
        line=dict(color="rgba(0,0,0,0)"), hoverinfo='skip', showlegend=False
    ))
    trend_fig.add_trace(go.Scatter(
        x=city_df["timestamp"], y=city_df["aqi"],
        mode='lines', name='AQI',
        line=dict(color=col, width=2.5, shape='spline', smoothing=0.8),
        hovertemplate="<b>%{y:.0f} AQI</b><br>%{x}<extra></extra>"
    ))
    for threshold, lbl, band_col, band_bg in AQI_LEVELS[::-1]:
        prev_val = [0, 50, 100, 150, 200, 300][[i for i,(t,l,c,b) in enumerate(AQI_LEVELS) if t==threshold][0]]
        trend_fig.add_hrect(y0=prev_val, y1=threshold,
                            fillcolor=band_bg.replace("0.12", "0.03"), line_width=0, layer="below")
    trend_fig.add_hline(y=avg_aqi, line_dash="dash",
                        line_color="rgba(255,255,255,0.2)", line_width=1,
                        annotation_text=f"Avg {avg_aqi:.0f}",
                        annotation_font_color="rgba(255,255,255,0.45)", annotation_font_size=10)
    trend_fig.update_layout(**dark_layout(height=440))
    trend_fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="AQI",
                            hovermode="x unified")
    st.plotly_chart(trend_fig, use_container_width=True)

# ---------------------------
# POLLUTANTS + WEATHER
# ---------------------------
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
t1, t2 = st.columns([1.1, 1])

with t1:
    st.markdown('<div class="section-title">☁️ POLLUTANT LEVELS</div>', unsafe_allow_html=True)
    pollutants  = ["PM2.5", "PM10", "NO₂", "SO₂", "O₃", "CO"]
    values      = [float(latest_row[k]) for k in ["pm25","pm10","no2","so2","o3","co"]]
    bar_colors  = ["#00f5ff","#7b2fff","#ff00cc","#ff6b35","#00ff88","#f5e642"]

    poll_fig = go.Figure()
    for p, v, c in zip(pollutants, values, bar_colors):
        poll_fig.add_trace(go.Bar(
            x=[p], y=[v], name=p,
            marker=dict(color=c, opacity=0.85, line=dict(color=c, width=1.5)),
            hovertemplate=f"<b>{p}</b>: %{{y:.2f}}<extra></extra>",
            width=0.55,
        ))
    poll_fig.update_layout(**dark_layout(height=420))
    poll_fig.update_layout(showlegend=False, barmode="group", bargap=0.25,
                           yaxis_title="Concentration (μg/m³)")
    st.plotly_chart(poll_fig, use_container_width=True)

with t2:
    st.markdown('<div class="section-title">🌡️ WEATHER CONDITIONS</div>', unsafe_allow_html=True)
    temp = float(latest_row["temperature"])
    hum  = float(latest_row["humidity"])
    wind = float(latest_row["wind_speed"])

    weather_fig = go.Figure()
    weather_fig.add_trace(go.Barpolar(
        r=[temp, hum, wind],
        theta=["Temperature °C", "Humidity %", "Wind m/s"],
        width=[120, 120, 120],
        marker_color=["#ff6b35","#00f5ff","#7b2fff"],
        marker_opacity=0.82,
        hovertemplate="<b>%{theta}</b>: %{r:.1f}<extra></extra>"
    ))
    weather_fig.update_layout(
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            angularaxis=dict(linecolor="rgba(0,245,255,0.2)", gridcolor="rgba(0,245,255,0.08)",
                             tickcolor="rgba(0,245,255,0.4)",
                             tickfont=dict(color="#7a9bbf", size=11, family="Share Tech Mono")),
            radialaxis=dict(linecolor="rgba(0,245,255,0.1)", gridcolor="rgba(0,245,255,0.06)",
                            tickfont=dict(color="#7a9bbf", size=9))
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c8d8f0", family="Rajdhani"),
        height=420, margin=dict(l=40,r=40,t=40,b=40),
        title=dict(text=f"Temp {temp:.1f}°C  ·  Humidity {hum:.0f}%  ·  Wind {wind:.1f} m/s",
                   font=dict(size=11, color="#7a9bbf", family="Share Tech Mono"), x=0.5)
    )
    st.plotly_chart(weather_fig, use_container_width=True)

# ---------------------------
# TABS
# ---------------------------
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📈  TREND ANALYSIS",
    "🧠  3D INTELLIGENCE",
    "🌍  SPATIAL MAP",
    "📊  FEATURE INSIGHTS",
    "🔮  AQI PREDICTOR",
    "🔁  MODEL RETRAINING"
])

# ── TAB 1 ──────────────────────────────────────────────────────────────────
with tab1:
    st.markdown('<div class="section-title" style="margin-top:1rem;">POLLUTANT TIME SERIES</div>', unsafe_allow_html=True)
    sub1, sub2 = st.columns(2)

    with sub1:
        f1 = go.Figure()
        f1.add_trace(go.Scatter(
            x=city_df["timestamp"], y=city_df["pm25"],
            fill='tozeroy', fillcolor="rgba(0,245,255,0.05)",
            line=dict(color="#00f5ff", width=2, shape='spline', smoothing=0.6),
            hovertemplate="<b>%{y:.1f} μg/m³</b><br>%{x}<extra>PM2.5</extra>"
        ))
        f1.update_layout(**dark_layout(height=380))
        f1.update_layout(title="PM 2.5 Trend", showlegend=False)
        st.plotly_chart(f1, use_container_width=True)

    with sub2:
        f2 = go.Figure()
        f2.add_trace(go.Scatter(
            x=city_df["timestamp"], y=city_df["pm10"],
            fill='tozeroy', fillcolor="rgba(123,47,255,0.06)",
            line=dict(color="#7b2fff", width=2, shape='spline', smoothing=0.6),
            hovertemplate="<b>%{y:.1f} μg/m³</b><br>%{x}<extra>PM10</extra>"
        ))
        f2.update_layout(**dark_layout(height=380))
        f2.update_layout(title="PM 10 Trend", showlegend=False)
        st.plotly_chart(f2, use_container_width=True)

    st.markdown('<div class="section-title">CORRELATION MATRIX</div>', unsafe_allow_html=True)
    corr = city_df.select_dtypes(include=np.number).corr()
    heatmap_fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.columns.tolist(),
        colorscale=[[0,"#0a0025"],[0.3,"#1a0050"],[0.5,"#2d006e"],[0.7,"#7b2fff"],[0.85,"#ff00cc"],[1,"#00f5ff"]],
        text=np.round(corr.values, 2), texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z:.3f}<extra></extra>",
        showscale=True,
        colorbar=dict(tickcolor="rgba(0,245,255,0.3)", tickfont=dict(color="#7a9bbf", size=10), outlinewidth=0)
    ))
    heatmap_fig.update_layout(**dark_layout(height=620))
    heatmap_fig.update_layout(title="Feature Correlation Heatmap")
    st.plotly_chart(heatmap_fig, use_container_width=True)

# ── TAB 2 ──────────────────────────────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-title" style="margin-top:1rem;">3D POLLUTION SCATTER</div>', unsafe_allow_html=True)
    scatter3d = px.scatter_3d(
        city_df, x="pm25", y="no2", z="o3",
        color="aqi", size="pm10",
        hover_data=["timestamp","aqi"],
        color_continuous_scale=[[0,"#00ff88"],[0.2,"#f5e642"],[0.4,"#ff8c42"],
                                 [0.6,"#ff4d4f"],[0.8,"#b44fff"],[1.0,"#ff2255"]],
        size_max=18, opacity=0.85,
    )
    scatter3d.update_traces(
        marker=dict(line=dict(width=0.5, color="rgba(255,255,255,0.2)")),
        hovertemplate="<b>PM2.5:</b> %{x:.1f}<br><b>NO₂:</b> %{y:.1f}<br><b>O₃:</b> %{z:.1f}<extra></extra>"
    )
    scatter3d.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="PM 2.5", gridcolor="rgba(0,245,255,0.08)", backgroundcolor="rgba(0,0,0,0)", color="#7a9bbf"),
            yaxis=dict(title="NO₂",   gridcolor="rgba(0,245,255,0.08)", backgroundcolor="rgba(0,0,0,0)", color="#7a9bbf"),
            zaxis=dict(title="O₃",    gridcolor="rgba(0,245,255,0.08)", backgroundcolor="rgba(0,0,0,0)", color="#7a9bbf"),
        ),
        font=dict(family="Rajdhani", color="#c8d8f0"),
        height=680, margin=dict(l=0,r=0,t=30,b=0),
        coloraxis_colorbar=dict(title="AQI", tickcolor="rgba(0,245,255,0.3)",
                                tickfont=dict(color="#7a9bbf",size=10), outlinewidth=0,
                                title_font=dict(color="#7a9bbf",size=11,family="Share Tech Mono"))
    )
    st.plotly_chart(scatter3d, use_container_width=True)

    # 3D Surface — AQI response over PM2.5 x PM10
    st.markdown('<div class="section-title">3D SURFACE — AQI RESPONSE SURFACE (PM2.5 × PM10)</div>', unsafe_allow_html=True)
    try:
        pm25_bins = np.linspace(city_df["pm25"].min(), city_df["pm25"].max(), 30)
        pm10_bins = np.linspace(city_df["pm10"].min(), city_df["pm10"].max(), 30)
        pm25_g, pm10_g = np.meshgrid(pm25_bins, pm10_bins)
        other_means = {f: city_df[f].mean() for f in FEATURES if f not in ["pm25","pm10"]}
        grid_df = pd.DataFrame({"pm25": pm25_g.ravel(), "pm10": pm10_g.ravel()})
        for f, v in other_means.items():
            grid_df[f] = v
        z_vals = model.predict(grid_df[FEATURES]).reshape(pm25_g.shape)

        surf_fig = go.Figure(go.Surface(
            x=pm25_bins, y=pm10_bins, z=z_vals,
            colorscale=[[0,"#00ff88"],[0.25,"#f5e642"],[0.5,"#ff8c42"],[0.75,"#ff4d4f"],[1.0,"#b44fff"]],
            opacity=0.88,
            contours=dict(z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)),
            hovertemplate="PM2.5: %{x:.1f}<br>PM10: %{y:.1f}<br>AQI: %{z:.1f}<extra></extra>"
        ))
        surf_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            scene=dict(
                bgcolor="rgba(0,0,0,0)",
                xaxis=dict(title="PM 2.5", color="#7a9bbf", gridcolor="rgba(0,245,255,0.08)", backgroundcolor="rgba(0,0,0,0)"),
                yaxis=dict(title="PM 10",  color="#7a9bbf", gridcolor="rgba(0,245,255,0.08)", backgroundcolor="rgba(0,0,0,0)"),
                zaxis=dict(title="AQI",    color="#7a9bbf", gridcolor="rgba(0,245,255,0.08)", backgroundcolor="rgba(0,0,0,0)"),
            ),
            font=dict(family="Rajdhani", color="#c8d8f0"),
            height=600, margin=dict(l=0,r=0,t=20,b=0),
        )
        st.plotly_chart(surf_fig, use_container_width=True)
    except Exception as e:
        st.info(f"Surface plot unavailable: {e}")

# ── TAB 3 ──────────────────────────────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-title" style="margin-top:1rem;">GLOBAL AQI GEOGRAPHIC MAP</div>', unsafe_allow_html=True)
    map_df = df.groupby(["city","country","latitude","longitude"], as_index=False).agg(
        {"aqi":"mean","pm25":"mean","pm10":"mean"}
    )
    try:
        map_fig = px.scatter_map(
            map_df, lat="latitude", lon="longitude",
            color="aqi", size="pm25", hover_name="city",
            hover_data={"country":True,"pm10":":.1f","aqi":":.0f","latitude":False,"longitude":False},
            zoom=1, height=680,
            color_continuous_scale=[[0,"#00ff88"],[0.2,"#f5e642"],[0.4,"#ff8c42"],
                                     [0.6,"#ff4d4f"],[0.8,"#b44fff"],[1.0,"#ff2255"]],
            size_max=22, opacity=0.9,
        )
        map_fig.update_layout(
            map_style="carto-darkmatter",
            paper_bgcolor="rgba(0,0,0,0)",
            margin={"r":0,"t":0,"l":0,"b":0},
            coloraxis_colorbar=dict(title="AQI", tickcolor="rgba(0,245,255,0.3)",
                                    tickfont=dict(color="#7a9bbf",size=10), outlinewidth=0,
                                    title_font=dict(color="#7a9bbf",size=11,family="Share Tech Mono"))
        )
    except Exception:
        map_fig = px.scatter_mapbox(
            map_df, lat="latitude", lon="longitude", color="aqi", size="pm25",
            hover_name="city", zoom=1, height=680,
            color_continuous_scale="turbo", mapbox_style="carto-darkmatter",
        )
        map_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(map_fig, use_container_width=True)

# ── TAB 4 ──────────────────────────────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-title" style="margin-top:1rem;">MODEL FEATURE IMPORTANCE</div>', unsafe_allow_html=True)
    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})\
                  .sort_values("Importance", ascending=True)
        fi_fig = go.Figure(go.Bar(
            x=fi_df["Importance"], y=fi_df["Feature"], orientation="h",
            marker=dict(color=fi_df["Importance"],
                        colorscale=[[0,"#0a0025"],[0.5,"#7b2fff"],[1,"#00f5ff"]],
                        line=dict(width=0), opacity=0.9),
            hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>",
            text=fi_df["Importance"].round(4), textposition="outside",
            textfont=dict(color="#7a9bbf", size=10, family="Share Tech Mono")
        ))
        fi_fig.update_layout(**dark_layout(height=460))
        fi_fig.update_layout(title="XGBoost Feature Importance", xaxis_title="Importance Score", showlegend=False)
        st.plotly_chart(fi_fig, use_container_width=True)

    st.markdown('<div class="section-title">AQI DISTRIBUTION</div>', unsafe_allow_html=True)
    dist_fig = px.histogram(
        city_df, x="aqi", nbins=50, color="AQI_Category",
        color_discrete_map={
            "Good":"#00ff88","Moderate":"#f5e642",
            "Unhealthy for Sensitive Groups":"#ff8c42",
            "Unhealthy":"#ff4d4f","Very Unhealthy":"#b44fff","Hazardous":"#ff2255"
        }, opacity=0.85,
    )
    dist_fig.update_layout(**dark_layout(height=420))
    dist_fig.update_layout(title="AQI Distribution by Category",
                           xaxis_title="AQI Value", yaxis_title="Frequency", bargap=0.05)
    st.plotly_chart(dist_fig, use_container_width=True)

    st.markdown('<div class="section-title">POLLUTANT VIOLIN DISTRIBUTIONS</div>', unsafe_allow_html=True)
    viol_cols   = ["pm25","pm10","no2","so2","o3","co"]
    viol_colors = ["#00f5ff","#7b2fff","#ff00cc","#ff6b35","#00ff88","#f5e642"]
    viol_fig = go.Figure()
    for vc, c in zip(viol_cols, viol_colors):
        viol_fig.add_trace(go.Violin(
            y=city_df[vc].dropna(), name=vc.upper(),
            box_visible=True, meanline_visible=True,
            line_color=c, opacity=0.8,
        ))
    viol_fig.update_layout(**dark_layout(height=450))
    viol_fig.update_layout(title="Pollutant Distribution Violins", showlegend=False)
    st.plotly_chart(viol_fig, use_container_width=True)

# ── TAB 5 ──────────────────────────────────────────────────────────────────
with tab5:
    st.markdown('<div class="section-title" style="margin-top:1rem;">AQI PREDICTION ENGINE</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; letter-spacing:2px;
                color:rgba(0,245,255,0.5); margin-bottom:1.2rem;">
        ENTER ENVIRONMENTAL PARAMETERS → XGBoost MODEL PREDICTS AQI
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1:
        st.markdown('<div class="section-label">Particulates</div>', unsafe_allow_html=True)
        pm25_i = st.number_input("PM 2.5 (μg/m³)", value=float(city_df["pm25"].mean()), step=0.1)
        pm10_i = st.number_input("PM 10  (μg/m³)", value=float(city_df["pm10"].mean()), step=0.1)
        no2_i  = st.number_input("NO₂   (μg/m³)", value=float(city_df["no2"].mean()),  step=0.1)
    with p2:
        st.markdown('<div class="section-label">Gases</div>', unsafe_allow_html=True)
        so2_i  = st.number_input("SO₂   (μg/m³)", value=float(city_df["so2"].mean()),  step=0.1)
        o3_i   = st.number_input("O₃    (μg/m³)", value=float(city_df["o3"].mean()),   step=0.1)
        co_i   = st.number_input("CO    (mg/m³)", value=float(city_df["co"].mean()),   step=0.01)
    with p3:
        st.markdown('<div class="section-label">Weather</div>', unsafe_allow_html=True)
        temp_i  = st.number_input("Temperature (°C)", value=float(city_df["temperature"].mean()), step=0.1)
        hum_i   = st.number_input("Humidity (%)",     value=float(city_df["humidity"].mean()),    step=0.1)
        wind_i  = st.number_input("Wind Speed (m/s)", value=float(city_df["wind_speed"].mean()),  step=0.1)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("⚡  PREDICT AQI NOW", use_container_width=True):
        input_df = pd.DataFrame([{
            "pm10": pm10_i, "pm25": pm25_i, "no2": no2_i, "so2": so2_i,
            "o3": o3_i, "co": co_i, "temperature": temp_i,
            "humidity": hum_i, "wind_speed": wind_i
        }])
        pred = float(model.predict(input_df)[0])
        pred_cat, pred_col, pred_bg = aqi_meta(pred)

        st.markdown(f"""
        <div style="background:{pred_bg}; border:1px solid {pred_col}50;
                    border-radius:16px; padding:1.5rem; text-align:center; margin:1rem 0;
                    box-shadow: 0 0 30px {pred_col}20;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                        letter-spacing:4px; color:{pred_col}; margin-bottom:0.5rem;">PREDICTED RESULT</div>
            <div style="font-family:'Orbitron',monospace; font-size:3.5rem;
                        font-weight:900; color:white; line-height:1;">{pred:.0f}</div>
            <div style="font-family:'Rajdhani',sans-serif; font-size:1.2rem;
                        color:{pred_col}; font-weight:600; margin-top:4px;">{pred_cat}</div>
        </div>
        """, unsafe_allow_html=True)

        pred_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred,
            number={"font": {"size": 48, "color": "white", "family": "Orbitron"}},
            title={"text": f"<b>Predicted AQI</b><br><span style='color:{pred_col};font-size:14px'>{pred_cat}</span>",
                   "font": {"size": 14, "color": "white", "family": "Orbitron"}},
            gauge={"axis": {"range": [0,500], "tickcolor": "rgba(255,255,255,0.25)"},
                   "bar": {"color": pred_col, "thickness": 0.25},
                   "bgcolor": "rgba(0,0,0,0)", "borderwidth": 0,
                   "steps": gauge_steps,
                   "threshold": {"line": {"color":"white","width":3},"thickness":0.8,"value":pred}}
        ))
        pred_gauge.update_layout(**dark_layout(height=430))
        st.plotly_chart(pred_gauge, use_container_width=True)

        st.markdown('<div class="section-title">INPUT FEATURE OVERVIEW</div>', unsafe_allow_html=True)
        inp_fig = go.Figure(go.Bar(
            x=["PM10","PM2.5","NO₂","SO₂","O₃","CO","Temp","Humidity","Wind"],
            y=[pm10_i, pm25_i, no2_i, so2_i, o3_i, co_i, temp_i, hum_i, wind_i],
            marker_color=["#00f5ff","#7b2fff","#ff00cc","#ff6b35","#00ff88","#f5e642",
                          "#ff4d4f","#00f5ff","#7b2fff"],
            opacity=0.85,
            hovertemplate="<b>%{x}</b>: %{y:.2f}<extra></extra>"
        ))
        inp_fig.update_layout(**dark_layout(height=360))
        inp_fig.update_layout(title="Your Input Parameters", showlegend=False, yaxis_title="Value")
        st.plotly_chart(inp_fig, use_container_width=True)

# ── TAB 6 ──────────────────────────────────────────────────────────────────
with tab6:
    st.markdown('<div class="section-title" style="margin-top:1rem;">AUTOMATED MODEL RETRAINING</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem; letter-spacing:2px;
                color:rgba(0,245,255,0.4); margin-bottom:1.5rem; line-height:2;">
        UPLOAD NEW CSV → VALIDATE SCHEMA → RETRAIN XGBOOST → SAVE MODEL<br>
        REQUIRED: pm10, pm25, no2, so2, o3, co, temperature, humidity, wind_speed, aqi
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV for Retraining", type=["csv"])
    if uploaded_file is not None:
        new_df = pd.read_csv(uploaded_file)
        st.markdown('<div class="section-title">DATASET PREVIEW</div>', unsafe_allow_html=True)
        st.dataframe(new_df.head(10), use_container_width=True)

        missing = set(FEATURES + ["aqi"]) - set(new_df.columns)
        if missing:
            st.error(f"❌  Missing columns: `{', '.join(missing)}`")
        else:
            r1, r2, r3 = st.columns(3)
            r1.metric("Rows",      f"{len(new_df):,}")
            r2.metric("Columns",   f"{len(new_df.columns)}")
            r3.metric("AQI Range", f"{new_df['aqi'].min():.0f} – {new_df['aqi'].max():.0f}")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🔁  RETRAIN & SAVE MODEL", use_container_width=True):
                with st.spinner("Training XGBoost model on new data..."):
                    new_df = new_df.drop_duplicates().ffill()
                    model.fit(new_df[FEATURES], new_df["aqi"])
                    with open(MODEL_PATH, "wb") as f:
                        pickle.dump(model, f)
                st.success("✅  Model retrained and saved successfully!")
                st.balloons()

                train_fig = px.histogram(new_df, x="aqi", nbins=40, opacity=0.8,
                                         color_discrete_sequence=["#00f5ff"])
                train_fig.update_layout(**dark_layout(height=360))
                train_fig.update_layout(title="Training Data AQI Distribution",
                                        xaxis_title="AQI", yaxis_title="Count")
                st.plotly_chart(train_fig, use_container_width=True)

# ---------------------------
# RAW DATA
# ---------------------------
if show_raw:
    st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 RAW DATA TABLE</div>', unsafe_allow_html=True)
    st.dataframe(city_df, use_container_width=True)

# ---------------------------
# FOOTER
# ---------------------------
st.markdown('<div class="neon-divider"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    AIRAWARE INTELLIGENCE PLATFORM &nbsp;·&nbsp; STREAMLIT · PLOTLY · XGBOOST · PANDAS
    <br><span style="opacity:0.4;">© 2025 · ALL SYSTEMS NOMINAL</span>
</div>
""", unsafe_allow_html=True)