import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# Page Configuration
# =========================================================
st.set_page_config(
    page_title="🚲 Bike Rebalancing Gap Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# Model Definition
# =========================================================
HASH_BUCKETS = {
    'start_station_id': 10000, 'start_h3_9': 5000, 'start_borough': 10,
    'start_mode_landuse': 50, 'start_mode_zipcode': 300,
    'weather_code_wmo_code': 40, 'holiday_name': 30,
    'hour': 24, 'day_of_week': 7, 'month': 13
}
NUM_LEN = 19
num_cols_list = [
    'start_lat', 'start_lng', 'temperature_2m_C', 'apparent_temperature_C',
    'relative_humidity_2m_pct', 'precipitation_mm', 'rain_mm', 'snowfall_cm',
    'wind_speed_10m_kmh', 'wind_gusts_10m_kmh', 'cloud_cover_pct',
    'restaurants_in_start_h3_9', 'avg_score_in_start_h3_9',
    'hex_mta_entries', 'hex_mta_exits', 'start_avg_numfloors',
    'start_sum_unitsres', 'start_sum_unitstotal', 'is_school_holiday'
]

class DemandPredictionModel(nn.Module):
    def __init__(self, cat_dims, num_features_len):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_classes, max(2, min(50, num_classes // 2)))
            for num_classes in cat_dims.values()
        ])
        total_emb_dim = sum(max(2, min(50, num_classes // 2)) for num_classes in cat_dims.values())
        self.fc_layers = nn.Sequential(
            nn.Linear(total_emb_dim + num_features_len, 1024),
            nn.BatchNorm1d(1024), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, 1)
        )
    def forward(self, cat_x, num_x):
        emb_outs = [emb(cat_x[:, i]) for i, emb in enumerate(self.embeddings)]
        return self.fc_layers(torch.cat(emb_outs + [num_x], dim=1))

# =========================================================
# Load Model and Data (cached, loaded only once)
# =========================================================
@st.cache_resource
def load_model():
    model = DemandPredictionModel(HASH_BUCKETS, NUM_LEN)
    model.load_state_dict(torch.load(
        "net_flow_model_v3_mse_BEST.pth", map_location="cpu"
    ))
    model.eval()
    return model

@st.cache_data
def load_data():
    return pd.read_csv("full_dashboard_8am.csv")

model = load_model()
df_base = load_data()

# =========================================================
# Inference Function
# =========================================================
def run_inference(df):
    """Run inference on each row of the DataFrame, returning predicted_gap."""
    # Hash-encode categorical features
    cat_idx_cols = []
    for c, b in HASH_BUCKETS.items():
        name = c + "_idx"
        if c in df.columns:
            df[name] = df[c].apply(
                lambda x: abs(hash(str(x) if pd.notna(x) else "u")) % b
            )
        else:
            df[name] = 0
        cat_idx_cols.append(name)

    # Z-score normalize continuous features
    scaled_vals = []
    for c in num_cols_list:
        if c in df.columns:
            col = df[c].fillna(0.0)
            m, s = col.mean(), col.std()
            s = s if s and s != 0 else 1.0
            scaled_vals.append(((col - m) / s).values)
        else:
            scaled_vals.append(np.zeros(len(df)))

    with torch.no_grad():
        c_x = torch.tensor(df[cat_idx_cols].values, dtype=torch.long)
        n_x = torch.tensor(np.stack(scaled_vals, axis=1), dtype=torch.float32)
        preds = model(c_x, n_x).numpy().flatten()

    return preds

# =========================================================
# Sidebar: Feature Control Panel
# =========================================================
st.sidebar.title("🎛️ Feature Control Panel")
st.sidebar.markdown("Adjust the parameters below — the map updates predictions in real time.")

st.sidebar.subheader("🕐 Time")
hour = st.sidebar.slider("Hour", 0, 23, 8)
day_of_week = st.sidebar.selectbox(
    "Day of Week", [1, 2, 3, 4, 5, 6, 7],
    format_func=lambda x: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][x - 1]
)
is_holiday = st.sidebar.checkbox("Public Holiday", value=False)

st.sidebar.subheader("🌤️ Weather")
temperature = st.sidebar.slider("Temperature (°C)", -10.0, 40.0,
    float(df_base["temperature_2m_C"].mean()), step=0.5)
precipitation = st.sidebar.slider("Precipitation (mm)", 0.0, 50.0,
    float(df_base["precipitation_mm"].mean()), step=0.5)
wind_speed = st.sidebar.slider("Wind Speed (km/h)", 0.0, 80.0,
    float(df_base["wind_speed_10m_kmh"].mean()), step=1.0)
humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 60.0, step=1.0)
cloud_cover = st.sidebar.slider("Cloud Cover (%)", 0.0, 100.0, 50.0, step=1.0)

st.sidebar.subheader("🚇 MTA Subway Ridership")
mta_entries = st.sidebar.slider("Entries", 0, 5000,
    int(df_base["hex_mta_entries"].mean()), step=50)
mta_exits = st.sidebar.slider("Exits", 0, 5000,
    int(df_base["hex_mta_exits"].mean()), step=50)

st.sidebar.subheader("🏘️ Filter by Borough")
all_boroughs = sorted(df_base["start_borough"].dropna().unique().tolist())
selected_boroughs = st.sidebar.multiselect(
    "Show Boroughs", all_boroughs, default=all_boroughs
)

# =========================================================
# Build Prediction Input
# =========================================================
df_input = df_base.copy()

# Override weather and MTA features with sidebar values (applied uniformly to all stations)
df_input["temperature_2m_C"] = temperature
df_input["apparent_temperature_C"] = temperature - 2
df_input["precipitation_mm"] = precipitation
df_input["rain_mm"] = precipitation
df_input["wind_speed_10m_kmh"] = wind_speed
df_input["wind_gusts_10m_kmh"] = wind_speed * 1.3
df_input["relative_humidity_2m_pct"] = humidity
df_input["cloud_cover_pct"] = cloud_cover
df_input["hex_mta_entries"] = mta_entries
df_input["hex_mta_exits"] = mta_exits
df_input["is_school_holiday"] = int(is_holiday)
df_input["hour"] = hour
df_input["day_of_week"] = day_of_week
df_input["month"] = 12
df_input["snowfall_cm"] = 0.0
df_input["rain_mm"] = precipitation

# Borough filter
df_input = df_input[df_input["start_borough"].isin(selected_boroughs)]

# Run inference
df_input["predicted_gap"] = run_inference(df_input.copy())

# Status classification
def classify(gap):
    if gap < -3:   return "🔴 Severe Shortage"
    elif gap < 0:  return "🟠 Mild Shortage"
    elif gap < 3:  return "🟢 Balanced"
    elif gap < 7:  return "🔵 Mild Surplus"
    else:          return "🟣 Severe Surplus"

df_input["status"] = df_input["predicted_gap"].apply(classify)

color_map = {
    "🔴 Severe Shortage": "#d73027",
    "🟠 Mild Shortage":   "#fc8d59",
    "🟢 Balanced":        "#91cf60",
    "🔵 Mild Surplus":    "#4575b4",
    "🟣 Severe Surplus":  "#313695"
}

# =========================================================
# Main Page
# =========================================================
st.title("🚲 Bike-Share Rebalancing Gap Prediction Dashboard")
st.caption(f"Time: 08:00 AM Rush Hour | Stations: {len(df_input)} | Model: V3 MSE BEST")

# KPI metric cards
col1, col2, col3, col4 = st.columns(4)
col1.metric("🔴 Shortage Stations", f"{(df_input['predicted_gap'] < 0).sum()}")
col2.metric("🔵 Surplus Stations",  f"{(df_input['predicted_gap'] >= 3).sum()}")
col3.metric("🟢 Balanced Stations",
    f"{((df_input['predicted_gap'] >= 0) & (df_input['predicted_gap'] < 3)).sum()}")
col4.metric("City-wide Avg Gap", f"{df_input['predicted_gap'].mean():.2f}")

st.divider()

# Map + charts
left, right = st.columns([2, 1])

with left:
    st.subheader("📍 City-wide Station Rebalancing Gap Map")
    fig_map = px.scatter_mapbox(
        df_input,
        lat="start_lat", lon="start_lng",
        color="status",
        color_discrete_map=color_map,
        size=df_input["predicted_gap"].abs() + 2,
        size_max=18,
        hover_name="start_station_id",
        hover_data={
            "predicted_gap": ":.2f",
            "start_borough": True,
            "temperature_2m_C": ":.1f",
            "hex_mta_entries": ":.0f",
            "start_lat": False,
            "start_lng": False,
        },
        mapbox_style="carto-positron",
        zoom=10,
        center={
            "lat": df_input["start_lat"].mean(),
            "lon": df_input["start_lng"].mean()
        },
        height=500
    )
    fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0}, legend_title="Station Status")
    st.plotly_chart(fig_map, use_container_width=True)

with right:
    st.subheader("📊 Gap Summary by Borough")
    borough_summary = df_input.groupby("start_borough").agg(
        avg_gap=("predicted_gap", "mean"),
        count=("predicted_gap", "count")
    ).reset_index().sort_values("avg_gap")

    fig_bar = px.bar(
        borough_summary,
        x="avg_gap", y="start_borough",
        orientation="h",
        color="avg_gap",
        color_continuous_scale="RdYlGn",
        text=borough_summary["avg_gap"].apply(lambda x: f"{x:.2f}"),
        labels={"avg_gap": "Avg Gap", "start_borough": "Borough"},
        height=250
    )
    fig_bar.update_layout(coloraxis_showscale=False, margin={"t":10})
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📈 Gap Distribution")
    fig_hist = px.histogram(
        df_input, x="predicted_gap",
        nbins=25, color="start_borough",
        labels={"predicted_gap": "Predicted Gap"},
        height=220, barmode="overlay", opacity=0.7
    )
    fig_hist.add_vline(x=0, line_dash="dash", line_color="black")
    fig_hist.update_layout(margin={"t":10}, legend_title="Borough")
    st.plotly_chart(fig_hist, use_container_width=True)

# Bottom data table
st.divider()
st.subheader("📋 Station Detail (sorted by shortage first)")
table_df = (
    df_input[["start_station_id", "start_borough", "start_mode_landuse",
              "start_lat", "start_lng", "predicted_gap", "status"]]
    .sort_values("predicted_gap")
    .reset_index(drop=True)
)
st.dataframe(table_df, use_container_width=True, height=300)
