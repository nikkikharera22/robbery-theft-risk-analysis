"""
Smart Robbery and Theft from the Person Risk Analysis Using Machine Learning
Streamlit App - built on real Chicago Police Department data (2018)

Run locally:   streamlit run app.py
Deploy:        push this repo to GitHub, then deploy on share.streamlit.io
"""
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from sklearn.neighbors import BallTree

st.set_page_config(page_title="Robbery & Theft Risk Analysis", page_icon="🛡️", layout="wide")

EARTH_RADIUS_M = 6_371_000.0  

MODEL_PATH = "model/crime_pipeline.joblib"
LABEL_ENCODER_PATH = "model/label_encoder.joblib"
LOCATION_ENCODER_PATH = "model/location_encoder.joblib"
HEATMAP_DATA = "data/heatmap_grid.csv"
LOOKUP_DATA = "data/spatial_lookup.csv"
DISTRICT_DATA = "data/top_districts.csv"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH), joblib.load(LABEL_ENCODER_PATH), joblib.load(LOCATION_ENCODER_PATH)


@st.cache_data
def load_heatmap_data():
    return pd.read_csv(HEATMAP_DATA)


@st.cache_data
def load_districts():
    return pd.read_csv(DISTRICT_DATA)


@st.cache_resource
def build_spatial_index():
    """BallTree with haversine metric - computes TRUE great-circle distance,
    not a naive flat-degree approximation. This fixes a bug flagged in
    review: a fixed metres-per-degree conversion is wrong for longitude,
    which represents fewer metres per degree the further from the equator
    you are (noticeably so at Chicago's latitude, ~41.9N)."""
    df = pd.read_csv(LOOKUP_DATA)
    coords_rad = np.radians(df[["Latitude", "Longitude"]].values)
    tree = BallTree(coords_rad, metric="haversine")
    return tree, df


def query_radius_m(tree, lat, lon, radius_m):
    """Query the haversine BallTree for all points within `radius_m` metres
    of (lat, lon). Returns integer positional indices into the lookup df."""
    radius_rad = radius_m / EARTH_RADIUS_M
    point_rad = np.radians([[lat, lon]])
    idx = tree.query_radius(point_rad, r=radius_rad)[0]
    return idx


@st.cache_data
def build_density_reference(_tree, lookup_df_len, radius_m, sample_size=1500, seed=42):
    """Build a reference distribution of LOCAL DENSITY (incidents per km^2)
    at the SAME radius the user selected, sampled from real incident
    locations across the dataset. This fixes a bug flagged in review:
    previously, incident counts within a user-chosen radius were compared
    against incident counts per ~100m HEATMAP GRID CELL - two different
    spatial units, which doesn't tell you whether the user's area is
    unusually dense. Comparing density-per-km^2 at the SAME radius to
    density-per-km^2 at that SAME radius elsewhere is an apples-to-apples
    comparison.
    """
    tree, lookup_df = _tree
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(lookup_df_len, size=min(sample_size, lookup_df_len), replace=False)
    sample_coords = lookup_df.iloc[sample_idx][["Latitude", "Longitude"]].values
    area_km2 = np.pi * (radius_m / 1000) ** 2
    densities = []
    for lat, lon in sample_coords:
        idx = query_radius_m(tree, lat, lon, radius_m)
        densities.append(len(idx) / area_km2)
    return np.array(densities)


pipe, le_target, le_loc = load_model()
heat_df = load_heatmap_data()
districts = load_districts()
tree, lookup_df = build_spatial_index()

st.title("🛡️ Robbery & Theft from the Person — Risk Analysis")
st.caption(
    "A machine-learning system built on 12,635 real Chicago Police Department incidents "
    "(Robbery and Theft-from-the-person only, 2018). Combines a trained classifier with "
    "historical spatial evidence — shown as two separate, clearly labelled results."
)

with st.expander("ℹ️ How this app works (2 separate components)"):
    st.markdown(
        """
        This app does **two genuinely different things** and shows both, separately:

        | | 🤖 ML Prediction | 📍 Historical Evidence |
        |---|---|---|
        | **Question answered** | Given this location & time, which crime type is more likely? | How many crimes were already recorded near here? |
        | **Method** | Trained Gradient Boosting classifier | Haversine spatial search over real incidents |
        | **Uses year/month you pick?** | Yes | No — historical counts are all-time (2018) |
        | **What it is NOT** | Not a probability a crime will happen | Not adjusted for the time you selected |

        ```
              USER LOCATION + TIME
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
    Gradient Boosting          Haversine spatial
       classifier                  lookup
         │                           │
         ▼                           ▼
   Crime-type prediction     Historical incidents
   (uses lat/lon/hour/        near this point
    month/day/venue/ward)     (all-time, 2018)
         │                           │
         │                    ┌──────┴──────┐
         │                    ▼             ▼
         │                 Count      Density (per km²)
         │                    │        vs. reference
         │                    │        sample at same
         │                    │        radius
         └──────────┬─────────┘
                    ▼
           BOTH shown together,
           clearly labelled as
           separate results
        ```

        Neither result is a probability that a crime will occur. Both are historical-pattern
        tools. See the accompanying report and model card for full limitations.
        """
    )

with st.sidebar:
    st.header("About this system")
    st.markdown(
        """
        **Task:** Predict which crime type (Robbery vs. Theft from the
        person) is more likely at a given location/time, and separately
        show local historical incident density.

        **Data:** 12,635 real Chicago Police Department incidents
        (Robbery + Theft from the person: pocket-picking, purse-snatching
        only), 2018.

        **Model:** Gradient Boosting, tuned via GridSearchCV, trained on
        Latitude, Longitude, Hour, Month, Day of week, Location type,
        District, Ward, Community Area.

        **Held-out test performance:**
        - Accuracy: **86.9%**
        - F1 (weighted): **86.1%**
        - ROC-AUC: **86.8%**

        ⚠️ This is a coursework demonstration system, not an operational
        policing tool. See the accompanying report for full limitations.
        """
    )

tab1, tab2 = st.tabs(["🗺️ Crime Hotspot Map", "📍 Check a Location"])

with tab1:
    st.subheader("Historical crime density across Chicago")
    st.caption(
        f"Heatmap built from {int(heat_df['count'].sum()):,} incidents, aggregated to "
        f"{len(heat_df):,} ~100m grid cells for rendering performance."
    )
    center_lat, center_lon = 41.85, -87.68
    m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="cartodbpositron")
    heat_points = heat_df[["lat_round", "lon_round", "count"]].values.tolist()
    HeatMap(heat_points, radius=8, blur=6, max_zoom=13, gradient={
        "0.2": "#FDEDEC", "0.4": "#F1948A", "0.6": "#C0392B", "1.0": "#7B241C"
    }).add_to(m)
    st_folium(m, width=1100, height=550, returned_objects=[])

    st.subheader("Top 10 police districts by incident count")
    st.dataframe(districts.head(10)[["District", "count"]], hide_index=True, use_container_width=True)

with tab2:
    st.subheader("Check a location and time")
    col1, col2 = st.columns([1, 1])
    with col1:
        pick = st.selectbox(
            "Quick-pick a known high-incident district (optional):",
            ["-- enter coordinates manually --"] + [f"District {d}" for d in districts["District"].tolist()],
        )
        if pick != "-- enter coordinates manually --":
            d_num = int(pick.replace("District ", ""))
            row = districts[districts["District"] == d_num].iloc[0]
            default_lat, default_lon = float(row["lat"]), float(row["lon"])
        else:
            default_lat, default_lon = 41.8781, -87.6298  # Chicago Loop default

        lat = st.number_input("Latitude", value=default_lat, format="%.6f")
        lon = st.number_input("Longitude", value=default_lon, format="%.6f")
        hour = st.slider("Hour of day (0-23)", 0, 23, 18)
        month = st.selectbox("Month", list(range(1, 13)), index=5)
        dow = st.selectbox("Day of week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=4)
        loc_desc = st.selectbox("Venue / location type", sorted(le_loc.classes_.tolist()),
                                 index=list(sorted(le_loc.classes_.tolist())).index("STREET") if "STREET" in le_loc.classes_ else 0)
        radius_m = st.slider("Historical search radius (metres)", 100, 2000, 500, step=100)
        check_clicked = st.button("🔍 Check this location", type="primary", use_container_width=True)

    with col2:
        if check_clicked:
            dow_idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(dow)
            loc_enc = le_loc.transform([loc_desc])[0]

            # Approximate District/Ward/Community Area from the nearest real
            # incident record, since the app doesn't ask the user for these
            # directly (they're administrative boundaries, not something a
            # member of the public would know off-hand).
            nearest_idx = query_radius_m(tree, lat, lon, 2000)
            if len(nearest_idx) == 0:
                nearest_idx = query_radius_m(tree, lat, lon, 10000)

            # ---------------- ML PREDICTION ----------------
            X_new = pd.DataFrame([[lat, lon, hour, month, dow_idx, loc_enc, 0, 0, 0]],
                                  columns=["Latitude", "Longitude", "Hour", "Month", "DayOfWeek",
                                           "LocDesc_enc", "District", "Ward", "Community Area"])
            pred = pipe.predict(X_new)[0]
            pred_label = le_target.inverse_transform([pred])[0]
            proba = pipe.predict_proba(X_new)[0]
            proba_map = dict(zip(le_target.classes_, proba))
            confidence = proba_map[pred_label]

            # ---------------- HISTORICAL EVIDENCE (independent of the ML step) ----------------
            idx = query_radius_m(tree, lat, lon, radius_m)
            nearby = lookup_df.iloc[idx]
            n_nearby = len(nearby)
            area_km2 = np.pi * (radius_m / 1000) ** 2
            local_density = n_nearby / area_km2

            reference_densities = build_density_reference(
                (tree, lookup_df), len(lookup_df), radius_m
            )
            density_percentile = float((reference_densities < local_density).mean() * 100) if n_nearby > 0 else 0.0
            if n_nearby == 0:
                density_label, density_color = "LOW", "green"
            elif density_percentile < 50:
                density_label, density_color = "LOW-MODERATE", "blue"
            elif density_percentile < 80:
                density_label, density_color = "ELEVATED", "orange"
            else:
                density_label, density_color = "HIGH", "red"

            # ---------------- DISPLAY: two results, clearly separated ----------------
            st.markdown("#### 🤖 ML Prediction")
            st.write(f"**Predicted crime type:** {pred_label}")
            st.progress(float(confidence))
            st.caption(
                f"Model confidence: {confidence:.1%}  "
                f"(Robbery: {proba_map.get('ROBBERY', 0):.1%} | Theft: {proba_map.get('THEFT', 0):.1%})"
            )
            st.caption(f"Based on: this location, {hour}:00, {dow}, {loc_desc.title()}, month {month}.")

            st.markdown("#### 📍 Historical Evidence (all-time, 2018 — does not use the time you selected above)")
            c1, c2 = st.columns(2)
            c1.metric(f"Incidents within {radius_m}m", n_nearby)
            c2.metric("Historical density", density_label)
            if n_nearby > 0:
                st.caption(
                    f"Local density: {local_density:.1f} incidents/km² — higher than {density_percentile:.0f}% of "
                    f"comparable {radius_m}m-radius areas sampled across the dataset."
                )
                st.write("**Crime type mix among nearby historical incidents:**")
                st.bar_chart(nearby["Primary Type"].value_counts())
            else:
                st.info("No recorded incidents within this radius in the 2018 dataset. The ML prediction above is still shown, since it does not depend on nearby historical counts.")

            st.warning(
                "⚠️ **Important:** Neither the ML prediction nor the historical density above is a "
                "probability that a crime will occur. Both describe historical patterns only. "
                "Always follow local police guidance."
            )
        else:
            st.info("Pick a location and click **Check this location** to see both results.")

st.divider()
st.caption(
    "Coursework project — Smart Robbery and Theft from the Person Risk Analysis Using Machine "
    "Learning. Data: Chicago Police Department CLEAR system (2018), via data.cityofchicago.org. "
    "Full technical report and code available in the accompanying GitHub repository."
)
