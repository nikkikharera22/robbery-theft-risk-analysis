# Smart Robbery and Theft from the Person Risk Analysis Using Machine Learning

A machine-learning system that analyses real Chicago Police Department crime data to
predict crime type (Robbery vs. Theft from the person) by location and time, and
separately shows historical crime density evidence via an interactive Streamlit app.

- **Problem:** predict which crime type is more likely at a given location/time, and
  separately surface historical incident density near that point
- **Data:** 12,635 real, police-recorded incidents (Robbery + Theft from the person —
  specifically pocket-picking and purse-snatching, the two person-to-person theft
  subtypes) — Chicago, 2018
- **Approach:** Logistic Regression baseline (~72%) → 5-model comparison (Logistic
  Regression, Decision Tree, Random Forest, Gradient Boosting, KNN), every model tuned
  under an identical GridSearchCV protocol on the full training set → final model =
  **tuned Gradient Boosting**
- **Held-out test performance:** Accuracy 86.9% · F1 (weighted) 86.1% · ROC-AUC 86.8% —
  a genuine, verified improvement over the ~72% baseline, driven by richer features
  (venue type, hour of day) that a purely geographic feature set (latitude/longitude/
  year/month alone) cannot capture




## Repository structure

```
.
├── app.py                       # Streamlit app (deployment entry point)
├── train.py                     # full pipeline: feature engineering → tuning → final model
├── requirements.txt             # minimal deps for the deployed Streamlit app
├── requirements-dev.txt         # extra deps needed to re-run train.py / the notebook
├── .streamlit/
│   └── config.toml              # native Streamlit theme (maroon accent, matches report)
├── data/
│   ├── feature_engineered_dataset.csv   # cleaned + feature-engineered data (3.5MB)
│   ├── spatial_lookup.csv               # per-incident lat/lon/type/date (used for historical lookups)
│   ├── heatmap_grid.csv                 # ~100m-grid-aggregated density (used by the map)
│   └── top_districts.csv                # top districts by incident count (quick-pick dropdown)
├── model/
│   ├── crime_pipeline.joblib            # fitted Gradient Boosting pipeline
│   ├── label_encoder.joblib             # crime-type label encoder
│   └── location_encoder.joblib          # venue-type (Location Description) encoder
├── figures/                     # all generated plots + results.json (used in the report)
└── notebook/
    └── crime_analysis.ipynb     # full, executed development notebook
```

**Note on the raw dataset:** the original `crimes_2018.csv` (63MB, all Chicago 2018
crime records, not just Robbery/Theft) is **not included** — it exceeds GitHub's 25MB
web-upload limit and isn't needed by the deployed app. `feature_engineered_dataset.csv`
(the already-filtered, feature-engineered version used for training, 3.5MB) **is**
included, so `train.py` is fully reproducible from this repo alone.

## Run locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Reproduce training (optional)

```bash
pip install -r requirements-dev.txt
python train.py
```

## Deploy to Streamlit Community Cloud

1. Push this folder as a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub.
3. **Create app** → **Deploy a public app from GitHub**.
4. Select this repo, branch `main`, **Main file path: `app.py`**.
5. Deploy.

## Data source

Chicago Police Department CLEAR (Citizen Law Enforcement Analysis and Reporting)
system, published via the City of Chicago's official open data portal
(data.cityofchicago.org). This project uses the 2018 calendar-year extract. Used here
for coursework purposes only.
