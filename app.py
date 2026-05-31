"""Streamlit app — CrossFit Madrid Top Performer Predictor.

Single-page demo for the Module 2 project. Asks the user for a few details
about a hypothetical new gym and returns:

1. Probability of becoming a "top performer" in its district.
2. Top 3 actions (classes/services) that would most boost that probability.

Run from the folder that contains both `app.py` and the enriched Excel:

    streamlit run app.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CrossFit Madrid — Top Performer Predictor",
    page_icon="🏋️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# LOAD DATA + TRAIN MODEL (cached, runs once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def train_pipeline() -> dict:
    """Replicate the notebook feature engineering, train XGBoost, return
    everything the UI needs."""
    candidates = [
        Path("crossfit_gyms_madrid_enriched.xlsx"),
        Path("output/crossfit_gyms_madrid_enriched.xlsx"),
        Path(r"C:\Users\javie\Desktop\Ironhack\Semana14- Modulo 2 Project\crossfit_gyms_madrid_enriched.xlsx"),
    ]
    excel_path = next((p for p in candidates if p.exists()), None)
    if excel_path is None:
        st.error(
            "❌ Could not find **crossfit_gyms_madrid_enriched.xlsx**.\n\n"
            "Place it next to `app.py` and reload."
        )
        st.stop()

    df = pd.read_excel(excel_path, sheet_name="Gimnasios")

    # ---- NaN handling (same as the notebook) ----
    df["Rating Google"] = df["Rating Google"].fillna(df["Rating Google"].median())
    df["Clases impartidas"] = df["Clases impartidas"].fillna("Desconocido")
    df["Ejemplos de precios"] = df["Ejemplos de precios"].fillna("Desconocido")
    df["Servicios"] = df["Servicios"].fillna("Desconocido")
    df["Nº reseñas Google"] = df["Nº reseñas Google"].fillna(df["Nº reseñas Google"].median())

    # ---- Target ----
    global_p75 = df["Nº reseñas Google"].quantile(0.75)
    gym_per_district = df["Distrito"].value_counts()
    district_p75 = df.groupby("Distrito")["Nº reseñas Google"].quantile(0.75)
    df["umbral_reseñas"] = df["Distrito"].apply(
        lambda d: district_p75[d] if gym_per_district[d] >= 4 else global_p75
    )
    df["es_top_performer"] = (
        (df["Rating Google"] >= 4.8) & (df["Nº reseñas Google"] >= df["umbral_reseñas"])
    ).astype(int)

    # Cache info per district BEFORE we one-hot the column away
    district_info = df.groupby("Distrito").first()[[
        "Renta media (€/año)", "Edad mediana", "Población", "Densidad (hab/km²)",
        "Latitud", "Longitud",
    ]]
    district_n_gyms = gym_per_district.to_dict()

    # ---- Market analysis features ----
    df["log_densidad"] = np.log1p(df["Densidad (hab/km²)"])
    df["n_gimnasios_distrito"] = df["Distrito"].map(gym_per_district)
    df["gimnasios_por_10k_hab"] = df["n_gimnasios_distrito"] / (df["Población"] / 10000)
    df["precio_vs_renta"] = df["Precio mínimo €/mes"] / (df["Renta media (€/año)"] / 12)

    # ---- Ejemplos de precios features ----
    precios = (
        df["Ejemplos de precios"]
        .fillna("")
        .str.findall(r"\d+")
        .apply(lambda x: [int(p) for p in x])
    )
    df["precios_publicos"] = (precios.str.len() > 0).astype(int)
    df["rango_precios"] = precios.apply(lambda l: max(l) - min(l) if len(l) >= 2 else 0)
    df["tiene_dropins"] = precios.apply(lambda l: int(any(p < 50 for p in l)))
    df["tiene_premium"] = precios.apply(lambda l: int(any(p > 200 for p in l)))

    # ---- One-hot distrito ----
    df = pd.get_dummies(df, columns=["Distrito"], drop_first=True)

    # ---- Multi-hot Servicios ----
    serv_dummies = (
        df["Servicios"].fillna("").str.split(", ").explode()
        .str.get_dummies().groupby(level=0).max()
    )
    service_names = [c for c in serv_dummies.columns if c != "Desconocido"]
    df = pd.concat([df, serv_dummies], axis=1)

    # ---- Multi-hot Clases ----
    class_dummies = (
        df["Clases impartidas"].fillna("").str.split(", ").explode()
        .str.get_dummies().groupby(level=0).max()
    )
    class_names = [c for c in class_dummies.columns if c != "Desconocido"]
    df = pd.concat([df, class_dummies], axis=1)

    # Drop "Desconocido" duplicates + redundant
    df = df.drop(columns=["Desconocido"], errors="ignore")
    df = df.loc[:, ~df.columns.duplicated()]

    drop_cols = [
        "Nombre", "Dirección", "Teléfono", "Web",
        "Rating Google", "Nº reseñas Google", "umbral_reseñas",
        "Clases impartidas", "Servicios", "Ejemplos de precios",
    ]
    df = df.drop(columns=drop_cols, errors="ignore")

    # ---- Train ----
    X = df.drop("es_top_performer", axis=1)
    y = df["es_top_performer"]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    train = pd.concat([X_train, y_train], axis=1)
    tp = train[train["es_top_performer"] == 1]
    ntp = train[train["es_top_performer"] == 0]
    tp_over = resample(tp, replace=True, n_samples=len(ntp), random_state=0)
    train_over = pd.concat([ntp, tp_over])
    X_train_over = train_over.drop("es_top_performer", axis=1)
    y_train_over = train_over["es_top_performer"]

    model = xgboost.XGBClassifier(random_state=42)
    model.fit(X_train_over, y_train_over)

    return {
        "model": model,
        "feature_cols": X.columns.tolist(),
        "district_info": district_info,
        "district_n_gyms": district_n_gyms,
        "class_names": sorted(class_names),
        "service_names": sorted(service_names),
    }


# ---------------------------------------------------------------------------
# BUILD A FEATURE VECTOR FOR ONE GYM
# ---------------------------------------------------------------------------
def build_features(
    artifact: dict,
    district: str,
    price_min: float,
    price_max: float,
    classes_sel: list[str],
    services_sel: list[str],
) -> pd.DataFrame:
    row = {col: 0 for col in artifact["feature_cols"]}
    info = artifact["district_info"].loc[district]

    # Demographics (auto)
    row["Renta media (€/año)"] = info["Renta media (€/año)"]
    row["Edad mediana"] = info["Edad mediana"]
    row["Población"] = info["Población"]
    row["Densidad (hab/km²)"] = info["Densidad (hab/km²)"]
    row["Latitud"] = info["Latitud"]
    row["Longitud"] = info["Longitud"]

    # Prices
    row["Precio mínimo €/mes"] = price_min
    row["Precio máximo €/mes"] = price_max

    # Market features
    row["log_densidad"] = float(np.log1p(info["Densidad (hab/km²)"]))
    row["n_gimnasios_distrito"] = artifact["district_n_gyms"].get(district, 5)
    row["gimnasios_por_10k_hab"] = (
        row["n_gimnasios_distrito"] / (info["Población"] / 10000)
    )
    row["precio_vs_renta"] = price_min / (info["Renta media (€/año)"] / 12)

    # Price-detail defaults (new gym, public pricing)
    row["precios_publicos"] = 1
    row["rango_precios"] = max(0, price_max - price_min)
    row["tiene_dropins"] = 0
    row["tiene_premium"] = 1 if price_max > 200 else 0

    # One-hot district (drop_first means Arganzuela is the baseline)
    col = f"Distrito_{district}"
    if col in row:
        row[col] = 1

    # Classes & services
    for c in classes_sel:
        if c in row:
            row[c] = 1
    for s in services_sel:
        if s in row:
            row[s] = 1

    return pd.DataFrame([row])[artifact["feature_cols"]]


# ---------------------------------------------------------------------------
# RECOMMENDATIONS ENGINE
# ---------------------------------------------------------------------------
def get_recommendations(
    artifact: dict,
    baseline_X: pd.DataFrame,
    classes_sel: list[str],
    services_sel: list[str],
    top_k: int = 3,
) -> tuple[float, list[dict]]:
    """For every class/service NOT in the current setup, simulate adding it
    and measure the change in probability. Return the top_k biggest gains."""
    model = artifact["model"]
    baseline_prob = float(model.predict_proba(baseline_X)[0, 1])

    candidates: list[dict] = []
    for name in artifact["class_names"]:
        if name in classes_sel:
            continue
        modified = baseline_X.copy()
        if name in modified.columns:
            modified.loc[modified.index[0], name] = 1
        new_prob = float(model.predict_proba(modified)[0, 1])
        candidates.append({
            "kind": "Class", "name": name,
            "delta": new_prob - baseline_prob, "new_prob": new_prob,
        })
    for name in artifact["service_names"]:
        if name in services_sel:
            continue
        modified = baseline_X.copy()
        if name in modified.columns:
            modified.loc[modified.index[0], name] = 1
        new_prob = float(model.predict_proba(modified)[0, 1])
        candidates.append({
            "kind": "Service", "name": name,
            "delta": new_prob - baseline_prob, "new_prob": new_prob,
        })

    candidates.sort(key=lambda x: x["delta"], reverse=True)
    return baseline_prob, candidates[:top_k]


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
artifact = train_pipeline()
districts = sorted(artifact["district_info"].index.tolist())

st.title("🏋️ CrossFit Madrid — Top Performer Predictor")
st.markdown(
    "Estimate the probability that a new gym will become a **top performer** "
    "in its district, and discover the **best moves** to improve those odds."
)

st.divider()

col_form, col_result = st.columns([1, 1.2], gap="large")

with col_form:
    st.subheader("Gym Configuration")

    district = st.selectbox(
        "District",
        options=districts,
        index=districts.index("Salamanca") if "Salamanca" in districts else 0,
        help="District demographics are auto-loaded.",
    )

    price_min, price_max = st.slider(
        "Monthly price range (€)",
        min_value=50, max_value=250, value=(85, 150), step=5,
    )

    classes_sel = st.multiselect(
        "Classes offered",
        options=artifact["class_names"],
        default=["CrossFit"] if "CrossFit" in artifact["class_names"] else [],
    )

    services_sel = st.multiselect(
        "Services",
        options=artifact["service_names"],
        default=[],
    )

    predict = st.button("Predict", type="primary", use_container_width=True)


with col_result:
    if predict:
        baseline_X = build_features(
            artifact, district, price_min, price_max, classes_sel, services_sel
        )
        prob, recs = get_recommendations(
            artifact, baseline_X, classes_sel, services_sel
        )

        st.subheader("Prediction")
        m1, m2 = st.columns([1, 1.6])
        m1.metric("Top Performer probability", f"{prob:.1%}")

        if prob >= 0.5:
            m2.success("Strong prospect — go for it 🎯")
        elif prob >= 0.3:
            m2.warning("Decent prospect — boost it with the actions below ⚙️")
        else:
            m2.error("Challenging prospect — see actions below ⚠️")

        st.progress(min(prob, 1.0))

        st.divider()
        st.subheader("Top 3 actions to improve your chances")

        positive_recs = [r for r in recs if r["delta"] > 0]
        if not positive_recs:
            st.info(
                "Your current setup already looks strong — no single addition "
                "boosts the probability significantly."
            )
        else:
            for i, rec in enumerate(positive_recs, 1):
                delta_pp = rec["delta"] * 100
                st.markdown(
                    f"**{i}. Add {rec['kind'].lower()}: `{rec['name']}`**  →  "
                    f"new probability **{rec['new_prob']:.1%}**  "
                    f"(+{delta_pp:.1f} pp)"
                )
    else:
        st.info("Fill the form and click **Predict** to see the result.")


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.divider()
with st.expander("ℹ️  About this model"):
    st.markdown(
        """
        - **Dataset**: 236 CrossFit / functional gyms in Madrid scraped from
          Google Places, enriched with demographic data per district
          (median income, age, density, etc.).
        - **Target**: a gym is *top performer* when its Google rating is
          ≥ 4.8 **and** its number of reviews is ≥ the 75ᵗʰ percentile of
          its district.
        - **Model**: XGBoost classifier trained on 188 gyms (with random
          oversampling of the minority class) and evaluated on 48.
          Current recall ≈ 0.37 on the test set.
        - **Limitations**: only 40 top performers exist in total. The model
          should be used to suggest direction, not to give verdicts. The
          presentation outlines the scalability roadmap.
        """
    )
