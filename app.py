"""Sleep Doctor — interactive demo of the project's two Linear Regression models.

Loads pre-trained model parameters from models.json (see train.py) rather than
fitting on startup, so the app's predictions are pinned to whatever was fit and
validated offline. Mirrors notebooks/05_model_refinements.ipynb: Path A
(research-question) vs. the lifestyle/context extension.
"""

import json

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from model_config import (
    ARTIFACT_PATH,
    CATEGORY_OPTIONS,
    EXTENSION_CATEGORICAL,
    PRIMARY_FEATURES,
    to_bucket,
)

FRIENDLY_BASE = {
    "age": "Age",
    "stress_score": "Stress level",
    "steps_that_day": "Steps that day",
    "exercise_day": "Exercised today",
    "work_hours_that_day": "Work hours",
    "shift_work": "Shift work",
    "caffeine_mg_before_bed": "Caffeine before bed",
    "alcohol_units_before_bed": "Alcohol before bed",
    "screen_time_before_bed_mins": "Screen time before bed",
    "bmi": "BMI",
    "nap_duration_mins": "Nap duration",
    "chronotype": "Chronotype",
    "mental_health_condition": "Mental health",
    "day_type": "Day type",
    "season": "Season",
    "gender": "Gender",
    "occupation": "Occupation",
}

# Diverging pair from the project's chart palette: blue = raises the score, red = lowers it.
POSITIVE_COLOR = "#2a78d6"
NEGATIVE_COLOR = "#e34948"
STATUS_COLORS = {"Low": "#d03b3b", "Medium": "#fab219", "High": "#0ca30c"}
STATUS_ICONS = {"Low": "⚠️", "Medium": "●", "High": "✓"}


def dummy_label(dummy_col: str) -> str:
    for base in EXTENSION_CATEGORICAL:
        prefix = base + "_"
        if dummy_col.startswith(prefix):
            return f"{FRIENDLY_BASE[base]}: {dummy_col[len(prefix):]}"
    return dummy_col


def friendly_primary_label(feature: str) -> str:
    return FRIENDLY_BASE.get(feature, feature)


@st.cache_resource
def load_models():
    try:
        with open(ARTIFACT_PATH) as f:
            artifact = json.load(f)
    except FileNotFoundError:
        st.error(
            f"Missing `{ARTIFACT_PATH}` — this app serves a pre-trained model rather than "
            "training on startup. Run `python train.py` once to generate it, then reload."
        )
        st.stop()

    def unpack(section):
        return {
            "coefs": pd.Series(section["coefficients"]),
            "intercept": section["intercept"],
            "means": pd.Series(section["means"]),
            "r2": section["test_r2"],
            "acc": section["test_accuracy"],
        }

    return {
        "primary": unpack(artifact["primary"]),
        "extension": {**unpack(artifact["extension"]), "columns": artifact["extension"]["columns"]},
    }


def predict_with_breakdown(coefs: pd.Series, intercept: float, means: pd.Series, x_row: pd.Series):
    """Predicted score + per-feature contribution vs. the training-set average person."""
    coefs = coefs.reindex(means.index)
    x_row = x_row.reindex(means.index).fillna(0.0)
    contributions = coefs * (x_row - means)
    pred = float(intercept + coefs.dot(x_row))
    return pred, contributions


ROW_HEIGHT = 34
BAR_THICKNESS = 20
LABEL_INK = "#52514e"


def contribution_chart(contributions: pd.Series, label_fn, top_n: int = 8) -> alt.Chart:
    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    ranked = ranked[ranked.abs() > 1e-6].head(top_n)
    plot_df = pd.DataFrame({
        "feature": [label_fn(f) for f in ranked.index],
        "contribution": ranked.values,
    })
    plot_df["direction"] = np.where(
        plot_df["contribution"] >= 0, "Raises predicted score", "Lowers predicted score"
    )
    order = plot_df["feature"].tolist()

    # Pad the x-domain so tip labels never get clipped at the chart edge.
    max_abs = max(plot_df["contribution"].abs().max(), 0.05) * 1.35
    y_enc = alt.Y(
        "feature:N",
        title=None,
        sort=order,
        scale=alt.Scale(paddingInner=0.4, paddingOuter=0.3),
        axis=alt.Axis(labelLimit=210, labelPadding=8, domain=False, ticks=False, labelFontSize=12),
    )

    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#c3c2b7").encode(x="x:Q")
    bars = (
        alt.Chart(plot_df)
        .mark_bar(size=BAR_THICKNESS, cornerRadiusEnd=4)
        .encode(
            x=alt.X(
                "contribution:Q",
                title="Effect on predicted score (points)",
                scale=alt.Scale(domain=[-max_abs, max_abs]),
            ),
            y=y_enc,
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(
                    domain=["Raises predicted score", "Lowers predicted score"],
                    range=[POSITIVE_COLOR, NEGATIVE_COLOR],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("contribution:Q", title="Effect (pts)", format="+.2f"),
            ],
        )
    )

    pos_df = plot_df[plot_df["contribution"] >= 0]
    neg_df = plot_df[plot_df["contribution"] < 0]
    pos_labels = (
        alt.Chart(pos_df)
        .mark_text(align="left", dx=6, fontSize=11, color=LABEL_INK)
        .encode(x="contribution:Q", y=y_enc, text=alt.Text("contribution:Q", format="+.2f"))
    )
    neg_labels = (
        alt.Chart(neg_df)
        .mark_text(align="right", dx=-6, fontSize=11, color=LABEL_INK)
        .encode(x="contribution:Q", y=y_enc, text=alt.Text("contribution:Q", format="+.2f"))
    )

    chart_height = ROW_HEIGHT * len(plot_df) + 20
    return (
        (zero_rule + bars + pos_labels + neg_labels)
        .properties(height=chart_height, padding={"left": 5, "right": 30, "top": 5, "bottom": 5})
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False)
    )


st.set_page_config(page_title="Sleep Doctor", page_icon="\U0001fa7a", layout="centered")
models = load_models()

st.title("\U0001fa7a\U0001f319 Sleep Doctor")
st.caption(
    "Based on age, stress, and activity level, can we predict sleep quality? "
    "An interactive demo of the project's Linear Regression models."
)

st.sidebar.header("Inputs")
mode = st.sidebar.radio(
    "Model",
    ["Research question (3 factors)", "Full lifestyle (extension)"],
    help="Primary mirrors the research question (age, stress, activity). "
         "Extension adds every lifestyle/context factor from the notebook's follow-up analysis.",
)
is_extension = mode == "Full lifestyle (extension)"

st.sidebar.subheader("Core factors")
age = st.sidebar.slider("Age", 18, 69, 35)
stress_score = st.sidebar.slider("Stress level (1 = calm, 10 = maxed out)", 1, 10, 5)
steps_that_day = st.sidebar.slider("Steps that day", 500, 20000, 7500, step=250)
exercise_day = st.sidebar.checkbox("Exercised today", value=False)

extension_inputs = {}
if is_extension:
    with st.sidebar.expander("Advanced: lifestyle & context", expanded=True):
        extension_inputs["work_hours_that_day"] = st.slider("Work hours that day", 0.0, 18.0, 7.0, 0.5)
        extension_inputs["shift_work"] = st.checkbox("Works a shift/night schedule")
        extension_inputs["caffeine_mg_before_bed"] = st.slider("Caffeine before bed (mg)", 0, 400, 40, 10)
        extension_inputs["alcohol_units_before_bed"] = st.slider("Alcohol before bed (units)", 0.0, 6.0, 0.5, 0.5)
        extension_inputs["screen_time_before_bed_mins"] = st.slider("Screen time before bed (min)", 0, 180, 60, 5)
        extension_inputs["bmi"] = st.slider("BMI", 16.0, 45.0, 26.0, 0.5)
        extension_inputs["nap_duration_mins"] = st.slider("Nap duration (min)", 0, 116, 15, 5)
        extension_inputs["chronotype"] = st.selectbox("Chronotype", CATEGORY_OPTIONS["chronotype"])
        extension_inputs["mental_health_condition"] = st.selectbox(
            "Mental health", CATEGORY_OPTIONS["mental_health_condition"]
        )
        extension_inputs["day_type"] = st.selectbox("Day type", CATEGORY_OPTIONS["day_type"])
        extension_inputs["season"] = st.selectbox("Season", CATEGORY_OPTIONS["season"])
        extension_inputs["gender"] = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])
        extension_inputs["occupation"] = st.selectbox("Occupation", CATEGORY_OPTIONS["occupation"])

raw_inputs = {
    "age": age,
    "stress_score": stress_score,
    "steps_that_day": steps_that_day,
    "exercise_day": int(exercise_day),
}

if is_extension:
    raw_inputs.update(extension_inputs)
    raw_inputs["shift_work"] = int(extension_inputs["shift_work"])
    input_row = pd.DataFrame([raw_inputs])
    dummies = pd.get_dummies(input_row, columns=EXTENSION_CATEGORICAL)
    extension = models["extension"]
    x_row = dummies.iloc[0].reindex(extension["columns"]).fillna(0.0)
    pred_score, contributions = predict_with_breakdown(
        extension["coefs"], extension["intercept"], extension["means"], x_row
    )
    r2, acc = extension["r2"], extension["acc"]
    label_fn = dummy_label
else:
    primary = models["primary"]
    x_row = pd.Series(raw_inputs)[PRIMARY_FEATURES]
    pred_score, contributions = predict_with_breakdown(
        primary["coefs"], primary["intercept"], primary["means"], x_row
    )
    r2, acc = primary["r2"], primary["acc"]
    label_fn = friendly_primary_label

bucket = to_bucket(pd.Series([pred_score])).iloc[0]
pred_display = float(np.clip(pred_score, 1, 10))

col1, col2 = st.columns(2)
col1.metric("Predicted sleep quality", f"{pred_display:.1f} / 10")
col2.markdown(
    f"**Predicted class**<br>"
    f"<span style='font-size:1.6rem;color:{STATUS_COLORS[bucket]}'>{STATUS_ICONS[bucket]} {bucket}</span>",
    unsafe_allow_html=True,
)

st.subheader("What drove this prediction")
st.caption(
    "Each bar is that input's pull on the score, versus an average person in the dataset "
    "(Linear Regression coefficient x how far your input sits from the dataset mean)."
)
st.markdown(
    f"<span style='color:{POSITIVE_COLOR}'>⬤</span> Raises predicted score"
    f"&nbsp;&nbsp;&nbsp;"
    f"<span style='color:{NEGATIVE_COLOR}'>⬤</span> Lowers predicted score",
    unsafe_allow_html=True,
)
st.altair_chart(contribution_chart(contributions, label_fn), width="stretch")

st.divider()
st.caption(
    f"**Honesty check:** this {'extension' if is_extension else 'research-question'} model "
    f"explains about **{r2:.0%}** of the variation in sleep quality (R²) and sorts people into "
    f"Low/Medium/High correctly about **{acc:.0%}** of the time on held-out data. "
    "Trained on a synthetic Kaggle dataset (100,000 rows) — these are patterns built into the "
    "data generator, not verified facts about human sleep."
)
