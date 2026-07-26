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

# Diverging pair stepped for the dark night-sky surface: blue = raises the score, red = lowers it.
POSITIVE_COLOR = "#3987e5"
NEGATIVE_COLOR = "#e66767"
STATUS_COLORS = {"Low": "#d03b3b", "Medium": "#fab219", "High": "#0ca30c"}
STATUS_ICONS = {"Low": "⚠️", "Medium": "●", "High": "✓"}


# Each bar shows how a person differs from the dataset average, so a switch that
# is OFF produces a bar with the opposite sign to its coefficient — being a
# non-shift-worker is a small plus. Naming the state rather than the column keeps
# the sign readable: "No shift work +0.08" instead of "shift_work +0.08".
BINARY_STATE_LABELS = {
    "shift_work": ("Works shifts", "No shift work"),
    "exercise_day": ("Exercised today", "No exercise today"),
}


def _is_on(value) -> bool:
    try:
        return float(value) >= 0.5
    except (TypeError, ValueError):
        return bool(value)


def dummy_label(dummy_col: str, value=None) -> str:
    if dummy_col in BINARY_STATE_LABELS:
        on, off = BINARY_STATE_LABELS[dummy_col]
        return on if _is_on(value) else off
    for base in EXTENSION_CATEGORICAL:
        prefix = base + "_"
        if dummy_col.startswith(prefix):
            category = dummy_col[len(prefix):]
            state = category if _is_on(value) else f"not {category}"
            return f"{FRIENDLY_BASE[base]}: {state}"
    return FRIENDLY_BASE.get(dummy_col, dummy_col)


def friendly_primary_label(feature: str, value=None) -> str:
    if feature in BINARY_STATE_LABELS:
        on, off = BINARY_STATE_LABELS[feature]
        return on if _is_on(value) else off
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


def biggest_lever(coefs: pd.Series, raw: dict, is_extension: bool):
    """The single actionable change that would raise the predicted score the most.

    Only considers levers a person can actually pull tonight (stress management,
    exercise, late caffeine/alcohol/screens) — not traits like age or occupation.
    Returns (phrase, gain_in_points) or None if nothing moves the needle.
    """
    candidates = []
    stress = raw["stress_score"]
    if stress > 1:
        drop = min(2, stress - 1)
        candidates.append(("stress_score", -drop, f"lowering stress from {stress} to {stress - drop}"))
    if raw["exercise_day"] == 0:
        candidates.append(("exercise_day", 1, "getting some exercise today"))
    if is_extension:
        if raw["screen_time_before_bed_mins"] >= 30:
            candidates.append(("screen_time_before_bed_mins", -30, "cutting screen time before bed by 30 minutes"))
        if raw["caffeine_mg_before_bed"] >= 50:
            candidates.append(("caffeine_mg_before_bed", -50, "skipping ~50 mg of late caffeine"))
        if raw["alcohol_units_before_bed"] > 0:
            candidates.append(("alcohol_units_before_bed", -raw["alcohol_units_before_bed"], "skipping alcohol before bed"))
    scored = [(phrase, float(coefs.get(col, 0.0)) * delta) for col, delta, phrase in candidates]
    scored = [(phrase, gain) for phrase, gain in scored if gain >= 0.05]
    if not scored:
        return None
    return max(scored, key=lambda t: t[1])


ROW_HEIGHT = 34
BAR_THICKNESS = 20
LABEL_INK = "#a7abc4"  # moonlit secondary ink — readable on the night surface


def contribution_chart(contributions: pd.Series, label_fn, values, top_n: int = 8) -> alt.Chart:
    ranked = contributions.reindex(contributions.abs().sort_values(ascending=False).index)
    ranked = ranked[ranked.abs() > 1e-6].head(top_n)
    plot_df = pd.DataFrame({
        "feature": [label_fn(f, values.get(f)) for f in ranked.index],
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

    zero_rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#3a4159").encode(x="x:Q")
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
    "Can a few facts about your day predict how well you'll sleep? "
    "Answer the questions in the sidebar and our model will guess your sleep quality — "
    "and show you exactly why."
)
st.caption("⬅️ Start in the **sidebar** — on a phone, tap the arrow in the top-left corner.")

st.sidebar.header("Tell us about your day")
mode = st.sidebar.radio(
    "How much detail?",
    ["Just the basics (age, stress, activity)", "Full picture (all lifestyle factors)"],
    help="'Just the basics' uses only the three factors from our research question. "
         "'Full picture' adds the rest of your day — work hours, caffeine, screen time, and more — "
         "which makes the prediction more accurate (68% vs 61%).",
)
is_extension = mode.startswith("Full picture")

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
        extension_inputs["chronotype"] = st.selectbox(
            "Morning or night person? (chronotype)", CATEGORY_OPTIONS["chronotype"]
        )
        extension_inputs["mental_health_condition"] = st.selectbox(
            "Mental health", CATEGORY_OPTIONS["mental_health_condition"]
        )
        extension_inputs["day_type"] = st.selectbox("Weekday or weekend?", CATEGORY_OPTIONS["day_type"])
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

# Final held-out test results from notebooks/05_model_refinements.ipynb
# (70/15/15 split, seed 117). Paths A and B are the two models the proposal
# promised; Gradient Boosting was added afterwards only as a sanity check. The
# app serves Linear Regression because it ties the rest while staying explainable.
ALGORITHM_COMPARISON = pd.DataFrame(
    {
        "Algorithm": ["Baseline (always guess Medium)",
                      "Path A — Linear Regression",
                      "Path B — Random Forest (max_depth=8)",
                      "Challenger — Gradient Boosting"],
        "Research-question features": ["44.5%", "61.2%", "61.3%", "61.8%*"],
        "Extension features": ["44.5%", "68.3%", "67.9%*", "68.8%"],
    }
).set_index("Algorithm")

bucket = to_bucket(pd.Series([pred_score])).iloc[0]
pred_display = float(np.clip(pred_score, 1, 10))

col1, col2 = st.columns(2)
col1.metric("Predicted sleep quality", f"{pred_display:.1f} / 10")
col2.markdown(
    f"**Sleep category**<br>"
    f"<span style='font-size:1.6rem;color:{STATUS_COLORS[bucket]}'>{STATUS_ICONS[bucket]} {bucket}</span>",
    unsafe_allow_html=True,
)

# One plain-language sentence: the single biggest reason behind this prediction.
top_feature = contributions.abs().idxmax()
top_effect = float(contributions[top_feature])
if abs(top_effect) >= 0.15:
    top_name = label_fn(top_feature, x_row.get(top_feature))
    verb = "pulling your score up" if top_effect > 0 else "pulling your score down"
    st.markdown(
        f"The biggest factor here: **{top_name}**, {verb} by about **{abs(top_effect):.1f} points**."
    )

st.caption(
    f"Model: **Linear Regression** · "
    f"{'full-picture' if is_extension else 'basic (3-factor)'} version · "
    "trained on 70,000 people, tested on 15,000 people it had never seen"
)

lever = biggest_lever(
    models["extension"]["coefs"] if is_extension else models["primary"]["coefs"],
    raw_inputs, is_extension,
)
if lever:
    phrase, gain = lever
    st.info(f"💡 **What would help most:** {phrase} would raise your predicted score by ~{gain:.1f} points.")

st.subheader("What drove this prediction")
st.caption(
    "Blue bars pushed your score up; red bars pulled it down. "
    "Each bar compares you with a typical person in our data."
)
st.markdown(
    f"<span style='color:{POSITIVE_COLOR}'>⬤</span> Raises predicted score"
    f"&nbsp;&nbsp;&nbsp;"
    f"<span style='color:{NEGATIVE_COLOR}'>⬤</span> Lowers predicted score",
    unsafe_allow_html=True,
)
st.altair_chart(contribution_chart(contributions, label_fn, x_row), width="stretch")

with st.expander("How are the bars calculated?"):
    st.markdown(
        "The model learned how much each factor matters (its *coefficient*). Each bar is that "
        "importance multiplied by how far **your** answer sits from the dataset average. "
        "That's why a factor can show a small bar even when it matters a lot — if your answer "
        "is close to typical, it isn't moving *your* prediction much. It also means labels "
        "describe your state: \"No shift work\" shows a small plus because *not* working shifts "
        "is slightly better than average, even though shift work itself costs about a full point."
    )

with st.expander("How our two models compared — and the challenger we tested"):
    st.table(ALGORITHM_COMPARISON)
    st.caption(
        "Held-out accuracy per model and feature set "
        "(*validation-set figure — each final model visits the test set only once). "
        "**Paths A and B are the project:** they tie, which is itself a finding — the signal is "
        "one clean linear stress effect. Gradient Boosting was added afterwards purely as a "
        "sanity check, and the extension feature set is a follow-up question, not the original "
        "research question. We deploy Linear Regression because it ties the rest **and** can "
        "explain every prediction — the chart above and the 💡 tip come straight from its "
        "coefficients, which tree-based models can't provide as directly."
    )

st.divider()
st.caption(
    f"**Honesty check:** this {'full-picture' if is_extension else 'basic'} model explains about "
    f"**{r2:.0%}** of what makes sleep quality differ between people (R²), and sorts people into "
    f"Low/Medium/High correctly about **{acc:.0%}** of the time on people it never saw during "
    "training. It was trained on a synthetic (computer-generated) Kaggle dataset of 100,000 "
    "records — so these are patterns built into that data, not verified facts about human sleep. "
    "This is a class project, not medical advice."
)
st.caption(
    "Full analysis & code: [github.com/Poudel-Sanskriti/Sleep_Doctor]"
    "(https://github.com/Poudel-Sanskriti/Sleep_Doctor) · AI4ALL Ignite portfolio project"
)
