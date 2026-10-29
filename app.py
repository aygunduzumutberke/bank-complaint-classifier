"""Streamlit app: classify complaints, run a batch, and compare the LLM with the baseline.

Run:
    streamlit run app.py
"""

from __future__ import annotations

import io
import json
import os

import pandas as pd
import streamlit as st

from src.config import CATEGORIES, GEMINI_MODEL, RESULTS_DIR
from src.llm_classifier import LLMClassifier
from src.plots import confusion_heatmap, f1_by_category

st.set_page_config(page_title="Bank Complaint Classifier", layout="wide")

EXAMPLES = {
    "EN – unauthorized card charges": (
        "I noticed three charges on my credit card statement from an online store I have never "
        "used. I called the bank twice and they told me to wait, but the money is still gone and "
        "now I am being charged interest on it."
    ),
    "EN – debt I don't recognise": (
        "A collection agency keeps calling me at work about a $1,200 debt that is not mine. I asked "
        "them to send proof of the debt and they refused and said they will sue me."
    ),
    "TR – EFT ulaşmadı": (
        "Üç gün önce başka bir bankadaki hesabıma 15.000 TL EFT gönderdim. Para hesabımdan çıktı ama "
        "karşı hesaba geçmedi. Müşteri hizmetleri sadece bekleyin diyor, kiramı ödeyemiyorum."
    ),
    "TR – kredi notu hatası": (
        "Kapattığım bir kredi kartı borcu Findeks raporumda hâlâ gecikmede görünüyor. Bu yüzden konut "
        "kredisi başvurum reddedildi. Bankaya iki kez yazdım, düzeltilmedi."
    ),
    "Write your own": "",
}
URGENCY_LABEL = {"high": "High", "medium": "Medium", "low": "Low"}


@st.cache_resource
def get_classifier(provider: str) -> LLMClassifier:
    return LLMClassifier(provider=provider)


@st.cache_resource
def get_baseline():
    try:
        from src.baseline import load_model

        return load_model()
    except FileNotFoundError:
        return None


def to_excel(df: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    df.to_excel(buffer, index=False, engine="openpyxl")
    return buffer.getvalue()


def label_columns(results: list[dict]) -> pd.DataFrame:
    """Prediction columns, prefixed so they never clash with the uploaded file's own columns."""
    df = pd.DataFrame(results)[["category", "urgency", "sentiment", "summary"]]
    df["category_tr"] = df["category"].map(lambda c: CATEGORIES.get(c, {}).get("label_tr", c))
    df["route_to"] = df["category"].map(lambda c: CATEGORIES.get(c, {}).get("team", "Manual review"))
    return df.add_prefix("pred_")


# ---- sidebar ------------------------------------------------------------------
has_key = bool(os.getenv("GEMINI_API_KEY"))
with st.sidebar:
    st.header("Settings")
    use_mock = st.toggle("Offline mock mode", value=not has_key, disabled=not has_key,
                         help="Keyword rules instead of the LLM. On automatically when no API key is set.")
    provider = "mock" if use_mock else "gemini"
    st.caption(f"Model: `{GEMINI_MODEL if provider == 'gemini' else 'keyword rules (mock)'}`")
    if not has_key:
        st.info("No GEMINI_API_KEY found. Add it to `.env` to use the real model.")
    st.divider()
    st.caption("Data: CFPB Consumer Complaint Database (public). Personal data in narratives is "
               "masked by CFPB as XXXX.")

st.title("Bank Complaint Classifier")
st.write("Routes customer complaints to the right team with an LLM, and measures it against a "
         "classical TF-IDF + Logistic Regression model.")
if provider == "mock":
    st.warning("Mock mode: answers come from simple keyword rules, not the LLM.")

classifier = get_classifier(provider)
tab_single, tab_batch, tab_compare, tab_explore = st.tabs(
    ["Classify a complaint", "Batch (CSV / Excel)", "Model comparison", "Explore predictions"]
)

# ---- single complaint ---------------------------------------------------------
with tab_single:
    choice = st.selectbox("Start from an example", list(EXAMPLES))
    text = st.text_area("Complaint text (English or Turkish)", value=EXAMPLES[choice], height=160,
                        key=f"text_{choice}")
    if st.button("Classify", type="primary", disabled=not text.strip()):
        with st.spinner("Classifying..."):
            try:
                result = classifier.classify(text)
            except Exception as exc:
                st.error(f"The model call failed: {exc}")
                st.stop()
        info = CATEGORIES[result["category"]]
        # Two columns per row so long labels are not cut off on narrow screens.
        c1, c2 = st.columns(2)
        c1.markdown(f"**Category**  \n{info['label_en']}")
        c2.markdown(f"**Route to**  \n{info['team']}")
        c3, c4 = st.columns(2)
        c3.markdown(f"**Urgency**  \n{URGENCY_LABEL[result['urgency']]}")
        c4.markdown(f"**Sentiment**  \n{result['sentiment'].replace('_', ' ')}")
        st.info(result["summary"])
        st.caption(f"{info['label_tr']} · {result['model']} · "
                   f"{'from cache' if result['cached'] else str(result['latency_ms']) + ' ms'}")

        baseline = get_baseline()
        if baseline is not None:
            guess = baseline.predict([text])[0]
            agree = "agrees" if guess == result["category"] else "disagrees"
            st.caption(f"Baseline model says **{CATEGORIES[guess]['label_en']}** ({agree}). It was trained "
                       "on English text only, so ignore it for Turkish complaints.")

# ---- batch ---------------------------------------------------------------------
with tab_batch:
    st.write("Upload a file with one complaint per row. Results download as Excel.")
    upload = st.file_uploader("CSV or Excel file", type=["csv", "xlsx"])
    if upload is not None:
        data = pd.read_csv(upload) if upload.name.endswith(".csv") else pd.read_excel(upload)
        text_col = st.selectbox("Which column holds the complaint text?", list(data.columns))
        limit = st.slider("Rows to classify", 1, min(300, len(data)), min(50, len(data)),
                          help="The free Gemini tier allows only a few requests per minute.")
        if st.button("Classify batch", type="primary"):
            rows = data.head(limit)
            progress = st.progress(0.0, text="Starting...")
            results = []
            for i, value in enumerate(rows[text_col].astype(str), start=1):
                try:
                    results.append(classifier.classify(value))
                except Exception as exc:
                    results.append({"category": "error", "urgency": None, "sentiment": None,
                                    "summary": f"ERROR: {exc}"})
                progress.progress(i / len(rows), text=f"{i}/{len(rows)} classified")
            output = pd.concat([rows.reset_index(drop=True), label_columns(results)], axis=1)
            st.session_state["batch_output"] = output

    if "batch_output" in st.session_state:
        output = st.session_state["batch_output"]
        c1, c2 = st.columns(2)
        c1.write("**By category**")
        c1.bar_chart(output["pred_category"].value_counts(), horizontal=True)
        c2.write("**By urgency**")
        c2.bar_chart(output["pred_urgency"].value_counts(), horizontal=True)
        st.dataframe(output, width="stretch", hide_index=True)
        st.download_button("Download results (.xlsx)", to_excel(output), "classified_complaints.xlsx")

# ---- comparison ----------------------------------------------------------------
with tab_compare:
    path = RESULTS_DIR / "comparison.json"
    if not path.exists():
        st.info("No evaluation yet. Run `python -m src.baseline` and then `python -m src.evaluate_llm`.")
    else:
        comparison = json.loads(path.read_text())
        llm, base = comparison["llm"], comparison["baseline"]
        st.write(f"Both models scored on the same **{comparison['n']}** test complaints. "
                 f"LLM: `{llm['model']}` (zero-shot).")
        c1, c2 = st.columns(2)
        c1.metric("LLM accuracy", f"{llm['accuracy']:.1%}",
                  delta=f"{100 * (llm['accuracy'] - base['accuracy']):+.1f} pts vs baseline")
        c2.metric("Baseline accuracy", f"{base['accuracy']:.1%}")
        c3, c4 = st.columns(2)
        c3.metric("LLM macro F1", f"{llm['macro_f1']:.3f}")
        c4.metric("Baseline macro F1", f"{base['macro_f1']:.3f}")
        st.pyplot(f1_by_category(comparison))
        left, right = st.columns(2)
        left.pyplot(confusion_heatmap(llm, "LLM: actual vs predicted"))
        right.pyplot(confusion_heatmap(base, "Baseline: actual vs predicted"))

# ---- explore -------------------------------------------------------------------
with tab_explore:
    path = RESULTS_DIR / "llm_predictions.csv"
    if not path.exists():
        st.info("Run `python -m src.evaluate_llm` to create predictions to explore.")
    else:
        preds = pd.read_csv(path)
        c1, c2, c3 = st.columns(3)
        cats = c1.multiselect("Actual category", sorted(preds["category"].unique()))
        urg = c2.multiselect("Urgency", ["high", "medium", "low"])
        only_wrong = c3.checkbox("Only LLM mistakes")
        view = preds
        if cats:
            view = view[view["category"].isin(cats)]
        if urg:
            view = view[view["urgency"].isin(urg)]
        if only_wrong:
            view = view[~view["llm_correct"]]
        st.caption(f"{len(view)} of {len(preds)} complaints")
        columns = ["category", "llm_category", "baseline_category", "urgency", "sentiment", "summary", "text"]
        st.dataframe(view[columns], width="stretch", hide_index=True)
        st.download_button("Download view (.xlsx)", to_excel(view[columns]), "predictions.xlsx")
