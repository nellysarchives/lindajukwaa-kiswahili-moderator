import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(
    page_title="LindaJukwaa: Kiswahili Content Moderator",
    layout="centered"
)

st.markdown(
    """
    <style>
    .stApp, h1, h2, h3, h4, p, label, input, textarea, button,
    [data-testid="stMarkdownContainer"], [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        font-family: "Times New Roman", Times, serif !important;
    }
    [data-testid="stIconMaterial"], .material-symbols-rounded, .material-icons {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
    }
    code, pre {
        font-family: monospace !important;
    }
    .hero-banner {
        background-color: #FFFFFF;
        border-left: 5px solid #0F766E;
        padding: 1.1rem 1.4rem;
        border-radius: 0.5rem;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08);
        margin-bottom: 1.2rem;
    }
    .hero-banner h1 {
        margin: 0 0 0.35rem 0;
        font-size: 2rem;
        color: #0F172A;
    }
    .hero-banner p {
        margin: 0;
        font-size: 1.05rem;
        color: #334155;
    }
    </style>
    """,
    unsafe_allow_html=True
)

MODEL_PATH = "nellymbuvi/afro-xlmr-kiswahili-moderation"
LABELS = ["Neutral", "Offensive", "Hate Speech"]

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

def predict_batch(texts: list[str]):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).tolist()
    pred_indices = torch.argmax(outputs.logits, dim=-1).tolist()
    return [
        (LABELS[idx], dict(zip(LABELS, p)))
        for idx, p in zip(pred_indices, probs)
    ]

st.markdown(
    """
    <div class="hero-banner">
        <h1>LindaJukwaa: Kiswahili Content Moderator</h1>
        <p>Automated detection of <strong>Neutral</strong>, <strong>Offensive</strong>, and <strong>Hate Speech</strong> comments in Kiswahili and English–Swahili text.</p>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2 = st.tabs(["💬 Single Comment Analysis", "📂 Batch CSV Moderation"])

with tab1:
    with st.container(border=True):
        st.subheader("Test a Social Media Comment")
        st.caption("Click a preset sample below or type your own Kiswahili / Sheng comment:")

        col1, col2, col3 = st.columns(3)
        if col1.button("🟢 Sample: Neutral", use_container_width=True):
            st.session_state["sample_text"] = "Habari za asubuhi ndugu zangu, tuendelee kujenga nchi yetu kwa amani."
            st.rerun()
        if col2.button("🟡 Sample: Offensive", use_container_width=True):
            st.session_state["sample_text"] = "Wewe ni mjinga sana na hujui unachokiongea kabisa."
            st.rerun()
        if col3.button("🔴 Sample: Hate Speech", use_container_width=True):
            st.session_state["sample_text"] = "Watu wa kabila hilo hawafai kuishi hapa, wafukuzwe wote."
            st.rerun()

        user_input = st.text_area(
            "Enter Kiswahili or Sheng/English–Swahili text:",
            key="sample_text",
            height=110,
            placeholder="Andika maoni hapa..."
        )

        analyze_clicked = st.button("🔍 Analyze Comment", type="primary")

    if analyze_clicked:
        if not user_input.strip():
            st.warning("Please enter a comment to classify.")
        else:
            label, probs = predict_batch([user_input])[0]

            with st.container(border=True):
                if label == "Neutral":
                    st.success(f"**Classification: {label}** ({probs[label]*100:.1f}% confidence)")
                    st.info("**Recommended Moderation Action:** Approve — Safe for publication.")
                elif label == "Offensive":
                    st.warning(f"**Classification: {label}** ({probs[label]*100:.1f}% confidence)")
                    st.info("**Recommended Moderation Action:** Flag for Warning / Reduce Visibility.")
                else:
                    st.error(f"**Classification: {label}** ({probs[label]*100:.1f}% confidence)")
                    st.info("**Recommended Moderation Action:** Block Immediately & Escalate to Human Moderator.")

                if probs[label] < 0.60:
                    st.caption("⚠️ Low model confidence (<60%) — recommended for secondary human review.")

                st.markdown("#### Class Confidence Breakdown")
                for cls_name in LABELS:
                    st.write(f"**{cls_name}:** {probs[cls_name]*100:.2f}%")
                    st.progress(probs[cls_name])

with tab2:
    with st.container(border=True):
        st.subheader("Moderate a Dataset (.csv)")
        st.write("Upload a CSV file containing a **`text`** or **`clean_text`** column to classify multiple comments at once.")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            text_col = "clean_text" if "clean_text" in df.columns else ("text" if "text" in df.columns else None)

            if text_col is None:
                st.error("CSV must contain a 'text' or 'clean_text' column.")
            else:
                if st.button("⚙️ Run Batch Classification", type="primary"):
                    texts = df[text_col].fillna("").astype(str).tolist()
                    results = []
                    progress = st.progress(0)
                    total = len(texts)
                    batch_size = 32

                    for start in range(0, total, batch_size):
                        batch = texts[start : start + batch_size]
                        batch_preds = predict_batch(batch)
                        for lbl, pr in batch_preds:
                            results.append({
                                "Prediction": lbl,
                                "Confidence (%)": round(pr[lbl] * 100, 2),
                                "Neutral_Prob": round(pr["Neutral"], 4),
                                "Offensive_Prob": round(pr["Offensive"], 4),
                                "Hate_Speech_Prob": round(pr["Hate Speech"], 4)
                            })
                        progress.progress(min((start + batch_size) / total, 1.0))

                    st.session_state["batch_out_df"] = pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    if "batch_out_df" in st.session_state:
        out_df = st.session_state["batch_out_df"]
        total = len(out_df)
        counts = out_df["Prediction"].value_counts()

        with st.container(border=True):
            st.success(f"Classified {total} comments.")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Scanned", total)
            m2.metric("Neutral", f"{counts.get('Neutral', 0)} ({counts.get('Neutral', 0)/total*100:.0f}%)")
            m3.metric("Offensive", f"{counts.get('Offensive', 0)} ({counts.get('Offensive', 0)/total*100:.0f}%)")
            m4.metric("Hate Speech", f"{counts.get('Hate Speech', 0)} ({counts.get('Hate Speech', 0)/total*100:.0f}%)")

            chart_data = pd.DataFrame({
                "Count": [counts.get(lbl, 0) for lbl in LABELS]
            }, index=LABELS)
            st.bar_chart(chart_data, color="#0F766E")

            filter_choice = st.selectbox("Filter preview by class:", ["All", "Hate Speech", "Offensive", "Neutral"])
            filtered_df = out_df if filter_choice == "All" else out_df[out_df["Prediction"] == filter_choice]
            st.dataframe(filtered_df.head(50), use_container_width=True)

            csv_bytes = out_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Moderated CSV",
                data=csv_bytes,
                file_name="moderated_comments.csv",
                mime="text/csv"
            )