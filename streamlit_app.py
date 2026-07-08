import os
import json
import pandas as pd
import streamlit as st

APP_TITLE = "ChronoTrack Benchmark UI"

st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.caption("Visualize ChronoTrack dataset + evaluation results produced by eval_harness.py")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "chronotrack_dataset.jsonl")
RAW_CSV = os.path.join(BASE_DIR, "results_raw.csv")
SUMMARY_CSV = os.path.join(BASE_DIR, "results_summary.csv")


@st.cache_data(show_spinner=False)
def load_dataset(path: str, limit: int | None = None):
    if not os.path.exists(path):
        return []
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            items.append(json.loads(line))
            if limit is not None and i + 1 >= limit:
                break
    return items


@st.cache_data(show_spinner=False)
def load_csv(path: str):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


dataset = load_dataset(DATASET_PATH)
raw_df = load_csv(RAW_CSV)
summary_df = load_csv(SUMMARY_CSV)

col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("Dataset")
    if not dataset:
        st.warning(f"Missing dataset file: {DATASET_PATH}")
    else:
        ids = [it.get("id") for it in dataset]
        selected_id = st.selectbox(
            "Select item id",
            options=sorted(ids),
            index=0 if ids else None,
        )

        item = next((x for x in dataset if x.get("id") == selected_id), None)

        if item is not None:
            meta = item.get("meta", {})
            st.markdown("**Meta**")
            st.json({
                "order_mode": meta.get("order_mode"),
                "n_distractors": meta.get("n_distractors"),
                "query_type": meta.get("query_type"),
                "kind": meta.get("kind"),
                "target_entity": meta.get("target_entity"),
                "n_target_events": meta.get("n_target_events"),
            })

            st.markdown("**Question**")
            st.write(item.get("question"))

with col_right:
    st.subheader("Story + Evaluation")

    if not dataset or item is None:
        st.info("Select a dataset item to view story.")
    else:
        st.markdown("**Story**")
        st.write(item.get("story"))

        st.markdown("---")
        st.markdown("**Ground Truth**")
        st.write(f"gold_answer: **{item.get('gold_answer')}**")

        if raw_df.empty:
            st.warning(f"Missing results file(s). Expected: {RAW_CSV}")
        else:
            st.markdown("**Model predictions for this item**")
            pred_rows = raw_df[raw_df["item_id"] == item["id"]].copy() if "item_id" in raw_df.columns else pd.DataFrame()
            if pred_rows.empty:
                st.info("No prediction rows found for this item in results_raw.csv")
            else:
                # One row per model; this lets you compare accuracy per selected sentence/item.
                show_cols = [c for c in ["model", "order_mode", "n_distractors", "query_type", "prediction", "gold", "correct", "self_consistent", "latency_s"] if c in pred_rows.columns]
                pred_rows = pred_rows[show_cols].sort_values(["model", "query_type", "n_distractors", "order_mode"])
                st.dataframe(pred_rows, width='stretch', hide_index=True)

                # Aggregate accuracy for this single sentence/item.
                acc_by_model = (
                    pred_rows.groupby("model", as_index=False)["correct"].mean()
                    .rename(columns={"correct": "accuracy_for_this_item"})
                )
                st.markdown("### Accuracy per model (for selected item)")
                st.dataframe(acc_by_model, width='stretch', hide_index=True)



st.divider()
st.subheader("Accuracy summary (results_summary.csv)")

if summary_df.empty:
    st.warning(f"Missing results summary file: {SUMMARY_CSV}")
else:
    # Basic table
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Pivot view
    # We expect columns: model, order_mode, n_distractors, query_type, accuracy, n
    required = {"model", "order_mode", "n_distractors", "query_type", "accuracy"}
    if required.issubset(set(summary_df.columns)):
        st.markdown("### Pivot: Accuracy by condition")
        pivot = summary_df.pivot_table(
            index=["model", "query_type"],
            columns=["order_mode", "n_distractors"],
            values="accuracy",
            aggfunc="mean",
        )
        st.dataframe(pivot, use_container_width=True)

        # Order gap
        st.markdown("### Chronological vs Scrambled gap")
        gap_rows = []
        for model in summary_df["model"].unique():
            sub = summary_df[summary_df["model"] == model]
            # aggregate both query types and distractor levels
            chrono = sub[sub["order_mode"] == "chronological"]["accuracy"].mean() if not sub[sub["order_mode"] == "chronological"].empty else 0
            scram = sub[sub["order_mode"] == "scrambled"]["accuracy"].mean() if not sub[sub["order_mode"] == "scrambled"].empty else 0
            gap_rows.append({"model": model, "chronological_mean_acc": chrono, "scrambled_mean_acc": scram, "gap": chrono - scram})
        gap_df = pd.DataFrame(gap_rows).sort_values("gap", ascending=False)
        st.dataframe(gap_df, use_container_width=True, hide_index=True)

