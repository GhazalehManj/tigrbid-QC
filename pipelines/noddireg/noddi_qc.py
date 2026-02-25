#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image
import re


# -----------------------------
# Argument parsing
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="AMICO NODDI QC")
    parser.add_argument("--noddireg_dir", required=True)
    parser.add_argument("--participant_labels", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


# -----------------------------
# Participants
# -----------------------------
def load_participants(tsv):
    df = pd.read_csv(tsv, sep="\t")
    if "participant_id" not in df.columns:
        raise ValueError("participants.tsv must contain participant_id")
    return df["participant_id"].tolist()


# -----------------------------
# Detect sessions from filenames
# -----------------------------
def detect_sessions(subj_dir, subject):
    sessions = set()
    for f in subj_dir.glob(f"{subject}_*noddi*.png"):
        m = re.search(r"(ses-[a-zA-Z0-9]+)", f.name)
        sessions.add(m.group(1) if m else "no-session")
    return sorted(sessions)


def do_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    elif hasattr(st, "experimental_rerun"):
        st.experimental_rerun()
    else:
        st.warning("No rerun function available in this Streamlit version.")


# -----------------------------
# QC radio
# -----------------------------
QC_OPTIONS = ["—", "PASS", "FAIL", "UNCERTAIN"]  # "—" means not selected


def qc_radio(label: str, key: str):
    val = st.radio(
        label,
        QC_OPTIONS,
        horizontal=True,
        key=key,
        index=QC_OPTIONS.index(st.session_state.get(key, "—")) if st.session_state.get(key, "—") in QC_OPTIONS else 0,
    )
    return None if val == "—" else val


# -----------------------------
# Load existing saved QC for subject
# -----------------------------
def load_saved_for_subject(out_tsv: Path, subject: str) -> dict:
    """
    Returns:
      saved[session] = {
        "density_qc": "PASS"/"FAIL"/"UNCERTAIN"/None,
        "density_comment": "...",
        "od_qc": ...,
        "od_comment": ...,
        "icvf_qc": ...,
        "icvf_comment": ...,
        "isovf_qc": ...,
        "isovf_comment": ...
      }
    """
    if not out_tsv.exists():
        return {}

    df = pd.read_csv(out_tsv, sep="\t")

    if "participant_id" not in df.columns:
        return {}

    df = df[df["participant_id"] == subject]
    if df.empty:
        return {}

    saved = {}
    for _, r in df.iterrows():
        ses = r.get("session")
        if pd.isna(ses) or ses == "":
            ses = "no-session"

        def clean(v):
            if pd.isna(v) or v == "":
                return None
            return str(v)

        saved[ses] = {
            "density_qc": clean(r.get("density_qc")),
            "density_comment": "" if pd.isna(r.get("density_comment")) else str(r.get("density_comment")),
            "od_qc": clean(r.get("od_qc")),
            "od_comment": "" if pd.isna(r.get("od_comment")) else str(r.get("od_comment")),
            "icvf_qc": clean(r.get("icvf_qc")),
            "icvf_comment": "" if pd.isna(r.get("icvf_comment")) else str(r.get("icvf_comment")),
            "isovf_qc": clean(r.get("isovf_qc")),
            "isovf_comment": "" if pd.isna(r.get("isovf_comment")) else str(r.get("isovf_comment")),
        }

    return saved


# -----------------------------
# Save helper
# -----------------------------
def save_subject_rows(output_dir: Path, subject: str, out_rows: list[dict]):
    """
    Save current subject's QC rows to noddi_qc.tsv
    - overwrites any existing rows for this subject (all sessions)
    """
    out_tsv = output_dir / "noddi_qc.tsv"
    new_df = pd.DataFrame(out_rows)

    if out_tsv.exists():
        old = pd.read_csv(out_tsv, sep="\t")
        if "participant_id" in old.columns:
            old = old[old["participant_id"] != subject]
        out = pd.concat([old, new_df], ignore_index=True)
    else:
        out = new_df

    out.to_csv(out_tsv, sep="\t", index=False, na_rep="")
    return out_tsv


# -----------------------------
# Streamlit app
# -----------------------------
def main():
    args = parse_args()

    noddireg_dir = Path(args.noddireg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants = load_participants(args.participant_labels)

    st.set_page_config(layout="centered")
    st.markdown('<div id="top_of_page"></div>', unsafe_allow_html=True)
    st.title("AMICO NODDI QC")

    # Session state
    if "subj_idx" not in st.session_state:
        st.session_state.subj_idx = 0

    # Page jump widget key
    if "page_jump_widget" not in st.session_state:
        st.session_state.page_jump_widget = 1

    # Current subject
    subject = participants[st.session_state.subj_idx]

    st.session_state.page_jump_widget = st.session_state.subj_idx + 1

    st.sidebar.markdown(f"**Subject {st.session_state.subj_idx + 1} / {len(participants)}**")
    st.sidebar.markdown(f"### {subject}")

    subj_dir = noddireg_dir / subject
    if not subj_dir.exists():
        st.error(f"Missing directory: {subj_dir}")
        return

    sessions = detect_sessions(subj_dir, subject)
    if not sessions:
        st.warning("No NODDI images found")
        return

    # Load saved values for this subject
    out_tsv = output_dir / "noddi_qc.tsv"
    saved = load_saved_for_subject(out_tsv, subject)

    out_rows = []

    for ses in sessions:
        st.divider()
        st.header(f"{subject} — {ses}")

        ses_prefix = f"{subject}_{ses}" if ses != "no-session" else subject

        # -----------------------------
        # Density plot
        # -----------------------------
        st.subheader("Tissue Density (CSF / GM / WM)")

        density_pngs = list(subj_dir.glob(f"{ses_prefix}*_desc-dsegtissue_model-noddi_density.png"))

        if density_pngs:
            for png in density_pngs:
                st.image(Image.open(png), use_container_width=True)
                st.caption(str(png))
        else:
            st.warning("Density plot not found")

        density_key = f"{subject}_{ses}_density_qc"
        density_comment_key = f"{subject}_{ses}_density_comment"

        if density_key not in st.session_state:
            st.session_state[density_key] = saved.get(ses, {}).get("density_qc") or "—"
        if density_comment_key not in st.session_state:
            st.session_state[density_comment_key] = saved.get(ses, {}).get("density_comment", "")

        density_qc = qc_radio("Density QC", key=density_key)

        density_comment = st.text_area(
            "Density comments",
            key=density_comment_key,
        )

        st.divider()

        # -----------------------------
        # Parcel-wise QA
        # -----------------------------
        st.subheader("Parcel-wise NODDI Metrics (Schaefer 4S1056)")

        metrics = ["od_mean", "icvf_mean", "isovf_mean"]
        metric_qc = {}
        metric_comment = {}

        for metric in metrics:
            st.markdown(f"**{metric.upper()}**")

            pngs = sorted(subj_dir.glob(f"{ses_prefix}*_{metric}_qc.png"))

            if pngs:
                for png in pngs:
                    st.image(Image.open(png), use_container_width=True)
                    st.caption(str(png))
            else:
                st.warning(f"{metric.upper()} QA plot not found")

            # map metric -> saved columns
            col_qc = {"od_mean": "od_qc", "icvf_mean": "icvf_qc", "isovf_mean": "isovf_qc"}[metric]
            col_cmt = {"od_mean": "od_comment", "icvf_mean": "icvf_comment", "isovf_mean": "isovf_comment"}[metric]

            mkey = f"{subject}_{ses}_{metric}_qc"
            ckey = f"{subject}_{ses}_{metric}_comment"

            if mkey not in st.session_state:
                st.session_state[mkey] = saved.get(ses, {}).get(col_qc) or "—"
            if ckey not in st.session_state:
                st.session_state[ckey] = saved.get(ses, {}).get(col_cmt, "")

            metric_qc[metric] = qc_radio(f"{metric.upper()} QC", key=mkey)

            metric_comment[metric] = st.text_area(
                f"{metric.upper()} comments",
                key=ckey,
            )

            st.divider()

        out_rows.append(
            {
                "participant_id": subject,
                "session": ses,
                "density_qc": density_qc,
                "density_comment": density_comment,
                "od_qc": metric_qc["od_mean"],
                "od_comment": metric_comment["od_mean"],
                "icvf_qc": metric_qc["icvf_mean"],
                "icvf_comment": metric_comment["icvf_mean"],
                "isovf_qc": metric_qc["isovf_mean"],
                "isovf_comment": metric_comment["isovf_mean"],
            }
        )

    # -----------------------------
    # Jump callback (SAVE then jump)
    # -----------------------------
    def jump_to_page():
        save_subject_rows(output_dir, subject, out_rows)
        st.session_state.subj_idx = int(st.session_state.page_jump_widget) - 1
        do_rerun()

    # -----------------------------
    # Bottom navigation (Prev/Next SAVE + Page Jump + Top)
    # -----------------------------
    st.divider()
    total_pages = len(participants)

    bottom = st.columns((1.2, 2.4, 2.4, 1.0))

    with bottom[0]:
        st.markdown(f"Page **{st.session_state.subj_idx + 1}** of **{total_pages}**")

    with bottom[1]:
        c1, c2 = st.columns(2, gap="small")

        if c1.button("⬅️ Prev (save)", use_container_width=True):
            save_subject_rows(output_dir, subject, out_rows)
            if st.session_state.subj_idx > 0:
                st.session_state.subj_idx -= 1
            do_rerun()

        if c2.button("Next (save) ➡️", use_container_width=True):
            save_subject_rows(output_dir, subject, out_rows)
            if st.session_state.subj_idx < total_pages - 1:
                st.session_state.subj_idx += 1
            do_rerun()

    with bottom[2]:
        st.number_input(
            "Go to page",
            min_value=1,
            max_value=total_pages,
            step=1,
            key="page_jump_widget",
            on_change=jump_to_page,
        )

    with bottom[3]:
        st.markdown(
            """
            <a href="#top_of_page">
                <button style="
                    background-color: white;
                    border: 1px solid #d1d5db;
                    padding: 0.5rem 0.8rem;
                    border-radius: 0.5rem;
                    cursor: pointer;
                    width: 100%;">
                    ⬆️ Top
                </button>
            </a>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
