#!/usr/bin/env python3
import sys
from pathlib import Path

UI_DIR = Path(__file__).resolve().parents[2] / "ui"
if str(UI_DIR) not in sys.path:
    sys.path.insert(0, str(UI_DIR))

import argparse
import math
import re
from datetime import datetime
from typing import Dict, Optional, Tuple, List

import pandas as pd
import streamlit as st
from PIL import Image

from models import MetricQC, QCRecord


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="AMICO NODDI QC")
    parser.add_argument("--noddireg_dir", required=True)
    parser.add_argument("--participant_labels", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(args)


def load_participants_df(tsv: str) -> pd.DataFrame:
    df = pd.read_csv(tsv, sep="\t")
    if "participant_id" not in df.columns:
        raise ValueError("participants.tsv must contain participant_id")
    return df


def detect_sessions(subj_dir: Path, subject: str) -> List[str]:
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


def init_session_state():
    if "current_page" not in st.session_state:
        st.session_state.current_page = 1


def get_current_batch(df: pd.DataFrame, current_page: int):
    batch_size = 1
    total_rows = len(df)
    start_idx = (current_page - 1) * batch_size
    end_idx = min(start_idx + batch_size, total_rows)
    return total_rows, df.iloc[start_idx:end_idx]


QC_OPTIONS = ("pass", "fail", "uncertain")


def seed_from_saved(key: str, saved_val):
    if saved_val is None:
        return
    if key not in st.session_state:
        st.session_state[key] = saved_val
        return
    cur = st.session_state.get(key)
    if cur is None:
        st.session_state[key] = saved_val
        return
    if isinstance(cur, str) and cur.strip() == "":
        st.session_state[key] = saved_val
        return
    if key.endswith("_qc") and cur not in QC_OPTIONS:
        st.session_state[key] = saved_val
        return


def qc_radio(label: str, key: str, stored_val: Optional[str]):
    if stored_val not in QC_OPTIONS:
        stored_val = None
    seed_from_saved(key, stored_val)
    idx = QC_OPTIONS.index(st.session_state[key]) if st.session_state.get(key) in QC_OPTIONS else None
    return st.radio(label, QC_OPTIONS, key=key, horizontal=True, index=idx)


def notes_box(label: str, key: str, stored_val: str):
    stored_val = "" if stored_val is None else str(stored_val)
    seed_from_saved(key, stored_val)
    return st.text_area(label, key=key)


def compute_overall(metrics: List[MetricQC]) -> Optional[str]:
    qcs = [m.qc for m in metrics if m.qc in QC_OPTIONS]
    if not qcs:
        return None
    if "fail" in qcs:
        return "fail"
    if "uncertain" in qcs:
        return "uncertain"
    return "pass"


Key = Tuple[str, Optional[str], str]

OUTPUT_COLUMNS = [
    "participant_id",
    "qc_task",
    "session_id",
    "task_id",
    "run_id",
    "rater_id",
    "final_qc",
    "notes",
]


def read_existing_csv(out_file: Path) -> pd.DataFrame:
    if not out_file.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        df = pd.read_csv(out_file)
        for c in OUTPUT_COLUMNS:
            if c not in df.columns:
                df[c] = ""
        return df[OUTPUT_COLUMNS].copy()
    except Exception:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def build_lookup(out_file: Path) -> Dict[Key, Dict[str, str]]:
    df = read_existing_csv(out_file)
    if df.empty:
        return {}
    lookup: Dict[Key, Dict[str, str]] = {}
    for _, r in df.iterrows():
        pid = str(r.get("participant_id", ""))
        ses = r.get("session_id")
        ses = None if pd.isna(ses) or ses == "" else str(ses)
        qc_task = str(r.get("qc_task", ""))
        final_qc = r.get("final_qc")
        final_qc = None if pd.isna(final_qc) or final_qc == "" else str(final_qc)
        notes = r.get("notes")
        notes = "" if pd.isna(notes) else str(notes)
        rater = r.get("rater_id")
        rater = "" if pd.isna(rater) else str(rater)
        lookup[(pid, ses, qc_task)] = {"final_qc": final_qc or "", "notes": notes, "rater_id": rater}
    return lookup


def lookup_val(lookup: Dict[Key, Dict[str, str]], pid: str, ses: Optional[str], qc_task: str):
    d = lookup.get((pid, ses, qc_task), {})
    qc = d.get("final_qc", "")
    qc = qc if qc in QC_OPTIONS else None
    notes = d.get("notes", "")
    rater = d.get("rater_id", "")
    return qc, notes, rater


def qcrecords_to_clean_df(qc_records: List[QCRecord]) -> pd.DataFrame:
    df = pd.DataFrame([r.model_dump() for r in qc_records])
    for c in OUTPUT_COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[OUTPUT_COLUMNS].copy()
    for c in ["session_id", "task_id", "run_id", "rater_id"]:
        df[c] = df[c].replace({None: "", "None": ""})
    df["notes"] = df["notes"].replace({None: ""})
    return df


def save_records_overwrite_csv(out_file: Path, qc_records: List[QCRecord]) -> Path:
    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    old = read_existing_csv(out_file)
    new_df = qcrecords_to_clean_df(qc_records)

    if new_df.empty and old.empty:
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(out_file, index=False)
        return out_file

    if new_df.empty:
        old.to_csv(out_file, index=False)
        return out_file

    subset = ["participant_id", "session_id", "qc_task"]

    def make_key(df: pd.DataFrame) -> pd.Series:
        tmp = df.copy()
        tmp["session_id"] = tmp["session_id"].fillna("").astype(str)
        tmp["participant_id"] = tmp["participant_id"].fillna("").astype(str)
        tmp["qc_task"] = tmp["qc_task"].fillna("").astype(str)
        return tmp[subset].agg("||".join, axis=1)

    new_keys = set(make_key(new_df))
    if not old.empty:
        old_keys = make_key(old)
        old = old[~old_keys.isin(new_keys)].copy()

    out = pd.concat([old, new_df], ignore_index=True)
    out = out.sort_values(by=["participant_id", "session_id", "qc_task"]).reset_index(drop=True)
    out.to_csv(out_file, index=False)
    return out_file


def main():
    args = parse_args()

    noddireg_dir = Path(args.noddireg_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participants_df = load_participants_df(args.participant_labels)

    st.set_page_config(layout="centered")
    st.markdown('<div id="top_of_page"></div>', unsafe_allow_html=True)
    st.title("AMICO NODDI QC")

    init_session_state()

    out_file = output_dir / "noddireg_QC_status.csv"
    lookup = build_lookup(out_file)

    rater_name = st.text_input("Rater name (optional):", value="")
    typed_rater_id = (rater_name or "").strip()

    total_rows, current_batch = get_current_batch(participants_df, st.session_state.current_page)
    total_pages = max(1, math.ceil(total_rows / 1))

    qc_records: List[QCRecord] = []

    for _, prow in current_batch.iterrows():
        subject = str(prow["participant_id"])
        st.sidebar.markdown(f"### {subject}")

        subj_dir = noddireg_dir / subject
        if not subj_dir.exists():
            st.error(f"Missing directory: {subj_dir}")
            continue

        sessions = detect_sessions(subj_dir, subject)
        if not sessions:
            st.warning(f"No NODDI images found for {subject}")
            continue

        for ses in sessions:
            st.divider()
            st.header(f"{subject} — {ses}")

            ses_prefix = f"{subject}_{ses}" if ses != "no-session" else subject
            session_id = None if ses == "no-session" else ses

            st.subheader("Tissue Density (CSF / GM / WM)")
            density_pngs = list(subj_dir.glob(f"{ses_prefix}*_desc-dsegtissue_model-noddi_density.png"))
            if density_pngs:
                for png in density_pngs:
                    st.image(Image.open(png), use_container_width=True)
                    st.caption(str(png))
            else:
                st.warning("Density plot not found")

            metrics_for_overall: List[MetricQC] = []

            density_task = "noddireg_density"
            stored_qc, stored_notes, stored_rater = lookup_val(lookup, subject, session_id, density_task)
            effective_rater_id = typed_rater_id if typed_rater_id != "" else stored_rater

            density_qc_key = f"{subject}_{ses}_{density_task}_qc"
            density_notes_key = f"{subject}_{ses}_{density_task}_notes"

            density_qc = qc_radio("Density QC", density_qc_key, stored_qc)
            density_notes = notes_box("Density comments", density_notes_key, stored_notes)

            metrics_for_overall.append(MetricQC(name="density", qc=density_qc, notes=density_notes))

            qc_records.append(
                QCRecord(
                    qc_task=density_task,
                    participant_id=subject,
                    session_id=session_id,
                    task_id=None,
                    run_id=None,
                    pipeline="noddireg",
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    rater_id=effective_rater_id,
                    rater_experience=None,
                    rater_fatigue=None,
                    final_qc=density_qc,
                    notes=density_notes,
                )
            )

            st.divider()

            st.subheader("Parcel-wise NODDI Metrics (Schaefer 4S1056)")
            metric_specs = [
                ("od_mean", "noddireg_od", "od", "OD"),
                ("icvf_mean", "noddireg_icvf", "icvf", "ICVF"),
                ("isovf_mean", "noddireg_isovf", "isovf", "ISOVF"),
            ]

            for metric_fname, qc_task, metric_name, label in metric_specs:
                st.markdown(f"**{label}**")

                pngs = sorted(subj_dir.glob(f"{ses_prefix}*_{metric_fname}_qc.png"))
                if pngs:
                    for png in pngs:
                        st.image(Image.open(png), use_container_width=True)
                        st.caption(str(png))
                else:
                    st.warning(f"{label} QA plot not found")

                stored_qc_m, stored_notes_m, stored_rater_m = lookup_val(lookup, subject, session_id, qc_task)
                effective_rater_m = typed_rater_id if typed_rater_id != "" else stored_rater_m

                qc_key = f"{subject}_{ses}_{qc_task}_qc"
                notes_key = f"{subject}_{ses}_{qc_task}_notes"

                qc_val = qc_radio(f"{label} QC", qc_key, stored_qc_m)
                notes_val = notes_box(f"{label} comments", notes_key, stored_notes_m)

                metrics_for_overall.append(MetricQC(name=metric_name, qc=qc_val, notes=notes_val))

                qc_records.append(
                    QCRecord(
                        qc_task=qc_task,
                        participant_id=subject,
                        session_id=session_id,
                        task_id=None,
                        run_id=None,
                        pipeline="noddireg",
                        timestamp=datetime.now().isoformat(timespec="seconds"),
                        rater_id=effective_rater_m,
                        rater_experience=None,
                        rater_fatigue=None,
                        final_qc=qc_val,
                        notes=notes_val,
                    )
                )

                st.divider()

            overall_task = "noddireg_overall"
            overall_stored_qc, overall_stored_notes, overall_stored_rater = lookup_val(
                lookup, subject, session_id, overall_task
            )
            effective_rater_overall = typed_rater_id if typed_rater_id != "" else overall_stored_rater

            overall_notes_key = f"{subject}_{ses}_{overall_task}_notes"
            overall_notes = notes_box("Overall notes", overall_notes_key, overall_stored_notes)

            overall_auto = compute_overall(metrics_for_overall)
            overall_final = overall_auto if overall_auto is not None else overall_stored_qc

            qc_records.append(
                QCRecord(
                    qc_task=overall_task,
                    participant_id=subject,
                    session_id=session_id,
                    task_id=None,
                    run_id=None,
                    pipeline="noddireg",
                    timestamp=datetime.now().isoformat(timespec="seconds"),
                    rater_id=effective_rater_overall,
                    rater_experience=None,
                    rater_fatigue=None,
                    final_qc=overall_final,
                    notes=overall_notes,
                )
            )

    def save_now():
        save_records_overwrite_csv(out_file, qc_records)
        st.success(f"Saved: {out_file}")

    if st.button("💾 Save now"):
        save_now()

    bottom = st.columns((1, 2, 1))

    with bottom[0]:
        st.markdown(f"Page **{st.session_state.current_page}** of **{total_pages}**")

    with bottom[1]:
        c1, c2, c3 = st.columns([1, 1, 1], gap="small")

        if c1.button("⬅️"):
            save_now()
            if st.session_state.current_page > 1:
                st.session_state.current_page -= 1
            do_rerun()

        new_page = c2.number_input(
            "Page",
            min_value=1,
            max_value=total_pages,
            value=st.session_state.current_page,
            step=1,
        )
        if new_page != st.session_state.current_page:
            save_now()
            st.session_state.current_page = new_page
            do_rerun()

        if c3.button("➡️"):
            save_now()
            if st.session_state.current_page < total_pages:
                st.session_state.current_page += 1
            do_rerun()

    with bottom[2]:
        st.markdown("")

    st.markdown(
        """
        <a href="#top_of_page">
            <button style="
                background-color: white;
                border: 1px solid #d1d5db;
                padding: 0.5rem 1rem;
                border-radius: 0.5rem;
                cursor: pointer;">
                ⬆️ Scroll to Top
            </button>
        </a>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
