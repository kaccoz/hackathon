from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from rail_cdm.features import extract_features
from rail_cdm.io import ALLOWED_LABELS, read_sensor_csv

MODEL_PATH = Path("artifacts/rail_model.joblib")

st.set_page_config(page_title="Rail Corrugation Monitor", page_icon="🚆", layout="wide")
st.title("Rail Corrugation Monitor")
st.write(
    "Upload one or more one-second axle-box sensor recordings. "
    "The model classifies each recording as Normal, Side I, or Side II."
)

if not MODEL_PATH.exists():
    st.warning("Train the baseline first. The app expects artifacts/rail_model.joblib.")
    st.stop()

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

uploads = st.file_uploader("Rail sensor CSV files", type="csv", accept_multiple_files=True)

if uploads:
    rows: list[dict[str, object]] = []
    details: list[dict[str, object]] = []

    with st.spinner("Extracting signal features and making predictions..."):
        for upload in uploads:
            try:
                frame = read_sensor_csv(upload)
                features = extract_features(frame)
                feature_frame = pd.DataFrame([features]).reindex(columns=feature_columns)
                prediction = str(model.predict(feature_frame)[0])
                probabilities = model.predict_proba(feature_frame)[0]
                confidence_by_class = dict(zip(model.classes_, probabilities, strict=True))
                if prediction not in ALLOWED_LABELS:
                    raise ValueError(f"Unexpected model output: {prediction}")

                rows.append({"file_id": upload.name, "prediction": prediction})
                details.append(
                    {
                        "file": upload.name,
                        "prediction": prediction,
                        **{f"P({name})": value for name, value in confidence_by_class.items()},
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one bad upload should not hide good files.
                st.error(f"{upload.name}: {exc}")

    if rows:
        predictions = pd.DataFrame(rows, columns=["file_id", "prediction"])
        st.subheader("Predictions")
        st.dataframe(pd.DataFrame(details).style.format(precision=3), use_container_width=True)

        csv_bytes = predictions.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download rail_predictions.csv",
            data=BytesIO(csv_bytes),
            file_name="rail_predictions.csv",
            mime="text/csv",
        )
