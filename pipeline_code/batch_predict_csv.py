import os
import json
import csv
import argparse
import pandas as pd

from pipeline_code.ensemble_api import (
    fetch_satellite_image,
    ensemble_predict,
    build_output_json
)

# ----------------------------------------------------
# LOAD INPUT FILE (CSV or XLSX)
# ----------------------------------------------------
def load_input_file(filepath):
    if filepath.lower().endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Use CSV or XLSX.")


# ----------------------------------------------------
# PROCESS A FULL CSV/XLSX OF LAT/LON → GENERATE JSONS
# ----------------------------------------------------
def process_file(input_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    df = load_input_file(input_path)

    required_cols = ["sample_id", "latitude", "longitude"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    output_rows = []

    for idx, row in df.iterrows():
        sample_id = row["sample_id"]
        lat = float(row["latitude"])
        lon = float(row["longitude"])

        print(f"\n➡ Processing {sample_id} at ({lat}, {lon})")

        # Fetch satellite image
        img_path = fetch_satellite_image(lat, lon)

        # Run ensemble prediction
        has_solar, conf, boxes, area_sqm, reason = ensemble_predict(img_path)

        # Build JSON
        json_data = build_output_json(
            sample_id=sample_id,
            lat=lat,
            lon=lon,
            has_solar=has_solar,
            confidence=conf,
            bboxes=boxes,
            area_sqm=area_sqm
        )

        # Write JSON file
        json_path = os.path.join(output_folder, f"{sample_id}.json")
        with open(json_path, "w") as jf:
            json.dump(json_data, jf, indent=4)

        print(f"✔ Saved JSON → {json_path}")
        print(f"Reason: {reason}")

        # Append row for summary CSV
        output_rows.append({
            "sample_id": sample_id,
            "latitude": lat,
            "longitude": lon,
            "has_solar": has_solar,
            "confidence": round(conf, 3),
            "pv_area_sqm": round(area_sqm, 2),
            "bbox_count": len(boxes),
            "reason": reason
        })

    # Save summary CSV
    summary_csv = os.path.join(output_folder, "results_summary.csv")
    pd.DataFrame(output_rows).to_csv(summary_csv, index=False)

    print("\n==============================")
    print("      BATCH PROCESS COMPLETE")
    print("==============================")
    print(f"Summary CSV saved → {summary_csv}")
    print(f"All JSONs saved in → {output_folder}")


# ----------------------------------------------------
# Run from command line
# ----------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Input CSV or XLSX file")
    parser.add_argument("--out", default="batch_output", help="Output folder")
    args = parser.parse_args()

    process_file(args.csv, args.out)
