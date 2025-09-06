import os
import sys
import pandas as pd


def read_tsv(path: str, expected_cols):
    df = pd.read_csv(path, sep='\t', keep_default_na=False, na_filter=False, dtype=str)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {path}. Found columns: {list(df.columns)}")
    return df


def main():
    pred_1a = "subtask_1A.tsv"             
    pred_1b = "subtask_1B.tsv"              
    pred_sev = "subtask_1C_severity.tsv"    
    output_path = "subtask_1C.tsv"         

    for path in [pred_1a, pred_1b, pred_sev]:
        if not os.path.exists(path):
            sys.exit(1)

    df_1a = read_tsv(pred_1a, ["id", "label", "model"]) 
    df_1b = read_tsv(pred_1b, ["id", "label", "model"]) 
    df_sev = read_tsv(pred_sev, ["id", "hate_severity", "model"])  # severity

    df_1a = df_1a.rename(columns={"label": "hate_type", "model": "model_1a"})
    df_1b = df_1b.rename(columns={"label": "to_whom", "model": "model_1b"})
    df_sev = df_sev.rename(columns={"model": "model_sev"})

    merged = df_1a.merge(df_sev, on="id", how="inner").merge(df_1b, on="id", how="inner")

    def pick_model(row):
        return row.get("model_sev") or row.get("model_1a") or row.get("model_1b")

    merged["model"] = merged.apply(pick_model, axis=1)

    out_cols = ["id", "hate_type", "hate_severity", "to_whom", "model"]
    out_df = merged[out_cols]

    out_df.to_csv(output_path, sep='\t', index=False)
    print(f"✅ Wrote merged 1C file: {output_path}")
    print(f"   Rows: {len(out_df)}")


if __name__ == "__main__":
    main()
