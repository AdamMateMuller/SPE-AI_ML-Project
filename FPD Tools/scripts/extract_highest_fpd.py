#!/usr/bin/env python3
# Minimal CLI: extract the HIGHEST (Actual) FPD per section from an Excel sheet.

import argparse, re, sys
import pandas as pd
from pathlib import Path

DRILLING_SECTIONS = ['34"', '22"', '16"', '12-1/4"', '8-3/8"', '6-1/8"']
CASING_SECTIONS    = ['24"', '18-5/8"', '13-3/8"', '9-5/8"', '7"']
SECTION_TYPE = {**{s: "Drilling" for s in DRILLING_SECTIONS},
                **{s: "Casing" for s in CASING_SECTIONS}}
FINAL_ORDER = ['34"', '24"', '22"', '18-5/8"', '16"', '13-3/8"', '12-1/4"', '9-5/8"', '8-3/8"', '7"', '6-1/8"']
ORDER_MAP = {s: i for i, s in enumerate(FINAL_ORDER)}

def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'["\',:()\[\]{}]', '', s)
    s = re.sub(r'\s+', '', s)
    return s

def section_key(sec: str) -> str:
    return _norm(sec).replace('"', '')

def pick_best_fpd_col_for_section(df_cols_norm, sec_norm_key):
    best, best_score = None, -1
    for nc in df_cols_norm:
        if 'fpd' not in nc: continue
        if any(x in nc for x in ['planned', 'plan', 'design', 'target']): continue
        score = 3 if sec_norm_key in nc else (2 if sec_norm_key.replace('-', '') in nc.replace('-', '') else 0)
        if any(t in nc for t in ['actual', 'avg', 'mean']): score += 1
        if score > best_score:
            best_score, best = score, nc
    return best

def extract_highest_fpd(df: pd.DataFrame, header_row: int):
    orig_cols = list(df.columns)
    norm_cols = [_norm(c) for c in orig_cols]
    df.columns = norm_cols
    norm_to_orig = {n: o for n, o in zip(norm_cols, orig_cols)}

    # guess a well id column
    well_col_norm = next((c for c in df.columns if any(tok in c for tok in ['wellid','well','uwi','api','name'])), None)

    # map sections -> chosen normalized FPD column
    sec_to_fpd = {}
    for sec in FINAL_ORDER:
        key = section_key(sec)
        chosen = pick_best_fpd_col_for_section(df.columns, key)
        if chosen: sec_to_fpd[sec] = chosen

    rows = []
    for sec in FINAL_ORDER:
        col = sec_to_fpd.get(sec)
        if not col: continue
        s = pd.to_numeric(df[col], errors='coerce')
        if not s.notna().any(): continue
        idx = s.idxmax()
        val = float(s.loc[idx])
        well_display = str(df.loc[idx, well_col_norm]) if (well_col_norm and well_col_norm in df.columns) else f"Row {int(idx) + 2 + header_row}"
        rows.append({
            "Section": sec,
            "Section Type": SECTION_TYPE.get(sec, ""),
            "FPD (Max)": val,
            "Well ID": well_display,
            "Matched Column": norm_to_orig.get(col, col)
        })

    out = pd.DataFrame(rows)
    if out.empty: return out
    out["__order"] = out["Section"].map(ORDER_MAP)
    out = out.sort_values("__order", kind="stable").drop(columns="__order")
    out.index = range(1, len(out) + 1)
    return out

def main():
    ap = argparse.ArgumentParser(description="Extract highest (actual) FPD per section.")
    ap.add_argument("--excel", required=True, help="Path to Excel file")
    ap.add_argument("--sheet", default="Technical Limit Drilling", help="Sheet name or index")
    ap.add_argument("--header-row", type=int, default=1, help="0-based header row (your Colab used 1)")
    ap.add_argument("--out", default="highest_fpd.csv", help="Output CSV path")
    args = ap.parse_args()

    xls = Path(args.excel)
    if not xls.exists():
        print(f"ERROR: Excel not found: {xls}", file=sys.stderr); sys.exit(2)

    df = pd.read_excel(xls, sheet_name=args.sheet, header=args.header_row)
    out = extract_highest_fpd(df, header_row=args.header_row)

    if out.empty:
        pd.DataFrame([{"Info": "No matching FPD columns found."}]).to_csv(args.out, index=False)
        print("No matching FPD columns found.")
        sys.exit(1)

    out.to_csv(args.out)
    print(out.to_string(index=True))
    print(f"\nSaved -> {args.out}")

if __name__ == "__main__":
    main()
