#!/usr/bin/env python3
"""
Build Ultimate Drilling Curve from Max FPDs.
- Reads Excel (like in Colab).
- Reads depths from a CSV (instead of input()).
- Outputs results as CSV + PNG.
"""

import argparse, re, warnings
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Config: Sections ----
DRILLING = ['34"', '22"', '16"', '12-1/4"', '8-3/8"', '6-1/8"']
CASING   = ['24"', '18-5/8"', '13-3/8"', '9-5/8"', '7"']
SECTIONS = [d for pair in zip(DRILLING[:-1], CASING) for d in pair] + [DRILLING[-1]]
TYPES = {**{s: "Drilling" for s in DRILLING}, **{s: "Casing" for s in CASING}}
CASING_MAP = {'24"': '34"', '18-5/8"': '22"', '13-3/8"': '16"', '9-5/8"': '12-1/4"', '7"': '8-3/8"'}

def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'["\',:()\[\]{}]', '', s)
    s = re.sub(r'\s+', '', s)
    return s

def find_fpd_column(sec: str, norm_cols, section_types):
    key = _norm(sec)
    best, best_score = None, -1
    for c in norm_cols:
        if "fpd" not in c: continue
        if any(tok in c for tok in ["planned","plan","design","target"]): continue
        score = 0
        if key.replace('"','') in c: score = 3
        if section_types[sec] == "Casing" and any(t in c for t in ["csg","liner","casing"]): score += 1
        if section_types[sec] == "Drilling" and "section" in c: score += 1
        if any(t in c for t in ["actual","avg","mean"]): score += 1
        if score > best_score: best, best_score = c, score
    return best

def main():
    ap = argparse.ArgumentParser(description="Compute Ultimate Drilling Curve.")
    ap.add_argument("--excel", required=True, help="Input Excel file")
    ap.add_argument("--sheet", default="Technical Limit Drilling", help="Sheet name")
    ap.add_argument("--header-row", type=int, default=1, help="0-based header row")
    ap.add_argument("--depths", required=True, help="CSV with final (non-cumulative) drilling depths")
    ap.add_argument("--out-prefix", default="curve", help="Prefix for output files")
    args = ap.parse_args()

    # ---- Load Excel ----
    df = pd.read_excel(args.excel, sheet_name=args.sheet, header=args.header_row)
    orig_cols = list(df.columns)
    norm_cols = [_norm(c) for c in orig_cols]
    df.columns = norm_cols
    norm_to_orig = {n:o for n,o in zip(norm_cols, orig_cols)}

    # ---- Pick best FPD column per section ----
    fpd, matched = {}, {}
    for sec in DRILLING + CASING:
        col = find_fpd_column(sec, norm_cols, TYPES)
        if not col:
            warnings.warn(f"No FPD column for {sec}")
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        mx = series.max(skipna=True)
        if pd.notna(mx):
            fpd[sec] = float(mx)
            matched[sec] = norm_to_orig.get(col, col)

    # ---- Read depths CSV ----
    depths_df = pd.read_csv(args.depths)
    input_depths = dict(zip(depths_df["Section"], depths_df["Depth"]))
    # Auto-fill casing
    for c, d in CASING_MAP.items():
        input_depths[c] = input_depths[d]

    # ---- Calculate section days ----
    rows, cumulative_days, prev_depth = [], 0.0, 0.0
    for sec in SECTIONS:
        typ, depth, max_fpd = TYPES[sec], input_depths.get(sec), fpd.get(sec)
        if not max_fpd or max_fpd <= 0: days, footage = float("nan"), float("nan")
        else:
            if typ == "Drilling":
                footage = depth if sec=="34\"" else depth - prev_depth
                days = footage / max_fpd
                prev_depth = depth
            else:
                footage, days = depth, depth / max_fpd
        cumulative_days += 0 if pd.isna(days) else days
        rows.append({
            "Section": sec, "Type": typ, "Depth": depth,
            "Footage Used": round(footage,4) if pd.notna(footage) else None,
            "Max FPD": max_fpd, "Matched Column": matched.get(sec),
            "Section Days": round(days,4) if pd.notna(days) else None,
            "Cumulative Days": round(cumulative_days,4)
        })
    curve_df = pd.DataFrame(rows)
    total_days = curve_df["Section Days"].dropna().sum()
    curve_df.loc[len(curve_df.index)] = {"Section":"TOTAL","Section Days":total_days,"Cumulative Days":total_days}

    # ---- Save CSV ----
    out_csv = f"{args.out_prefix}.csv"
    curve_df.to_csv(out_csv, index=False)
    print(f"Saved results -> {out_csv}")

    # ---- Plot ----
    x_path, y_path, time, depth = [0],[0],0.0,0.0
    drill_endpoints=[]
    for _,r in curve_df.iloc[:-1].iterrows():
        if pd.isna(r["Section Days"]): continue
        time += r["Section Days"]
        if r["Type"]=="Drilling":
            depth=r["Depth"]; x_path.append(time); y_path.append(depth)
            drill_endpoints.append((time,depth,r["Section"]))
        else:
            x_path.append(time); y_path.append(depth)
    plt.figure(figsize=(8,6))
    plt.plot(x_path,y_path,marker="o")
    plt.gca().invert_yaxis()
    for x,y,s in drill_endpoints:
        plt.annotate(s,(x,y),textcoords="offset points",xytext=(-10,-5),ha="right")
    plt.title(f"Ultimate Drilling Curve — Total Days={round(total_days,2)}")
    plt.xlabel("Cumulative Days"); plt.ylabel("Depth")
    plt.grid(True); plt.tight_layout()
    out_png = f"{args.out_prefix}.png"
    plt.savefig(out_png,dpi=160)
    print(f"Saved plot -> {out_png}")

if __name__=="__main__":
    main()