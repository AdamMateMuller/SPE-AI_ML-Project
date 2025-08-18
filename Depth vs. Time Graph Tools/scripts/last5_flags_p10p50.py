#!/usr/bin/env python3
# Last 5 Wells → Flags → P10/P50 Time Calculator (GitHub/CLI)

import argparse, re, warnings, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------- Config --------
N_LAST = 5
MIN_VALID_POINTS = max(3, N_LAST - 2)  # allow a couple of NaNs but need >=3 points

DRILLING_SECTIONS = ['34"', '22"', '16"', '12-1/4"', '8-3/8"', '6-1/8"']
CASING_SECTIONS   = ['24"', '18-5/8"', '13-3/8"', '9-5/8"', '7"']
ALL_SECTIONS = DRILLING_SECTIONS + CASING_SECTIONS

CASING_DEPTH_MAP = {'24"':'34"', '18-5/8"':'22"', '13-3/8"':'16"', '9-5/8"':'12-1/4"', '7"':'8-3/8"'}

# Sequence for time accumulation: D -> C -> D ... -> final D
SECTIONS_ORDER = []
for d, c in zip(DRILLING_SECTIONS[:-1], CASING_SECTIONS):
    SECTIONS_ORDER.extend([d, c])
SECTIONS_ORDER.append(DRILLING_SECTIONS[-1])

def _norm(s: str) -> str:
    s = str(s).lower()
    s = re.sub(r'["\',:()\[\]{}]', '', s)
    s = re.sub(r'\s+', '', s)
    return s

def likely_date_series(sr: pd.Series) -> bool:
    try:
        parsed = pd.to_datetime(sr, errors='coerce', dayfirst=False)
        return parsed.notna().mean() > 0.5
    except Exception:
        return False

def find_best_perwell_sheet(xl: pd.ExcelFile):
    """Pick the sheet+header that looks like per-well FPD table."""
    best = None
    best_score = -1
    for sheet in xl.sheet_names:
        for header in (0, 1, 2):
            try:
                df = xl.parse(sheet, header=header)
            except Exception:
                continue
            if df.empty: 
                continue
            cols = list(df.columns)
            n_fpd = sum('fpd' in _norm(c) for c in cols)
            if n_fpd == 0:
                continue
            n_well = sum(any(t in _norm(c) for t in ['well','uwi','api','name','id']) for c in cols)
            has_date = any(likely_date_series(df[c]) for c in cols)
            n_planned = sum(any(t in _norm(c) for t in ['planned','plan','design','target']) for c in cols)
            score = (n_fpd*5) + (n_well*3) + (2 if has_date else 0) - (1 if n_planned>0 else 0)
            if score > best_score:
                best_score = score
                best = (sheet, header)
    return best

def map_section_columns(df: pd.DataFrame, sections):
    """Map each section label -> most likely ACTUAL FPD column (skip planned/design/target)."""
    norm_cols = [_norm(c) for c in df.columns]
    col_map = {}
    for sec in sections:
        key = _norm(sec).replace('"','')
        best, best_score = None, -1
        for c, nc in zip(df.columns, norm_cols):
            if 'fpd' not in nc:
                continue
            if any(t in nc for t in ['planned','plan','design','target']):
                continue
            score = 0
            if key in nc or key.replace('-','') in nc.replace('-',''):
                score = 3
            else:
                digits = re.sub(r'[^0-9/ -]', '', sec.lower()).strip().replace(' ','')
                if digits and (digits in nc): 
                    score = 2
            if any(t in nc for t in ['actual','avg','mean']):
                score += 1
            if score > best_score:
                best_score, best = score, c
        if best is not None:
            col_map[sec] = best
    return col_map

def pick_well_id_column(df: pd.DataFrame):
    cands = [c for c in df.columns if any(t in _norm(c) for t in ['well','uwi','api','name','id'])]
    return cands[0] if cands else None

def pick_date_column(df: pd.DataFrame):
    for c in df.columns:
        if likely_date_series(df[c]):
            return c
    return None

def compute_trend_flags(lastN_fpd: pd.DataFrame, all_sections, n_last, min_valid_points):
    ATTN_NET_PCT_THRESH = -10.0
    ATTN_MIN_DECREASES  = 2
    rows = []
    for sec in all_sections:
        col = f'{sec} FPD'
        series = lastN_fpd[col].astype(float)
        finite = [v for v in series.tolist() if pd.notna(v)]
        n_valid = len(finite)
        if n_valid < min_valid_points:
            rows.append({
                'Section': sec, 'Points Used': n_valid,
                'Decreases Count': None, f'Net Δ% (last {n_last})': None,
                'OLS Slope (FPD / well#)': None,
                'Needs Attention?': 'Insufficient data',
                'Reason': f'Need ≥{min_valid_points}, have {n_valid}'
            })
            continue
        dec_count = int(sum(finite[i+1] < finite[i] for i in range(n_valid-1)))
        net = float(finite[-1] - finite[0])
        net_pct = float((net / finite[0] * 100.0)) if finite[0] != 0 else np.nan
        x = np.arange(n_valid, dtype=float)
        slope, _ = np.polyfit(x, np.array(finite, dtype=float), 1)

        needs_by_slope = (slope < 0) and (dec_count >= ATTN_MIN_DECREASES)
        needs_by_net   = (not np.isnan(net_pct)) and (net_pct <= ATTN_NET_PCT_THRESH)
        needs = bool(needs_by_slope or needs_by_net)

        reason_bits = []
        if needs_by_slope: reason_bits.append(f"slope<0 & {dec_count} decreases")
        if needs_by_net:   reason_bits.append(f"net {net_pct:.1f}%")
        reason = " | ".join(reason_bits) if needs else ("ok" if dec_count == 0 else "minor dips")

        rows.append({
            'Section': sec,
            'Points Used': n_valid,
            'Decreases Count': dec_count,
            f'Net Δ% (last {n_last})': None if np.isnan(net_pct) else round(net_pct, 2),
            'OLS Slope (FPD / well#)': round(float(slope), 4),
            'Needs Attention?': "YES" if needs else "NO",
            'Reason': reason
        })
    return pd.DataFrame(rows)

def predict_times(depths_map: dict, fpd_profile: dict, sections_order, drilling_sections):
    rows = []
    cum = 0.0
    prev_drill_depth = 0.0
    for sec in sections_order:
        typ = "Drilling" if sec in drilling_sections else "Casing"
        depth = float(depths_map[sec])
        fpd_val = fpd_profile.get(sec, np.nan)

        if typ == "Drilling":
            footage = depth if sec == '34"' else (depth - prev_drill_depth)
            days = float(footage / fpd_val) if pd.notna(fpd_val) and fpd_val > 0 and footage >= 0 else 0.0
            prev_drill_depth = depth
        else:
            footage = depth
            days = float(depth / fpd_val) if pd.notna(fpd_val) and fpd_val > 0 else 0.0

        days = round(days, 4)
        cum = round(cum + days, 4)
        rows.append({
            "Section": sec, "Type": typ, "Depth": depth,
            "FPD": fpd_val, "Footage Used": round(footage, 4),
            "Section Days": days, "Cumulative Days": cum
        })
    df_out = pd.DataFrame(rows)
    total = round(df_out["Section Days"].sum(), 4)
    df_out.loc[len(df_out.index)] = {"Section":"TOTAL","Type":"","Depth":"","FPD":"","Footage Used":"","Section Days":total,"Cumulative Days":total}
    return df_out

def save_plot_p10_vs_p50(pred_p10, pred_p50, out_png, n_last):
    def build_path(pred_df):
        x, y, time, depth = [0.0], [0.0], 0.0, 0.0
        drill_labels=[]
        for _, row in pred_df.iloc[:-1].iterrows():
            if pd.isna(row["Section Days"]) or row["Section Days"] <= 0:
                continue
            time += row["Section Days"]
            if row["Type"] == "Drilling":
                depth = row["Depth"]
                x.append(time); y.append(depth)
                drill_labels.append((time, depth, row["Section"]))
            else:
                x.append(time); y.append(depth)
        return x, y, drill_labels

    x10, y10, _ = build_path(pred_p10)
    x50, y50, labels = build_path(pred_p50)

    plt.figure(figsize=(9,6))
    plt.plot(x10, y10, marker='o', linestyle='-', label=f'P10 (max of last {n_last})')
    plt.plot(x50, y50, marker='o', linestyle='--', label=f'P50 (mean of last {n_last})')
    plt.gca().invert_yaxis()
    for x, y, sec in labels:
        plt.annotate(sec, (x, y), textcoords="offset points", xytext=(-10, -5), ha='right')
    plt.title(f"Predicted Ultimate Drilling Curve — P10 vs P50 (last {n_last} wells)")
    plt.xlabel("Cumulative Days"); plt.ylabel("Section Final Depth")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser(description="Last-5 FPD + Flags + P10/P50 Time Calculator")
    ap.add_argument("--excel", required=True, help="Excel with per-well FPDs")
    ap.add_argument("--sheet", help="Override sheet name (optional)")
    ap.add_argument("--header-row", type=int, help="Override 0-based header row (optional)")
    ap.add_argument("--depths", help="CSV with final (non-cumulative) drilling depths (Section,Depth)")
    ap.add_argument("--out-prefix", default="last5", help="Prefix for outputs")
    args = ap.parse_args()

    # --- Load Excel and pick sheet/header ---
    xl = pd.ExcelFile(args.excel)
    if args.sheet is not None and args.header_row is not None:
        sheet_name, header_row = args.sheet, args.header_row
    else:
        sel = find_best_perwell_sheet(xl)
        if not sel:
            raise ValueError("No per-well FPD sheet detected.")
        sheet_name, header_row = sel
    dfw = xl.parse(sheet_name, header=header_row).dropna(how='all')

    # --- Map section columns to ACTUAL FPD ---
    sec_col_map = map_section_columns(dfw, ALL_SECTIONS)
    if not sec_col_map:
        raise ValueError("Could not match any section FPD columns.")
    well_col = pick_well_id_column(dfw)
    date_col = pick_date_column(dfw)
    fpd_cols = [sec_col_map[s] for s in sec_col_map]

    # Keep rows where NOT ALL FPDs are missing
    dfw2 = dfw.dropna(subset=fpd_cols, how='all').copy()

    # Determine last N wells
    if date_col:
        dfw2['_date'] = pd.to_datetime(dfw2[date_col], errors='coerce')
        dfw2 = dfw2.sort_values('_date', ascending=True)  # oldest -> newest
        lastN = dfw2.tail(N_LAST).copy()
    else:
        lastN = dfw2.tail(N_LAST).copy()

    # --- Build Last-N FPD table ---
    records = []
    for _, r in lastN.iterrows():
        rec = {}
        if well_col: rec['Well'] = r[well_col]
        if date_col: rec['Date'] = r[date_col]
        for sec in ALL_SECTIONS:
            col = sec_col_map.get(sec)
            rec[f'{sec} FPD'] = pd.to_numeric(r[col], errors='coerce') if col else np.nan
        records.append(rec)
    lastN_fpd = pd.DataFrame(records)

    # Save last-5 table
    out_last5 = f"{args.out_prefix}_last{N_LAST}.csv"
    lastN_fpd.to_csv(out_last5, index=False)

    # --- Trend diagnostics + flags ---
    needs_df = compute_trend_flags(lastN_fpd, ALL_SECTIONS, N_LAST, MIN_VALID_POINTS)

    # Strict order (34" … 6-1/8" with casing in between)
    final_order = ['34"', '24"', '22"', '18-5/8"', '16"', '13-3/8"', '12-1/4"', '9-5/8"', '8-3/8"', '7"', '6-1/8"']
    order_map = {s: i for i, s in enumerate(final_order)}
    needs_df['__order'] = needs_df['Section'].map(order_map).fillna(999)
    needs_df = needs_df.sort_values('__order', kind='stable').drop(columns='__order')

    out_flags = f"{args.out_prefix}_flags.csv"
    needs_df.to_csv(out_flags, index=False)

    # --- P10 / P50 profiles ---
    p10, p50 = {}, {}
    for sec in ALL_SECTIONS:
        series = lastN_fpd[f'{sec} FPD'].astype(float).dropna()
        p10[sec] = float(series.max())  if len(series) else np.nan
        p50[sec] = float(series.mean()) if len(series) else np.nan

    p10_df = pd.DataFrame({"Section": ALL_SECTIONS, f"P10 FPD (max last {N_LAST})": [p10[s] for s in ALL_SECTIONS]})
    p50_df = pd.DataFrame({"Section": ALL_SECTIONS, f"P50 FPD (mean last {N_LAST})": [p50[s] for s in ALL_SECTIONS]})
    out_p10 = f"{args.out_prefix}_p10.csv"; p10_df.to_csv(out_p10, index=False)
    out_p50 = f"{args.out_prefix}_p50.csv"; p50_df.to_csv(out_p50, index=False)

    # --- Optional: compute times if depths provided ---
    if args.depths:
        depths_df = pd.read_csv(args.depths)
        depths_map = dict(zip(depths_df["Section"], depths_df["Depth"]))
        # auto-fill casing
        for casing, drill in CASING_DEPTH_MAP.items():
            depths_map[casing] = depths_map[drill]

        pred_p10 = predict_times(depths_map, p10, SECTIONS_ORDER, DRILLING_SECTIONS)
        pred_p50 = predict_times(depths_map, p50, SECTIONS_ORDER, DRILLING_SECTIONS)

        out_pred_p10 = f"{args.out_prefix}_pred_p10.csv"
        out_pred_p50 = f"{args.out_prefix}_pred_p50.csv"
        pred_p10.to_csv(out_pred_p10, index=False)
        pred_p50.to_csv(out_pred_p50, index=False)

        # Plot P10 vs P50
        out_png = f"{args.out_prefix}_p10_vs_p50.png"
        save_plot_p10_vs_p50(pred_p10, pred_p50, out_png, N_LAST)

    print("Done.")
    print(f"- Last-{N_LAST} table: {out_last5}")
    print(f"- Flags:              {out_flags}")
    print(f"- P10 profile:        {out_p10}")
    print(f"- P50 profile:        {out_p50}")
    if args.depths:
        print(f"- Pred P10 times:     {out_pred_p10}")
        print(f"- Pred P50 times:     {out_pred_p50}")
        print(f"- Plot:               {out_png}")

if __name__ == "__main__":
    main()