# Final Project Report — Depth vs. Time Tools

> Course: SPE AI/ML  
> Author: Adam Muller  
> Repo: <https://github.com/AdamMateMuller/SPE-AI_ML-Project>   
> Date: 2025-08-18

---

## 1. Problem Definition
Drilling teams rely on **depth vs. time** graphs to plan operations efficiently and cost-effectively.  
**Goal:** provide a simple, reproducible Python toolkit that:
- extracts per-section **Feet-Per-Day (FPD)** performance,
- flags sections with degrading trends,
- builds **P10 (best)** and **P50 (average)** FPD profiles, and
- converts these into predicted **section times** and **cumulative depth-vs-time** curves.

**Scope & assumptions:** section-based analysis (e.g., 34" → 6-1/8"); FPD drives durations; casing time modeled as plateaus; inputs are public/sample/synthetic.

---

## 2. Background
This review was previously done manually in **Microsoft Excel**, which is slow, error-prone, and hard to repeat. The repository automates the workflow in Python so results are consistent, fast to update, and easy to reproduce under version control.

---

## 3. Data Sources
- **Primary dataset:** _Sample/Fictional_ per-well Excel table of sectional FPD values  
  `data/Historical Drilling Performance & Technical Limit Drilling Calculations.xlsx`
- **Depths file:** final (non-cumulative) **drilling** depths per section  
  `data/depths.csv` (example):
Section,Depth
34",1714
22",3213
16",7420
12-1/4",9118
8-3/8",13709
6-1/8",18993

markdown
Copy
Edit
- **Privacy/Ethics:** only sample/synthetic data used; no confidential identifiers.

---

## 4. Methods & Workflow
1. **Load & detect:** read Excel, normalize headers, map FPD columns to sections (robust matching; skip planned/design/target).
2. **Last-N extraction:** take the **last 5 wells** (by date if available; else by order).
3. **Diagnostics & flags:** per section compute OLS slope, count decreases, net % change; flag **Needs Attention** if  
 *(slope < 0 and ≥ 2 decreases) OR (net change ≤ −10%)* over last 5 wells.  
 (Allow some missing values; require ≥3 valid points.)
4. **Profiles:** build **P10 = max(last 5)** and **P50 = mean(last 5)** FPD per section.
5. **Time prediction:** convert FPD → **days** using final drilling depths; casing time uses plateau logic.
6. **Visualization:** export **Depth vs. Time** curves (P10 vs P50) and CSV/PNG artifacts.

---

## 5. Code Structure (summary of the 3 scripts)
Depth vs. Time Graph Tools/
├─ scripts/
│ ├─ extract_highest_fpd.py # audit: highest actual FPD per section + source column
│ ├─ drilling_curve.py # depth-vs-time curve from max FPD + depths.csv
│ └─ last5_flags_p10p50.py # last-5 table, trend flags, P10/P50 profiles, predicted times, plot
├─ data/
│ ├─ Historical Drilling Performance & Technical Limit Drilling Calculations.xlsx
│ └─ depths.csv
├─ out/ # generated CSV/PNG results
├─ requirements.txt
└─ Final_Project_Report.md

pgsql
Copy
Edit

**How they work (brief):**
- `extract_highest_fpd.py` — finds the best “actual” FPD column per section (skips planned/design), reports **max FPD** with the original header for traceability.  
- `drilling_curve.py` — uses **max FPD** and `depths.csv` to compute **section days** and an overall **depth-vs-time** curve; saves table + PNG.  
- `last5_flags_p10p50.py` — auto-detects per-well sheet/header, builds **last-5** sectional FPD table, computes **Needs Attention** flags, creates **P10/P50** FPD profiles, converts to **predicted times**, and plots **P10 vs P50** curves.

---

## 6. Reproducibility — How to Run
> Run these from **inside** the folder: `cd "Depth vs. Time Graph Tools"`

```bash
# Install once
pip install -r requirements.txt

# (A) Highest FPD (per section)
python scripts/extract_highest_fpd.py \
  --excel "data/Historical Drilling Performance & Technical Limit Drilling Calculations.xlsx" \
  --sheet "Technical Limit Drilling" --header-row 1 \
  --out out/highest_fpd.csv

# (B) Drilling curve (max FPD + depths)
python scripts/drilling_curve.py \
  --excel "data/Historical Drilling Performance & Technical Limit Drilling Calculations.xlsx" \
  --sheet "Technical Limit Drilling" --header-row 1 \
  --depths data/depths.csv \
  --out-prefix out/curve

# (C) Last-5 → Flags → P10/P50 (+ times & plot)
python scripts/last5_flags_p10p50.py \
  --excel "data/Historical Drilling Performance & Technical Limit Drilling Calculations.xlsx" \
  --depths data/depths.csv \
  --out-prefix out/last5
7. Results (from /out)
Audit & extraction

out/highest_fpd.csv — max actual FPD per section + matched source column.

Last-5 analysis & flags

out/last5_last5.csv — last-5 sectional FPDs

out/last5_flags.csv — Needs Attention? per section with reasons (slope/decreases/net %)

out/last5_p10.csv, out/last5_p50.csv — FPD profiles

Predicted time curves

out/last5_pred_p10.csv, out/last5_pred_p50.csv — section days & cumulative days

out/last5_p10_vs_p50.png — P10 vs P50 depth-vs-time plot (drilling labels)

out/curve.csv, out/curve.png — curve from overall max FPD (sanity check)

Summary: The toolkit reproduces the Excel workflow programmatically, highlights declining sections early, and provides P10/P50 time windows to plan against. All steps are version-controlled and repeatable.

8. Performance Analysis
Diagnostics: OLS slope sign, decreases count, net % change over last 5 wells (per section).

Time accuracy (when historical totals exist): compare predicted total days vs actual; report MAE / % error.

Sensitivity: vary last-N window (e.g., 3–7) and observe P10/P50 shifts.

Limitations: sensitivity to missing/outlier FPD values; relies on realistic section depths.

9. Conclusion
The Depth vs. Time Tools provide a lightweight, repeatable workflow to:

audit and extract per-section FPD,

flag sections needing attention, and

generate practical depth-vs-time scenarios (P10/P50) for planning.

This replaces manual Excel with scripted, reproducible analysis.

10. Way Forward
Implement the same or similar solution against larger or real-time operator databases (APIs/DBs), add robust statistics (outlier guards, time-decay), extend profiles (e.g., P25/P75), and automate scheduled reporting/dashboards.

11. References
Course project: https://github.com/saputelli/SPE-AI_ML-Course/tree/main/Project