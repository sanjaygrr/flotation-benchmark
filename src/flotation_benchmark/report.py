"""Self-contained HTML report (no external assets, charts are inline SVG)."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .benchmark import PROTOCOL_BY_KEY, PROTOCOLS
from .models import ALL_MODELS, LEARNED

MODEL_LABEL = {
    "mean": "Mean",
    "persistence": "Persistence",
    "ridge": "Ridge",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
}
MODEL_COLOR = {
    "mean": "#9aa3ad",
    "persistence": "#c2410c",
    "ridge": "#2563eb",
    "random_forest": "#059669",
    "lightgbm": "#7c3aed",
}

CSS = """
:root{--bg:#f4f1ec;--card:#fffdf9;--ink:#1d232b;--muted:#5d6672;--line:#ddd6cc;
--accent:#8a2c1f;--good:#0f766e;--bad:#b42318}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
font:16px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif}
header{background:#1d232b;color:#f4f1ec;padding:40px 16px 32px}
header .wrap,main{max-width:1080px;margin:0 auto}
header h1{margin:0 0 6px;font-size:30px;letter-spacing:-.01em}
header p{margin:0;color:#c9cfd6}
main{padding:24px 16px 60px}
section{background:var(--card);border:1px solid var(--line);border-radius:10px;
padding:22px 24px;margin:0 0 20px}
h2{margin:0 0 12px;font-size:21px;color:var(--accent)}
h3{margin:18px 0 8px;font-size:17px}
p{margin:0 0 10px}
.muted{color:var(--muted);font-size:14px}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin:6px 0 4px}
.kpi{border:1px solid var(--line);border-radius:8px;padding:14px 16px;background:#fff}
.kpi b{display:block;font-size:26px;line-height:1.2}
.kpi span{color:var(--muted);font-size:14px}
.scroll{overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:14px;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}
thead th{background:#efe9e0;font-weight:600}
tr.sep td{border-top:2px solid #b9b0a3}
td.best{font-weight:700}
.tag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:10px;margin-left:6px}
.tag.leaky{background:#fde7e4;color:var(--bad)}
.tag.honest{background:#dcf2ee;color:var(--good)}
ul{margin:0 0 10px;padding-left:20px}
svg text{font-family:inherit}
code{background:#efe9e0;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def _fmt(m, s=None, d=3):
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "&ndash;"
    out = f"{m:.{d}f}"
    if s is not None and not np.isnan(s):
        out += f" <span class='muted'>&plusmn; {s:.{d}f}</span>"
    return out


def _esc(x) -> str:
    return html.escape(str(x))


def _score_table(summary: pd.DataFrame) -> str:
    rows = []
    for p in PROTOCOLS:
        sub = summary[summary.protocol == p.key]
        if sub.empty:
            continue
        best_mae = sub["mae_mean"].min()
        for i, r in enumerate(sub.itertuples()):
            tag = ("<span class='tag honest'>honest</span>" if p.honest
                   else "<span class='tag leaky'>leaky</span>")
            name = f"{_esc(p.title)}{tag}" if i == 0 else ""
            cls = " class='sep'" if i == 0 else ""
            best = " class='best'" if r.mae_mean == best_mae else ""
            rows.append(
                f"<tr{cls}><td>{name}</td><td>{MODEL_LABEL.get(r.model, r.model)}</td>"
                f"<td{best}>{_fmt(r.mae_mean, r.mae_std)}</td>"
                f"<td>{_fmt(r.rmse_mean, r.rmse_std)}</td>"
                f"<td>{_fmt(r.r2_mean, r.r2_std)}</td><td>{r.n_folds}</td></tr>"
            )
    return (
        "<div class='scroll'><table><thead><tr><th>Protocol</th><th>Model</th>"
        "<th>MAE (% SiO&#8322;)</th><th>RMSE</th><th>R&sup2;</th><th>Folds</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "<p class='muted'>Unweighted mean &plusmn; standard deviation of the per-fold scores "
        "(test folds are of near-equal size). "
        "Bold: lowest MAE within the protocol.</p>"
    )


def _r2_chart(summary: pd.DataFrame) -> str:
    """Grouped horizontal bars of R² by protocol and model."""
    protos = [p for p in PROTOCOLS if p.key in set(summary.protocol)]
    models = [m for m in ALL_MODELS if m in set(summary.model)]
    lo = min(-0.2, float(np.floor(summary["r2_mean"].min() * 5) / 5))
    lo = max(lo, -1.0)
    hi = 1.0
    W, left, right = 760, 230, 20
    bar, gap_group = 11, 14
    h_group = bar * len(models) + gap_group
    H = 40 + h_group * len(protos) + 30
    x = lambda v: left + (min(max(v, lo), hi) - lo) / (hi - lo) * (W - left - right)
    parts = [f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' "
             "aria-label='R squared by protocol and model'>"]
    for t in np.arange(lo, hi + 1e-9, 0.2):
        xx = x(t)
        parts.append(f"<line x1='{xx:.1f}' x2='{xx:.1f}' y1='30' y2='{H - 26}' "
                     f"stroke='{'#1d232b' if abs(t) < 1e-9 else '#e3ddd3'}' />")
        parts.append(f"<text x='{xx:.1f}' y='{H - 8}' font-size='12' text-anchor='middle' "
                     f"fill='#5d6672'>{t:.1f}</text>")
    for gi, p in enumerate(protos):
        y0 = 36 + gi * h_group
        parts.append(f"<text x='{left - 10}' y='{y0 + bar * len(models) / 2 + 4}' font-size='13' "
                     f"text-anchor='end' fill='#1d232b'>{_esc(p.title)}</text>")
        for mi, m in enumerate(models):
            r = summary[(summary.protocol == p.key) & (summary.model == m)]
            if r.empty:
                continue
            v = float(r["r2_mean"].iloc[0])
            x0, x1 = sorted([x(0), x(v)])
            yy = y0 + mi * bar
            clipped = v < lo
            parts.append(f"<rect x='{x0:.1f}' y='{yy}' width='{max(x1 - x0, 1):.1f}' "
                         f"height='{bar - 2}' fill='{MODEL_COLOR[m]}'>"
                         f"<title>{_esc(p.title)} / {MODEL_LABEL[m]}: R2 = {v:.3f}</title></rect>")
            if clipped:
                parts.append(f"<text x='{x0 + 3:.1f}' y='{yy + bar - 3}' font-size='9' "
                             f"fill='#fff'>{v:.2f}</text>")
    lx = left
    for m in models:
        parts.append(f"<rect x='{lx}' y='8' width='12' height='12' fill='{MODEL_COLOR[m]}'/>"
                     f"<text x='{lx + 16}' y='18' font-size='12' fill='#1d232b'>{MODEL_LABEL[m]}</text>")
        lx += 16 + 8 * len(MODEL_LABEL[m]) + 18
    parts.append("</svg>")
    return "".join(parts)


def _drop_table(summary: pd.DataFrame) -> str:
    keys = ["naive", "rows_random_no_iron", "hourly_random_no_iron", "time_holdout", "walk_forward"]
    keys = [k for k in keys if k in set(summary.protocol)]
    if "naive" not in keys:
        return "<p class='muted'>Naive protocol not run.</p>"
    head = "".join(f"<th>{_esc(PROTOCOL_BY_KEY[k].title)}</th>" for k in keys)
    rows = []
    for m in LEARNED:
        cells = []
        for k in keys:
            r = summary[(summary.protocol == k) & (summary.model == m)]
            cells.append(f"<td>{_fmt(float(r.r2_mean.iloc[0])) if len(r) else '&ndash;'}</td>")
        rows.append(f"<tr><td>{MODEL_LABEL[m]}</td>{''.join(cells)}</tr>")
    return (f"<div class='scroll'><table><thead><tr><th>Model (R&sup2;)</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>")


def _persistence_table(summary: pd.DataFrame) -> tuple[str, list[str]]:
    rows, notes = [], []
    for p in PROTOCOLS:
        sub = summary[summary.protocol == p.key].set_index("model")
        if "persistence" not in sub.index:
            continue
        pm = sub.loc["persistence", "mae_mean"]
        learned = sub[sub.index.isin(LEARNED)]
        if learned.empty:
            continue
        best = learned["mae_mean"].idxmin()
        bm = learned.loc[best, "mae_mean"]
        skill = 1 - bm / pm
        pr = sub.loc["persistence", "rmse_mean"]
        best_r = learned["rmse_mean"].idxmin()
        br = learned.loc[best_r, "rmse_mean"]
        rmse_note = (f" In RMSE the best model ({MODEL_LABEL[best_r]}) scores {br:.3f} against "
                     f"{pr:.3f} for persistence.")
        rows.append(
            f"<tr><td>{_esc(p.title)}</td><td>{MODEL_LABEL[best]}</td><td>{bm:.3f}</td>"
            f"<td>{pm:.3f}</td><td style='color:{'#0f766e' if skill > 0 else '#b42318'}'>"
            f"{skill * 100:+.1f} %</td></tr>"
        )
        if p.honest:
            if skill <= 0:
                notes.append(f"Under <b>{_esc(p.title)}</b> no learned model beats persistence "
                             f"(best: {MODEL_LABEL[best]}, MAE {bm:.3f} vs {pm:.3f})." + rmse_note)
            elif skill < 0.10:
                notes.append(f"Under <b>{_esc(p.title)}</b> the best model ({MODEL_LABEL[best]}) "
                             f"beats persistence by only {skill * 100:.1f} % in MAE "
                             f"({bm:.3f} vs {pm:.3f})." + rmse_note)
            else:
                notes.append(f"Under <b>{_esc(p.title)}</b> the best model ({MODEL_LABEL[best]}) "
                             f"beats persistence by {skill * 100:.1f} % in MAE "
                             f"({bm:.3f} vs {pm:.3f})." + rmse_note)
    table = ("<div class='scroll'><table><thead><tr><th>Protocol</th><th>Best learned model</th>"
             "<th>MAE model</th><th>MAE persistence</th><th>MAE skill vs persistence</th></tr></thead>"
             f"<tbody>{''.join(rows)}</tbody></table></div>"
             "<p class='muted'>Skill = 1 &minus; MAE<sub>model</sub> / MAE<sub>persistence</sub>. "
             "Persistence predicts the previous hour's lab value (for D, the current hour's).</p>")
    return table, notes


def _sensitivity_table(pred: pd.DataFrame) -> str:
    """MAE on all test hours vs. only hours whose target is a fresh, non-interpolated value."""
    if "target_repeated" not in pred.columns:
        return ""
    rows = []
    for key in ["time_holdout", "walk_forward", "forecast"]:
        sub = pred[pred.protocol == key]
        if sub.empty:
            continue
        clean = sub[~sub.target_repeated.astype(bool) & ~sub.target_interpolated.astype(bool)]
        models = [m for m in ALL_MODELS if m in sub.columns and sub[m].notna().all()]
        for i, m in enumerate(models):
            def fold_mae(df):
                return df.groupby("fold").apply(
                    lambda g: float(np.mean(np.abs(g["y_true"] - g[m]))), include_groups=False).mean()
            a, c = fold_mae(sub), fold_mae(clean)
            first = i == 0
            rows.append(
                f"<tr{' class=sep' if first else ''}><td>"
                f"{_esc(PROTOCOL_BY_KEY[key].title) if first else ''}</td><td>{MODEL_LABEL[m]}</td>"
                f"<td>{a:.3f}</td><td>{c:.3f}</td>"
                f"<td>{(f'{len(clean)} of {len(sub)}') if first else ''}</td></tr>")
    return ("<div class='scroll'><table><thead><tr><th>Protocol</th><th>Model</th><th>MAE, all test hours</th>"
            "<th>MAE, fresh assays only</th><th>Hours kept</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
            "<p class='muted'>&ldquo;Fresh assays only&rdquo; drops test hours whose target repeats the previous "
            "hour exactly or is interpolated within the hour. Those hours flatter persistence, because a "
            "repeated value gives it zero error. Models are not refitted; only the scored hours change.</p>")


def _fold_table(scores: pd.DataFrame, key: str) -> str:
    sub = scores[scores.protocol == key]
    if sub.empty:
        return ""
    folds = sorted(sub.fold.unique())
    models = [m for m in ALL_MODELS if m in set(sub.model)]
    head = "".join(f"<th>{MODEL_LABEL[m]}</th>" for m in models)
    rows = []
    for f in folds:
        fs = sub[sub.fold == f].set_index("model")
        first = fs.iloc[0]
        cells = "".join(f"<td>{fs.loc[m, 'r2']:+.2f} <span class='muted'>/ {fs.loc[m, 'mae']:.3f}</span></td>"
                        for m in models)
        rows.append(f"<tr><td>{f}</td><td>{_esc(first.test_start[:10])} &rarr; {_esc(first.test_end[:10])} "
                    f"<span class='muted'>({first.n_train} train h)</span></td>{cells}</tr>")
    return (f"<h3>{_esc(PROTOCOL_BY_KEY[key].title)}</h3><div class='scroll'><table><thead><tr><th>Fold</th>"
            f"<th>Test period</th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>")


def _importance_chart(imp: pd.DataFrame, protocol: str, model: str, top: int = 12) -> str:
    sub = imp[(imp.protocol == protocol) & (imp.model == model)]
    if sub.empty:
        return ""
    agg = sub.groupby("feature")["importance"].agg(["mean", "std"]).sort_values("mean", ascending=False)
    agg = agg.head(top)
    W, left, right, bh = 760, 270, 70, 20
    H = 20 + bh * len(agg) + 10
    vmax = max(float((agg["mean"] + agg["std"].fillna(0)).max()), 1e-6)
    vmin = min(0.0, float((agg["mean"] - agg["std"].fillna(0)).min()))
    x = lambda v: left + (v - vmin) / (vmax - vmin) * (W - left - right)
    parts = [f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' "
             f"aria-label='Permutation importance, {protocol}, {model}'>"]
    for i, (feat, r) in enumerate(agg.iterrows()):
        y = 14 + i * bh
        x0, x1 = sorted([x(0), x(r["mean"])])
        parts.append(f"<text x='{left - 8}' y='{y + 13}' font-size='12' text-anchor='end' "
                     f"fill='#1d232b'>{_esc(feat)}</text>")
        parts.append(f"<rect x='{x0:.1f}' y='{y + 2}' width='{max(x1 - x0, 1):.1f}' height='{bh - 6}' "
                     f"fill='{MODEL_COLOR[model]}' opacity='.85'/>")
        if not np.isnan(r["std"]):
            a, b = x(r["mean"] - r["std"]), x(r["mean"] + r["std"])
            parts.append(f"<line x1='{a:.1f}' x2='{b:.1f}' y1='{y + bh / 2 - 1}' y2='{y + bh / 2 - 1}' "
                         "stroke='#1d232b' stroke-width='1.2'/>")
        sd = 0.0 if np.isnan(r["std"]) else r["std"]
        parts.append(f"<text x='{x(max(r['mean'] + sd, 0)) + 6:.1f}' y='{y + 13}' font-size='11' "
                     f"fill='#5d6672'>{r['mean']:.4f}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _prediction_chart(pred: pd.DataFrame, protocol: str = "walk_forward") -> str:
    sub = pred[pred.protocol == protocol]
    if sub.empty:
        return ""
    fold = sub["fold"].max()
    sub = sub[sub.fold == fold].sort_values("date")
    t = pd.to_datetime(sub["date"]).astype("datetime64[ns]")
    series = [("y_true", "Lab assay", "#1d232b", 1.6)]
    for m in ["persistence", "lightgbm"]:
        if m in sub.columns:
            series.append((m, MODEL_LABEL[m], MODEL_COLOR[m], 1.2))
    W, H, left, right, top, bot = 760, 280, 44, 12, 28, 30
    t0, t1 = t.min().value, t.max().value
    vals = np.concatenate([sub[c].to_numpy() for c, *_ in series])
    lo, hi = float(np.floor(vals.min())), float(np.ceil(vals.max()))
    X = lambda tv: left + (tv - t0) / max(t1 - t0, 1) * (W - left - right)
    Y = lambda v: top + (hi - v) / (hi - lo) * (H - top - bot)
    parts = [f"<svg viewBox='0 0 {W} {H}' width='100%' role='img' aria-label='Predictions over time'>"]
    for v in np.arange(lo, hi + 1e-9, 1.0):
        parts.append(f"<line x1='{left}' x2='{W - right}' y1='{Y(v):.1f}' y2='{Y(v):.1f}' stroke='#e3ddd3'/>"
                     f"<text x='{left - 6}' y='{Y(v) + 4:.1f}' font-size='11' text-anchor='end' "
                     f"fill='#5d6672'>{v:.0f}</text>")
    tv = t.astype("int64").to_numpy()
    # break lines across gaps longer than 2 h
    breaks = np.flatnonzero(np.diff(tv) > 2 * 3600e9) + 1
    for col, label, color, w in series:
        v = sub[col].to_numpy()
        for seg in np.split(np.arange(len(v)), breaks):
            if len(seg) < 2:
                continue
            pts = " ".join(f"{X(tv[i]):.1f},{Y(v[i]):.1f}" for i in seg)
            parts.append(f"<polyline points='{pts}' fill='none' stroke='{color}' stroke-width='{w}' "
                         f"opacity='{1 if col == 'y_true' else .8}'/>")
    lx = left
    for col, label, color, _ in series:
        parts.append(f"<rect x='{lx}' y='6' width='14' height='3' fill='{color}'/>"
                     f"<text x='{lx + 18}' y='12' font-size='12' fill='#1d232b'>{label}</text>")
        lx += 26 + 8 * len(label)
    for frac in (0, 0.5, 1):
        tt = pd.Timestamp(int(t0 + frac * (t1 - t0)))
        parts.append(f"<text x='{X(tt.value):.1f}' y='{H - 8}' font-size='11' "
                     f"text-anchor='{'start' if frac == 0 else 'end' if frac == 1 else 'middle'}' "
                     f"fill='#5d6672'>{tt:%Y-%m-%d}</text>")
    parts.append("</svg>")
    return "".join(parts)


def _data_section(meta: dict) -> str:
    d = meta["data"]
    src = meta.get("source") or {}
    gaps = "".join(f"<li>{_esc(g['last_before'])} &rarr; {_esc(g['first_after'])}: "
                   f"{g['missing_hours']} missing hours</li>" for g in d["gaps"]) or "<li>none</li>"
    sph = ", ".join(f"{k} rows: {v} h" for k, v in d["samples_per_hour"].items())
    return f"""
<p>Source: <a href="{_esc(src.get('openml_page', ''))}">{_esc(src.get('name', 'OpenML'))}</a>
(OpenML {_esc(src.get('openml_id', ''))}), license {_esc(src.get('license', 'n/a'))}.
File sha256 <code>{_esc(src.get('sha256', 'n/a'))}</code>.</p>
<ul>
<li>{d['rows_raw']:,} rows of 20 s samples covering {d['hours']:,} distinct hours,
from {_esc(d['start'])} to {_esc(d['end'])}.</li>
<li>Rows per hour: {_esc(sph)}.</li>
<li>Gaps in the hourly index:<ul>{gaps}</ul></li>
<li>{d['hours_target_interpolated']} hours have a target that changes within the hour
(a linear ramp between two assays in the published file); {d['hours_target_repeated']} hours repeat
the previous hour's value exactly.</li>
<li>The feed assays (% Iron Feed, % Silica Feed) change only {d['feed_assay_changes']} times over the
whole period, so they are not hourly measurements.</li>
<li>Correlation between % Iron Concentrate and % Silica Concentrate:
{d['corr_iron_silica_hourly']:.3f} (hourly), {d['corr_iron_silica_rows']:.3f} (20 s rows).
Lag-1 autocorrelation of hourly silica: {d['target_lag1_autocorr_hourly']:.3f}.</li>
</ul>"""


def _headline(summary: pd.DataFrame) -> str:
    def get(p, m, c="r2_mean"):
        r = summary[(summary.protocol == p) & (summary.model == m)]
        return float(r[c].iloc[0]) if len(r) else None

    cards = []
    naive_best = summary[(summary.protocol == "naive") & summary.model.isin(LEARNED)]
    wf_best = summary[(summary.protocol == "walk_forward") & summary.model.isin(LEARNED)]
    if len(naive_best):
        r = naive_best.loc[naive_best.r2_mean.idxmax()]
        cards.append((f"{r.r2_mean:.3f}", f"best R&sup2; under A. Naive ({MODEL_LABEL[r.model]})"))
    if len(wf_best):
        r = wf_best.loc[wf_best.r2_mean.idxmax()]
        cards.append((f"{r.r2_mean:.3f}", f"best R&sup2; under C. Walk-forward ({MODEL_LABEL[r.model]})"))
    p = get("walk_forward", "persistence")
    if p is not None:
        cards.append((f"{p:.3f}", "R&sup2; of persistence under C. Walk-forward"))
    f = summary[(summary.protocol == "forecast") & summary.model.isin(LEARNED)]
    pf = get("forecast", "persistence", "mae_mean")
    if len(f) and pf:
        r = f.loc[f.mae_mean.idxmin()]
        cards.append((f"{(1 - r.mae_mean / pf) * 100:+.1f} %",
                      f"MAE skill over persistence, D. Forecast ({MODEL_LABEL[r.model]})"))
    return "<div class='kpis'>" + "".join(
        f"<div class='kpi'><b>{v}</b><span>{lab}</span></div>" for v, lab in cards) + "</div>"


def build_report(results_dir: Path, out: Path | None = None) -> Path:
    results_dir = Path(results_dir)
    summary = pd.read_csv(results_dir / "summary.csv")
    scores = pd.read_csv(results_dir / "scores.csv")
    meta = json.loads((results_dir / "meta.json").read_text())
    imp_path = results_dir / "importances.csv"
    pred_path = results_dir / "predictions.csv"
    imp = pd.read_csv(imp_path) if imp_path.exists() else None
    pred = pd.read_csv(pred_path) if pred_path.exists() else None

    persist_table, persist_notes = _persistence_table(summary)
    protos = "".join(
        f"<tr><td>{_esc(p.title)}</td><td style='text-align:left;white-space:normal'>"
        f"{_esc(p.description)}</td></tr>" for p in PROTOCOLS
        if p.key in set(summary.protocol))
    cfg = meta["config"]

    imp_html = ""
    if imp is not None:
        for proto in ["walk_forward", "forecast"]:
            for m in ["lightgbm", "random_forest"]:
                chart = _importance_chart(imp, proto, m)
                if chart:
                    imp_html += (f"<h3>{_esc(PROTOCOL_BY_KEY[proto].title)}: {MODEL_LABEL[m]}</h3>"
                                 f"{chart}")
        imp_html += ("<p class='muted'>Permutation importance on each test fold: mean increase in MAE "
                     "when the feature is shuffled, averaged over folds (bar) &plusmn; one standard "
                     "deviation across folds (line). Top 12 features.</p>")

    pred_html = _prediction_chart(pred) if pred is not None else ""

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flotation soft-sensor benchmark</title><style>{CSS}</style></head>
<body>
<header><div class="wrap">
<h1>Flotation soft-sensor benchmark</h1>
<p>% Silica Concentrate on the public iron-ore flotation dataset, scored under leaky and
leakage-aware protocols. flotation-benchmark {_esc(meta['version'])}, run finished
{_esc(meta['finished'])}.</p>
</div></header>
<main>
<section><h2>Summary</h2>
{_headline(summary)}
<ul style="margin-top:12px">{''.join(f'<li>{n}</li>' for n in persist_notes)}</ul>
</section>

<section><h2>Scores by protocol and model</h2>
{_score_table(summary)}
</section>

<section><h2>R&sup2; across protocols</h2>
{_r2_chart(summary)}
<p class="muted">Bars below the axis minimum are clipped and labelled with their value.</p>
</section>

<section><h2>From naive to honest</h2>
<p>R&sup2; of each learned model as the leaks are removed one at a time: drop the iron column (A1),
collapse the repeated 20 s rows to hourly means (A2), keep time order (B), then walk forward with a
{cfg['gap_hours']} h gap (C).</p>
{_drop_table(summary)}
</section>

<section><h2>Walk-forward folds</h2>
<p class="muted">R&sup2; / MAE per fold. Each fold is refitted on all hours that end at least the gap before
its test block.</p>
{_fold_table(scores, "walk_forward")}
{_fold_table(scores, "forecast")}
</section>

<section><h2>Comparison with persistence</h2>
{persist_table}
</section>

<section><h2>Sensitivity to repeated and interpolated targets</h2>
{_sensitivity_table(pred) if pred is not None else ""}
</section>

<section><h2>Feature importance under the honest protocols</h2>
{imp_html or "<p class='muted'>Not computed in this run.</p>"}
</section>

<section><h2>Walk-forward predictions (last fold)</h2>
{pred_html or "<p class='muted'>No predictions saved.</p>"}
</section>

<section><h2>Protocols</h2>
<div class="scroll"><table><thead><tr><th>Protocol</th><th style="text-align:left">Definition</th></tr></thead>
<tbody>{protos}</tbody></table></div>
<p class="muted">Folds: {cfg['n_folds']}; gap: {cfg['gap_hours']} h; seed: {cfg['seed']};
20 s row stride: {cfg['rows_stride']}.</p>
</section>

<section><h2>Data</h2>
{_data_section(meta)}
<p class="muted">Python {_esc(meta['python'])};
{_esc(', '.join(f'{k} {v}' for k, v in meta['packages'].items()))}.</p>
</section>
</main></body></html>
"""
    out = Path(out) if out else results_dir / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out
