import csv
import html
import io
import json


def safe_cell(value):
    if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r", "\n")):
        return "'" + value
    return value


def csv_export(payload):
    out = io.StringIO(newline="")
    writer = csv.writer(out)
    writer.writerow(["source", "kind", "id", "model_label", "model_start_s", "model_end_s", "model_score",
                     "review_decision", "reviewed_label", "reviewed_start_s", "reviewed_end_s", "note"])
    for kind, items in [("event", payload["events"]), ("sleep", payload["sleep"])]:
        for item in items:
            row = [payload["source"], kind, item["id"], item.get("label", item.get("stage")),
                   item["start_s"], item["end_s"], item.get("peak_score"), item["decision"],
                   item.get("reviewed_label"), item.get("reviewed_start_s"), item.get("reviewed_end_s"),
                   item.get("review", {}).get("note", "")]
            writer.writerow([safe_cell(v) for v in row])
    return out.getvalue()


def html_export(payload):
    esc = lambda v: html.escape(str(v))
    rows = "".join(f'<tr><td>{esc(e["label"])}</td><td>{e["start_s"]:.1f}–{e["end_s"]:.1f} s</td>'
                   f'<td>{esc(e["decision"])}</td><td>{esc(e.get("review", {}).get("note", ""))}</td></tr>' for e in payload["events"])
    sleep = "".join(f'<tr><td>{esc(e["stage"])}</td><td>{e["start_s"]:.1f}–{e["end_s"]:.1f} s</td>'
                    f'<td>{esc(e["decision"])}</td><td>{esc(e.get("reviewed_label") or "")}</td></tr>' for e in payload["sleep"])
    return f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>NeuroScope review report</title>
<style>body{{font:15px system-ui;max-width:1000px;margin:40px auto;color:#172033}}table{{border-collapse:collapse;width:100%}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}@media print{{body{{margin:0}}tr{{break-inside:avoid}}}}</style>
<h1>NeuroScope · Review report</h1><p>{esc(payload["recording"]["name"])}</p><p><strong>{esc(payload["notice"])}</strong></p>
<p>Recording-relative timestamps; estimated model intervals. Research use, not medical advice or a final clinical report.</p>
<h2>Event candidates</h2><table><tr><th>Model finding</th><th>Model interval</th><th>Review</th><th>Notes</th></tr>{rows}</table>
<h2>Sleep epochs</h2><table><tr><th>Model stage</th><th>Interval</th><th>Review</th><th>Corrected stage</th></tr>{sleep}</table>
<h2>Coverage and provenance</h2><pre>{esc(json.dumps({"coverage":payload["coverage"],"provenance":payload["provenance"],"config":payload["run"]["config"]},indent=2))}</pre>
<h2>Append-only review history (includes boundary corrections)</h2><pre>{esc(json.dumps(payload["annotations"],indent=2))}</pre></html>'''
