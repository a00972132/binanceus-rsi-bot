import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from .candidate_strategy import PARAMS, STRATEGY_NOTES
from .evaluator import evaluate_candidate


ROOT_DIR = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT_DIR / "research" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_candidate() -> Dict[str, Any]:
    return dict(PARAMS)


def append_tsv(path: Path, row: Dict[str, Any]) -> None:
    header = (
        "timestamp\tscore\ttuning_mean_return_pct\tholdout_mean_return_pct\tholdout_worst_drawdown_pct\tholdout_mean_profit_factor\tpromote_to_paper\tnotes\n"
    )
    if not path.exists() or not path.read_text().startswith(header):
        path.write_text(header)
    with path.open("a") as file:
        file.write(
            "{timestamp}\t{score:.4f}\t{tuning_mean_return_pct:.4f}\t{holdout_mean_return_pct:.4f}\t{holdout_worst_drawdown_pct:.4f}\t{holdout_mean_profit_factor:.4f}\t{promote_to_paper}\t{notes}\n".format(
                **row
            )
        )


def render_markdown_report(payload: Dict[str, Any]) -> str:
    summary = payload["summary"]
    tuning = summary["tuning"]
    holdout = summary["holdout"]
    lines = [
        f"# Research Report: {payload['tag']}",
        "",
        f"- Timestamp: {payload['timestamp']}",
        f"- Notes: {payload['notes']}",
        f"- Promote to paper: {'YES' if summary['promote_to_paper'] else 'NO'}",
        f"- Overall score: {summary['score']:.4f}",
        "",
        "## Tuning Summary",
        "",
        f"- Mean return: {tuning['mean_return_pct']:.2f}%",
        f"- Worst drawdown: {tuning['worst_drawdown_pct']:.2f}%",
        f"- Mean trades: {tuning['mean_trades']:.2f}",
        f"- Mean profit factor: {tuning['mean_profit_factor']:.2f}",
        f"- Mean expectancy: {tuning['mean_expectancy_pct']:.2f}%",
        "",
        "## Holdout Summary",
        "",
        f"- Mean return: {holdout['mean_return_pct']:.2f}%",
        f"- Worst drawdown: {holdout['worst_drawdown_pct']:.2f}%",
        f"- Mean trades: {holdout['mean_trades']:.2f}",
        f"- Mean profit factor: {holdout['mean_profit_factor']:.2f}",
        f"- Mean expectancy: {holdout['mean_expectancy_pct']:.2f}%",
        "",
        "## Windows",
        "",
    ]
    for window in payload["tuning_windows"] + payload["holdout_windows"]:
        result = window["result"]
        lines.extend(
            [
                f"### {window['label']} ({window['set']})",
                "",
                f"- Return: {result['net_return_pct']:.2f}%",
                f"- Drawdown: {result['max_drawdown_pct']:.2f}%",
                f"- Trades: {int(result['trades'])}",
                f"- Profit factor: {result['profit_factor']:.2f}",
                f"- Expectancy: {result['expectancy_pct']:.2f}%",
                f"- Exposure: {result['exposure_pct']:.2f}%",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one offline research evaluation.")
    parser.add_argument("--tag", default="manual", help="Run label written to result artifacts.")
    args = parser.parse_args()

    params = load_candidate()
    payload = evaluate_candidate(params)
    summary = payload["summary"]
    timestamp = datetime.now(timezone.utc).isoformat()

    out = {
        "timestamp": timestamp,
        "tag": args.tag,
        "notes": STRATEGY_NOTES,
        "params": params,
        "summary": summary,
        "tuning_windows": payload["tuning_windows"],
        "holdout_windows": payload["holdout_windows"],
        "windows": payload["windows"],
    }

    latest_path = RESULTS_DIR / "latest.json"
    dated_path = RESULTS_DIR / f"{args.tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    latest_report_path = RESULTS_DIR / "latest.md"
    dated_report_path = RESULTS_DIR / f"{args.tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.md"
    latest_path.write_text(json.dumps(out, indent=2))
    dated_path.write_text(json.dumps(out, indent=2))
    report = render_markdown_report(out)
    latest_report_path.write_text(report)
    dated_report_path.write_text(report)

    append_tsv(
        RESULTS_DIR / "results.tsv",
        {
            "timestamp": timestamp,
            "score": summary["score"],
            "tuning_mean_return_pct": summary["tuning"]["mean_return_pct"],
            "holdout_mean_return_pct": summary["holdout"]["mean_return_pct"],
            "holdout_worst_drawdown_pct": summary["holdout"]["worst_drawdown_pct"],
            "holdout_mean_profit_factor": summary["holdout"]["mean_profit_factor"],
            "promote_to_paper": str(summary["promote_to_paper"]).lower(),
            "notes": STRATEGY_NOTES.replace("\t", " "),
        },
    )

    print(f"Tag: {args.tag}")
    print(f"Score: {summary['score']:.4f}")
    print(f"Tuning mean return: {summary['tuning']['mean_return_pct']:.2f}%")
    print(f"Holdout mean return: {summary['holdout']['mean_return_pct']:.2f}%")
    print(f"Holdout worst drawdown: {summary['holdout']['worst_drawdown_pct']:.2f}%")
    print(f"Promote to paper: {'YES' if summary['promote_to_paper'] else 'NO'}")
    print(f"Wrote: {dated_path}")


if __name__ == "__main__":
    main()
