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
    if not path.exists():
        path.write_text("timestamp\tscore\tmean_return_pct\tworst_drawdown_pct\tpositive_windows\tpromote_to_paper\tnotes\n")
    with path.open("a") as file:
        file.write(
            "{timestamp}\t{score:.4f}\t{mean_return_pct:.4f}\t{worst_drawdown_pct:.4f}\t{positive_windows}\t{promote_to_paper}\t{notes}\n".format(
                **row
            )
        )


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
        "windows": payload["windows"],
    }

    latest_path = RESULTS_DIR / "latest.json"
    dated_path = RESULTS_DIR / f"{args.tag}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    latest_path.write_text(json.dumps(out, indent=2))
    dated_path.write_text(json.dumps(out, indent=2))

    append_tsv(
        RESULTS_DIR / "results.tsv",
        {
            "timestamp": timestamp,
            "score": summary["score"],
            "mean_return_pct": summary["mean_return_pct"],
            "worst_drawdown_pct": summary["worst_drawdown_pct"],
            "positive_windows": summary["positive_windows"],
            "promote_to_paper": str(summary["promote_to_paper"]).lower(),
            "notes": STRATEGY_NOTES.replace("\t", " "),
        },
    )

    print(f"Tag: {args.tag}")
    print(f"Score: {summary['score']:.4f}")
    print(f"Mean return: {summary['mean_return_pct']:.2f}%")
    print(f"Worst drawdown: {summary['worst_drawdown_pct']:.2f}%")
    print(f"Positive windows: {summary['positive_windows']}/{len(payload['windows'])}")
    print(f"Promote to paper: {'YES' if summary['promote_to_paper'] else 'NO'}")
    print(f"Wrote: {dated_path}")


if __name__ == "__main__":
    main()
