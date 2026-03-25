"""Convert JSON-lines benchmark results to a markdown table.

Usage:
    uv run python -m cs336_systems.results_to_markdown results.jsonl -o results.md
"""
import argparse
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Path to JSON-lines results file")
    parser.add_argument("-o", "--output", default=None, help="Output markdown file (default: print to stdout)")
    args = parser.parse_args()

    with open(args.input) as f:
        records = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(records)

    # format the time column as "mean ± std"
    df["time (s)"] = df.apply(lambda r: f"{r['mean_time']:.4f} ± {r['std_time']:.4f}", axis=1)
    df = df.drop(columns=["mean_time", "std_time"])

    md = df.to_markdown(index=False)

    if args.output:
        with open(args.output, "w") as f:
            f.write(md + "\n")
        print(f"Written to {args.output}")
    else:
        print(md)


if __name__ == "__main__":
    main()
