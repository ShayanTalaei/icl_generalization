"""Plot ICL loss curves from eval_icl.py JSON results.

Usage:
    python scripts/plot_icl.py results/tf_untrained.json results/lstm_untrained.json
    python scripts/plot_icl.py results/*.json -o plots/icl_comparison.png
    python scripts/plot_icl.py results/*.json --logy   # log-scale y axis
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALPHA_BY_TRAINING = {
    "untrained": 0.40,
    "5k": 0.65,
    "10k": 0.80,
    "20k": 1.0,
}


def _model_family(data: dict) -> str:
    """Derive a model family key that distinguishes PE variants."""
    mt = data.get("model_type", "unknown")
    if mt == "transformer":
        pe = data.get("pos_encoding", "learned")
        if pe != "learned":
            return f"transformer_{pe}"
    return mt


FAMILY_STYLE = {
    "transformer":             {"color": "tab:blue",   "ls": "-"},
    "transformer_sinusoidal":  {"color": "tab:green",  "ls": "-"},
    "transformer_rope":        {"color": "tab:purple", "ls": "-"},
    "transformer_none":        {"color": "tab:cyan",   "ls": "-"},
    "lstm":                    {"color": "tab:red",    "ls": "--"},
    "gru":                     {"color": "tab:orange", "ls": "-."},
    "linear_rnn":              {"color": "tab:brown",  "ls": ":"},
    "linear_rnn_proj":         {"color": "tab:pink",   "ls": ":"},
}


def pick_style(label: str, data: dict):
    """Return (color, linestyle, alpha) based on model family and label."""
    family = _model_family(data)
    style = FAMILY_STYLE.get(family, {"color": "tab:gray", "ls": "-"})
    alpha = 1.0
    for key, a in ALPHA_BY_TRAINING.items():
        if key in label:
            alpha = a
            break
    return style["color"], style["ls"], alpha


def main():
    parser = argparse.ArgumentParser(description="Plot ICL loss curves")
    parser.add_argument("files", nargs="+", help="JSON result files from eval_icl.py")
    parser.add_argument("-o", "--output", default="plots/icl_comparison.png",
                        help="Output plot path")
    parser.add_argument("--logy", action="store_true", help="Log-scale y axis")
    parser.add_argument("--title", default="In-Context Learning: Per-Position Loss",
                        help="Plot title")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(10, 6))

    for fpath in sorted(args.files):
        with open(fpath) as f:
            data = json.load(f)
        losses = np.array(data["per_position_loss"])
        positions = np.arange(len(losses))
        label = data["label"]
        color, ls, alpha = pick_style(label, data)
        ax.plot(positions, losses, label=label, color=color,
                linestyle=ls, linewidth=2.2, alpha=alpha)

    ax.set_xlabel("Position in sequence (# demonstrations seen)", fontsize=13)
    ax.set_ylabel("MSE Loss", fontsize=13)
    ax.set_title(args.title, fontsize=14)
    ax.legend(fontsize=11, ncol=2)
    ax.grid(True, alpha=0.3)

    if args.logy:
        ax.set_yscale("log")

    fig.tight_layout()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"Saved plot -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
