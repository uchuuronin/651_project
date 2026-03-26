# Generates theoretical threshold figure and reliability diagrams.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.config import RESULTS_DIR, LOGGER
from src.thresholds import tau_U, tau_B


def plot_theoretical(save_path: str):
    lam = np.linspace(0.001,0.25,300)
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(lam, tau_U(lam), label=r"$\tau_U(\lambda)= \lambda/(1+\lambda)$",
            color="tab:blue", linewidth=2)
    ax.plot(lam, tau_B(lam), label=r"$\tau_B(\lambda)= (1+\sqrt{1-4\lambda})/2$",
            color="tab:orange", linewidth=2)

    ax.set_xlabel(r"Abstention cost $\lambda$",fontsize=12)
    ax.set_ylabel(r"Threshold $\tau$",fontsize=12)
    ax.set_title("Theoretical Abstention Thresholds by Loss Function",fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.25)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    out_path = RESULTS_DIR/save_path
    plt.savefig(out_path, bbox_inches="tight")
    LOGGER.info(f"Saved plot_theoretical figure to {out_path}")


def plot_reliability(data_path: str, save_path: str):
    # Plot for verbalized_conf omitted for now as signal degenerate in small model
    df = pd.read_csv(data_path)
    signals = ["token_prob_first", "token_prob_mean"]
    bins = np.linspace(0, 1, 11)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, signal in zip(axes, signals):
        df_clean = df[["correct", signal]].dropna()
        conf = df_clean[signal].values
        correct = df_clean["correct"].astype(int).values
        bin_indices = np.clip(np.digitize(conf, bins) - 1, 0, 9)

        mean_conf, mean_acc, counts = [], [], []
        for b in range(10):
            mask = bin_indices == b
            if mask.sum()==0:
                continue
            mean_conf.append(conf[mask].mean())
            mean_acc.append(correct[mask].mean())
            counts.append(mask.sum())

        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Perfect calibration")
        ax.scatter(mean_conf, mean_acc, s=[c * 2 for c in counts],
                   color="tab:blue", alpha=0.7, zorder=3)
        ax.plot(mean_conf, mean_acc, color="tab:blue", linewidth=1.5)
        ax.set_xlabel("Mean confidence",fontsize=11)
        ax.set_ylabel("Mean accuracy",fontsize=11)
        ax.set_title(signal,fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    title = data_path.split("/")[-1].replace(".csv", "").replace("_", " ")
    plt.suptitle(f"Reliability Diagrams for {title}",fontsize=12)
    plt.tight_layout()

    out_path = RESULTS_DIR/save_path
    plt.savefig(out_path, bbox_inches="tight")
    LOGGER.info(f"Saved plot_reliability figure to {out_path}")


if __name__ == "__main__":
    plot_theoretical("fig_thresholds_theoretical.png")
    plot_reliability("results/inference_triviaqa_500.csv","fig_reliability_diagram.png")