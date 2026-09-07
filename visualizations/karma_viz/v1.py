import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def plot_karma_explanation(karma_output):
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1.2, 1])

    # -----------------------
    # Level 1: Variable Importance
    # -----------------------
    ax1 = fig.add_subplot(gs[0, 0])
    vi = karma_output["variable_importance"]

    names = list(vi.keys())
    values = list(vi.values())

    ax1.barh(names, values, color="steelblue")
    ax1.invert_yaxis()
    ax1.set_title("Variable Importance")

    for i, v in enumerate(values):
        ax1.text(v + 0.01, i, f"{v:.2f}", va="center")

    # -----------------------
    # Level 2: Lag Profiles
    # -----------------------
    ax2 = fig.add_subplot(gs[0, 1])
    for var, vals in karma_output["lag_profiles"].items():
        ax2.plot(range(1, len(vals) + 1), vals, marker="o", label=var)

    ax2.set_title("Lag Profiles")
    ax2.set_xlabel("Lag k")
    ax2.legend()

    # -----------------------
    # Level 3: Heatmap
    # -----------------------
    ax3 = fig.add_subplot(gs[1, 0:2])
    tk = karma_output["transition_kernel"]

    sns.heatmap(
        tk["matrix"],
        annot=True,
        cmap="Blues",
        xticklabels=[f"s={i}" for i in range(tk["matrix"].shape[1])],
        yticklabels=tk["histories"],
        ax=ax3,
    )
    ax3.set_title("Transition Kernel")

    # Coverage
    for i, c in enumerate(tk["coverage"]):
        ax3.text(tk["matrix"].shape[1] + 0.2, i + 0.5, f"{c:.2f}", va="center")

    # -----------------------
    # Level 3: DAG
    # -----------------------
    ax4 = fig.add_subplot(gs[0:2, 2])
    G = nx.DiGraph()
    G.add_edges_from(karma_output["dag"]["edges"])

    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color="lightblue", ax=ax4)
    ax4.set_title("Causal DAG")

    # -----------------------
    # Level 4: Causal Effects
    # -----------------------
    ax5 = fig.add_subplot(gs[2, 0])
    ce = karma_output["causal_effects"]

    ax5.barh(ce["labels"], ce["values"], color="steelblue")
    ax5.axvline(ce["threshold"], color="red", linestyle="--")
    ax5.set_title("Causal Effects")

    # -----------------------
    # Level 4: Uncertainty
    # -----------------------
    ax6 = fig.add_subplot(gs[2, 1])
    un = karma_output["uncertainty"]

    ax6.bar(un["histories"], un["entropy"], color="gray")
    ax6.set_title("Aleatoric Entropy")

    plt.tight_layout()
    plt.show()
