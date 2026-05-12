import pandas as pd

from sklearn.metrics import cohen_kappa_score


human = pd.read_csv("phase-b/human_labels.csv")[
    "human_winner"
].tolist()

judge = pd.read_csv("phase-b/pairwise_results.csv")[
    "winner_after_swap"
].head(10).tolist()

kappa = cohen_kappa_score(human, judge)

print(f"Cohen's kappa: {kappa:.3f}")

if kappa < 0:
    print("Worse than chance")
elif kappa < 0.2:
    print("Slight agreement")
elif kappa < 0.4:
    print("Fair agreement")
elif kappa < 0.6:
    print("Moderate agreement")
elif kappa < 0.8:
    print("Substantial agreement")
else:
    print("Almost perfect agreement")