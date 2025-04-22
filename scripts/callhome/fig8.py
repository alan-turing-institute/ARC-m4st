import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_json("metric_evaluation.json")
df = df[df["originals"].transform(len) > 30].reset_index()

metric1 = "BLASERRefScore"
metric2 = "SacreBLEUScore"

m1s = df[metric1]
df[metric1] = (m1s - m1s.mean()) / m1s.std()
m2s = df[metric2]
df[metric2] = (m2s - m2s.mean()) / m2s.std()

diffs = df[metric1] - df[metric2]
N_largest = 200
N_sampled = 20
diff_largest = diffs.abs().nlargest(N_largest).sample(N_sampled)

df_sampled = df.iloc[diff_largest.index]
df_sampled[[metric1, metric2, "originals", "references", "translations"]].to_csv(
    "df_sampled.csv"
)

plt.scatter(df[metric1], df[metric2])
plt.scatter(df_sampled[metric1], df_sampled[metric2])
plt.plot(range(-2, 4), range(-2, 4), linestyle="--", color="black")
plt.xlabel(metric1)
plt.ylabel(metric2)
plt.savefig("./figure_8_correl.png")
