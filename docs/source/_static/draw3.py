import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Imposta lo stile
sns.set(style="whitegrid")

# Carica i dati
fir = pd.read_csv("fir.out")  # Sostituisci con il tuo path
iir = pd.read_csv("iir.out")

# Aggiungi colonna per tipo di filtro
fir["filter"] = "FIR"
iir["filter"] = "IIR"

# Unisci i dataset
df = pd.concat([fir, iir])

# Raggruppa per canali e tipo filtro, calcolando la media dei tempi
methods = ["gpu", "cpu", "scipy"]
grouped = df.groupby(["channels", "filter"])[methods].mean().reset_index()

# Crea i subplot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)


# Funzione per plottare i dati medi
def plot_means(ax, data, filter_type):
    filter_data = data[data["filter"] == filter_type]
    channels = filter_data["channels"].astype(str)
    width = 0.25
    x = range(len(channels))

    for i, method in enumerate(methods):
        ax.bar(
            [v + width * i for v in x],
            filter_data[method],
            width=width,
            label=method.upper(),
            color=sns.color_palette()[i],
        )

    ax.set_xticks([v + width for v in x])
    ax.set_xticklabels(channels)
    ax.set_xlabel("Number of channels")
    ax.set_ylabel("Execution time (s)")
    ax.set_title(f"{filter_type} filter")


# Plotta i grafici
plot_means(axes[0], grouped, "IIR")
plot_means(axes[1], grouped, "FIR")

# Legenda condivisa
plt.legend(title="Method", loc="upper center", ncol=3)
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig("3.png")
