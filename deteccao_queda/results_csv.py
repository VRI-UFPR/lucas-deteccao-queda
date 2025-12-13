import os
import re
import pandas as pd

folder = "results"

# regex antigo: "0.5327 (0.0376)"
metric_pattern_old = re.compile(r"([0-9.]+)\s*\(([-0-9.]+)\)")

# regex novo: "Acurácia:  0.7861"
metric_pattern_new = re.compile(
    r"(Acurácia|Sensibilidade|Especificidade|F1-score)\s*:\s*([0-9.]+)"
)

rows = []

for file in os.listdir(folder):
    if not file.endswith(".txt"):
        continue

    try:
        type_fall, frame, model_txt = file.split("_")
        model = model_txt.replace(".txt", "")
    except ValueError:
        print(f"Arquivo ignorado (formato inesperado): {file}")
        continue

    path = os.path.join(folder, file)
    with open(path, "r") as f:
        text = f.read()

    # Primeiro tenta o formato antigo
    metrics_old = metric_pattern_old.findall(text)

    if len(metrics_old) == 4:
        # Formato antigo OK
        acc_mean, acc_std = metrics_old[0]
        sen_mean, sen_std = metrics_old[1]
        spe_mean, spe_std = metrics_old[2]
        f1_mean,  f1_std  = metrics_old[3]

    else:
        # Tenta formato novo
        metrics_new = metric_pattern_new.findall(text)

        if len(metrics_new) != 4:
            print(f"Formato inesperado em {file}")
            continue

        # Converte para dicionário organizado
        metric_map = {name: float(val) for name, val in metrics_new}

        acc_mean = metric_map["Acurácia"]
        sen_mean = metric_map["Sensibilidade"]
        spe_mean = metric_map["Especificidade"]
        f1_mean  = metric_map["F1-score"]

        # No novo formato NÃO existe desvio padrão → definir como None ou 0
        acc_std = sen_std = spe_std = f1_std = 0.0

    rows.append({
        "type": type_fall,
        "frame": int(frame),
        "model": model,
        "accuracy_mean": float(acc_mean),
        "accuracy_std": float(acc_std),
        "sensitivity_mean": float(sen_mean),
        "sensitivity_std": float(sen_std),
        "specificity_mean": float(spe_mean),
        "specificity_std": float(spe_std),
        "f1_mean": float(f1_mean),
        "f1_std": float(f1_std),
    })

df = pd.DataFrame(rows)
df = df.sort_values(by=["type", "frame", "model"])

df.to_csv("results_summary.csv", index=False)
print("Arquivo gerado: results_summary.csv")

