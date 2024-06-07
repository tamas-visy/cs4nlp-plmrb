import json

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

with open("out/results.json") as f:
    results = pd.read_json(f)

results.index = pd.MultiIndex.from_tuples([json.loads(row.replace("(", "[").replace(")", "]").replace("'", '"'))
                                           for row in results.index.to_series()],
                                          names=['lm', 'result_type', 'group', 'subject'])

print(results)


def CustomPalette(data, colors):
    # data contains group, subject, etc. as columns
    palettes = {}
    colors_i = iter(colors)

    full = {}
    for group in data["group"].unique():
        palettes[group] = sns.color_palette(next(colors_i), n_colors=data[data["group"] == group]["subject"].nunique())

        assert np.any(data["result_type"] == "final"), "Final results are missing, can't order based on them"
        group_subjects = data[(data["group"] == group) & (data["result_type"] == "final")].sort_values(
            by="Mean error")["subject"].unique()
        for i, subject in enumerate(group_subjects):
            full[subject] = palettes[group][i]
    return full


current_results = results["Mean error"].reset_index()
for lm in current_results['lm'].unique():
    plt.figure(figsize=(10, 6))
    lm_data = current_results[current_results['lm'] == lm]
    lm_data["result_type"] = lm_data["result_type"].astype(str)

    # # sort reversed order by result_type
    # lm_data = lm_data.sort_values(by="Mean error")

    sns.lineplot(data=lm_data, x='result_type', y='Mean error', hue='subject', style='group',
                 markers=True, dashes=False,
                 palette=CustomPalette(data=lm_data, colors=["Blues", "Reds"])
                 )
    plt.title(f'Biases of {lm.removesuffix("LanguageModel")}')
    plt.xlabel('Results at')
    plt.ylabel("Mean error")
    plt.legend(title='Subject', bbox_to_anchor=(1.04, 0.5), loc="center left")
    plt.tight_layout()
    plt.grid(True)
    plt.show()
