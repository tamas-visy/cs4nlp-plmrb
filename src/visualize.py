import json
import os

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

# with open("out/results.json") as f:
#     results = pd.read_json(f)
#
# results.index = pd.MultiIndex.from_tuples([json.loads(row.replace("(", "[").replace(")", "]").replace("'", '"'))
#                                            for row in results.index.to_series()],
#                                           names=['g2', 'result_type', 'group', 'subject'])

with open("out/vansh.json") as f:
    raw_results = json.load(f)


def traverse(d, keys=None):
    if keys is None:
        keys = []
    if isinstance(d, dict):
        total = {}
        for key, value in d.items():
            total.update(traverse(value, keys + [key]))
        return total
    else:
        return {tuple(keys): d}


results = traverse(raw_results)

results = pd.DataFrame(list(results.items()), columns=['raw_keys', 'Value'])
results[['LM', 'Probe', 'Layer', 'Metric', 'Group']] = pd.DataFrame(results['raw_keys'].tolist(), index=results.index)
results['Group'] = results['Group'].map(dict(F="Female", M="Male"))
results = results.drop(columns=['raw_keys'])
results = results.drop(results[results['Probe'] == 'random_forest'].index)
print("WE ARE DROPPING RANDOM FOREST CUZ ITS SHITTY")
results = results.set_index(['LM', 'Probe', 'Layer', 'Metric', 'Group'])


# print(results)

def CustomPalette(data, colors):
    # data contains group, subject, etc. as columns
    palettes = {}
    colors_i = iter(colors)

    full = {}
    for group in data["Group"].unique():
        palettes[group] = sns.color_palette(next(colors_i), n_colors=data[data["Group"] == group]["subject"].nunique())

        assert np.any(data["Layer"] == "final"), "Final results are missing, can't order based on them"
        group_subjects = data[(data["Group"] == group) & (data["Layer"] == "final")].sort_values(
            by="Value")["subject"].unique()
        for i, subject in enumerate(group_subjects):
            full[subject] = palettes[group][i]
    return full


# g2_name, other = 'Probe', 'LM',
g2_name, other = 'LM', 'Probe',

current_results = results["Value"].reset_index()
directory = 'out/images'
os.makedirs(directory, exist_ok=True)

for g2 in current_results[g2_name].unique():
    print(g2)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    # Flatten the axes array for easy iteration
    axes_i = iter(axes.flatten())

    for metric in sorted(current_results['Metric'].unique()):
        metric_data = current_results[current_results['Metric'] == metric]

        lm_metric_data = metric_data[metric_data[g2_name] == g2]
        lm_metric_data.loc[:, 'Layer'] = lm_metric_data["Layer"].astype(str)

        ax = next(axes_i)
        sns.lineplot(data=lm_metric_data, x='Layer', y="Value",
                     hue='Group',
                     # style=other,  # enabling removes confidence, shows raw lines
                     markers=True, dashes=False,
                     sort=False,
                     # palette=CustomPalette(data=lm_metric_data, colors=["Blues", "Reds"])
                     ax=ax)
        ax.set_title(metric)
        ax.set_xlabel('')
    fig.suptitle(f'Results of {g2.removesuffix("LanguageModel")}')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    # plt.show()
    plt.savefig(f'{directory}/{g2.removesuffix("LanguageModel")}.png', dpi=300)
