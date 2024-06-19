import json
import os

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
print("Removing random forest")

results = results.drop(results[results['Metric'] == 'Training Accuracy'].index)
results = results.drop(results[results['Metric'] == 'Validation Accuracy'].index)
results = results.drop(results[results['Metric'] == 'Test Accuracy'].index)
print("Removing training and validation and test accuracy")

layermap = dict(initial=0, middle=1, final=2)
results = results.sort_values(by="Group", kind="stable")
results = results.sort_values(by="Layer", key=lambda column: column.map(lambda e: layermap[e]), kind="stable")
results = results.set_index(['LM', 'Probe', 'Layer', 'Metric', 'Group'])

# g2_name, other = 'Probe', 'LM',
g2_name, other = 'LM', 'Probe',

current_results = results["Value"].reset_index()
directory = 'out/images'
os.makedirs(directory, exist_ok=True)

print("#" * 32)
for g2 in current_results[g2_name].unique():
    print(g2)
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    # Flatten the axes array for easy iteration
    axes_i = iter(axes.flatten())

    for metric in sorted(current_results['Metric'].unique()):
        print("\t", metric)
        metric_data = current_results[current_results['Metric'] == metric]

        lm_metric_data = metric_data[metric_data[g2_name] == g2]
        lm_metric_data.loc[:, 'Layer'] = lm_metric_data["Layer"].astype(str)

        # Calculate only DIFFERENCE
        male_lm_metric_data = lm_metric_data[lm_metric_data['Group'] == "Male"].drop(columns="Group")
        male_lm_metric_data = male_lm_metric_data.set_index(['LM', 'Probe', 'Layer', 'Metric'])
        female_lm_metric_data = lm_metric_data[lm_metric_data['Group'] == "Female"].drop(columns="Group")
        female_lm_metric_data = female_lm_metric_data.set_index(['LM', 'Probe', 'Layer', 'Metric'])
        lm_metric_data = (male_lm_metric_data["Value"] - female_lm_metric_data["Value"]).to_frame()
        lm_metric_data = lm_metric_data.reset_index()

        ax = next(axes_i)
        sns.lineplot(data=lm_metric_data, x='Layer', y="Value",
                     # hue='Group',
                     # style=other,  # enabling removes confidence, shows raw lines
                     markers=True, dashes=False,
                     sort=False,
                     ax=ax)
        ax.set_title(metric)
        ax.set_xlabel('')
    fig.suptitle(f'Results of {g2.removesuffix("LanguageModel")}')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    # plt.show()
    plt.savefig(f'{directory}/{g2.removesuffix("LanguageModel")}.png', dpi=300)
