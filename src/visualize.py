import json

import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt

with open("out/results.json") as f:
    results = pd.read_json(f)

results.index = pd.MultiIndex.from_tuples([json.loads(row.replace("(", "[").replace(")", "]").replace("'", '"'))
                                           for row in results.index.to_series()],
                                          names=['lm', 'result_type', 'group', 'subject'])

print(results)

current_results = results["Mean error"].reset_index()
for lm in current_results['lm'].unique():
    plt.figure(figsize=(10, 6))
    lm_data = current_results[current_results['lm'] == lm]
    lm_data["result_type"] = lm_data["result_type"].astype(str)

    colors = iter(["Blues", "Reds"])
    for i, group in enumerate(lm_data['group'].unique()):
        group_lm_data = lm_data[current_results['group'] == group]

        sns.lineplot(data=group_lm_data, x='result_type', y='Mean error', hue='subject', style='group',
                     markers=True, dashes=False,
                     palette=sns.color_palette(next(colors), n_colors=len(group_lm_data['subject'].unique())))

    plt.title(f'Biases of {lm.removesuffix("LanguageModel")}')
    plt.xlabel('Results at')
    plt.ylabel("Mean error")
    plt.legend(title='Subject')
    plt.grid(True)
    plt.show()
