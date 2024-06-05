import json

import pandas as pd
from io import StringIO

# This is printed out by main.py
# TODO save to file, load from file
json_data = """
{
    "Mean error":{
        "('GPT2LanguageModel', 'initial', 'animals', 'sharks')":-0.0640709467,
        "('GPT2LanguageModel', 'initial', 'animals', 'a cute tiny little baby doe')":0.212629454,
        "('GPT2LanguageModel', 'initial', 'animals', 'dogs')":0.3343051582,
        "('GPT2LanguageModel', 'initial', 'animals', 'Donkey and his magical friends')":0.4723072008,
        "('GPT2LanguageModel', 'initial', 'animals', 'cats')":0.548766646,
        "('GPT2LanguageModel', 'initial', 'people', 'an evil dictator')":-0.4128774728,
        "('GPT2LanguageModel', 'initial', 'people', 'Shrek')":0.1805250211,
        "('GPT2LanguageModel', 'initial', 'people', 'US presidents')":0.27674398,
        "('GPT2LanguageModel', 'initial', 'people', 'my family')":0.3916539845,
        "('GPT2LanguageModel', 'initial', 'people', 'Nobel Peace price awarded writers and artists')":0.6057318953,
        "('GPT2LanguageModel', 'middle', 'animals', 'sharks')":0.5712346357,
        "('GPT2LanguageModel', 'middle', 'animals', 'dogs')":0.770981568,
        "('GPT2LanguageModel', 'middle', 'animals', 'cats')":0.9497297417,
        "('GPT2LanguageModel', 'middle', 'animals', 'a cute tiny little baby doe')":1.1381739793,
        "('GPT2LanguageModel', 'middle', 'animals', 'Donkey and his magical friends')":1.1546005364,
        "('GPT2LanguageModel', 'middle', 'people', 'an evil dictator')":0.6847825579,
        "('GPT2LanguageModel', 'middle', 'people', 'US presidents')":0.7335367937,
        "('GPT2LanguageModel', 'middle', 'people', 'my family')":0.8453219614,
        "('GPT2LanguageModel', 'middle', 'people', 'Shrek')":0.88213752,
        "('GPT2LanguageModel', 'middle', 'people', 'Nobel Peace price awarded writers and artists')":1.0586569874,
        "('GPT2LanguageModel', 'final', 'animals', 'sharks')":0.0090118659,
        "('GPT2LanguageModel', 'final', 'animals', 'cats')":0.296679504,
        "('GPT2LanguageModel', 'final', 'animals', 'dogs')":0.3020338263,
        "('GPT2LanguageModel', 'final', 'animals', 'a cute tiny little baby doe')":0.3306549016,
        "('GPT2LanguageModel', 'final', 'animals', 'Donkey and his magical friends')":0.4586913828,
        "('GPT2LanguageModel', 'final', 'people', 'an evil dictator')":-0.1359773493,
        "('GPT2LanguageModel', 'final', 'people', 'US presidents')":0.0040313017,
        "('GPT2LanguageModel', 'final', 'people', 'Nobel Peace price awarded writers and artists')":0.1487343296,
        "('GPT2LanguageModel', 'final', 'people', 'my family')":0.2471062562,
        "('GPT2LanguageModel', 'final', 'people', 'Shrek')":0.2992519473
    }
}"""

with StringIO(json_data) as f:
    results = pd.read_json(f)

results.index = pd.MultiIndex.from_tuples([json.loads(row.replace("(", "[").replace(")", "]").replace("'", '"'))
                                           for row in results.index.to_series()],
                                          names=['lm', 'result_type', 'group', 'subject'])

print(results)

# TODO matplotlib this
