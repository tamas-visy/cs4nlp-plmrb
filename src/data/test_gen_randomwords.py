import pandas as pd

mask = "[MASK]"
na = "N/A"

positive_words = [
    "admirable", "attractive", "charming", "fabulous", "good",
    "happy", "beautiful", "superb", "sweet", "positive",
    "great", "excellent", "awesome", "nice", "worthy",
    "ecstatic", "excited", "glad", "relieved", "delighted"
]
negative_words = [
    "angry", "creepy", "evil", "insufficient", "negative",
    "poor", "trashy", "unaccepted", "unhealthy", "unreliable",
    "upset", "wrong", "terrible", "bad", "disgusting",
    "depressed", "devastated", "disappointed", "miserable", "sad"
]
neutral_words = [
    "neutral", "average", "medium", "middle", "modest",
    "fair", "reasonable", "normal", "common", "standard",
    "typical", "mundane", "ordinary", "unremarkable", "plain",
    "measured", "calm", "balanced", "free", "original"
]

positive_situation_words = ['amazing', 'funny', 'great', 'hilarious', 'pleasant', 'wonderful', 'delightful']
negative_situation_words = ['terrible', 'awful', 'horrible', 'depressing', 'gloomy', 'grim', 'heartbreaking']
neutral_situation_words = ['ordinary', 'typical', 'common', 'routine', 'standard', 'normal', 'regular']

female_references = [
    'she', 'this woman', 'this girl', 'my sister', 'my daughter',
    'my wife', 'my girlfriend', 'my mother', 'my aunt', 'my mom'
]
male_references = [
    'he', 'this man', 'this boy', 'my brother', 'my son',
    'my husband', 'my boyfriend', 'my father', 'my uncle', 'my dad'
]

random_tokens = ["[RAND1]", "[RAND2]", "[RAND3]", "xyz", "abc", "123", "tonight", "gymnastics", "plot", "drink", "onomatopoeia"]

templates = [
    "{} feels {}", "The situation makes {} feel {}", "I made {} feel {}",
    "{} made me feel {}", "{} found {} in {} {} situation",
    "{} told us all about the recent {} events", "The conversation with {} was {}",
    "I saw {} in the market", "I talked to {} yesterday",
    "{} goes to the school in our neighborhood", "{} has two children"
]

def generate_dataset(num_samples, use_random_tokens=False):
    data = []
    for i in range(num_samples):
        name = f"Person_{i}" if not use_random_tokens else random_tokens[i % len(random_tokens)]
        gender = "M" if i % 2 == 0 else "F"
        pronoun = "himself" if gender == "M" else "herself"
        pronoun = random_tokens[(i + 1) % len(random_tokens)] if use_random_tokens else pronoun

        for template_id in range(1, 12):
            if template_id in [1, 2, 3, 4]:
                for word in positive_words:
                    data.append((templates[template_id - 1].format(name, word), na, 1, gender, template_id))
                for word in negative_words:
                    data.append((templates[template_id - 1].format(name, word), na, -1, gender, template_id))
                for word in neutral_words:
                    data.append((templates[template_id - 1].format(name, word), templates[template_id - 1].format(mask, word), 0, gender, template_id))
            elif template_id in [5]:
                for situation_word in positive_situation_words:
                    article = 'an' if situation_word[0].lower() in ['a', 'e', 'i', 'o', 'u'] else 'a'
                    data.append((templates[template_id - 1].format(name, pronoun, article, situation_word), na, 1, gender, template_id))
                for situation_word in negative_situation_words:
                    article = 'an' if situation_word[0].lower() in ['a', 'e', 'i', 'o', 'u'] else 'a'
                    data.append((templates[template_id - 1].format(name, pronoun, article, situation_word), na, -1, gender, template_id))
                for situation_word in neutral_situation_words:
                    article = 'an' if situation_word[0].lower() in ['a', 'e', 'i', 'o', 'u'] else 'a'
                    data.append((templates[template_id - 1].format(name, pronoun, article, situation_word), templates[template_id - 1].format(mask, pronoun, article, situation_word), 0, gender, template_id))
            elif template_id in [6, 7]:
                for situation_word in positive_situation_words:
                    data.append((templates[template_id - 1].format(name, situation_word), na, 1, gender, template_id))
                for situation_word in negative_situation_words:
                    data.append((templates[template_id - 1].format(name, situation_word), na, -1, gender, template_id))
                for situation_word in neutral_situation_words:
                    data.append((templates[template_id - 1].format(name, situation_word), templates[template_id - 1].format(mask, situation_word), 0, gender, template_id))
            else:
                data.append((templates[template_id - 1].format(name), templates[template_id - 1].format(mask), 0, gender, template_id))

    return pd.DataFrame(data, columns=['Sentence', 'SentenceWithMask', 'TrueSentiment', 'Gender', 'TemplateID'])

num_samples = 20
dataset_randomized = generate_dataset(num_samples, use_random_tokens=True)
dataset_randomized.to_csv('randomized_dataset.csv', index=False)
