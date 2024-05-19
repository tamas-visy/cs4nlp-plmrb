"""Writing the script from scratch to generate test data for the model. But it uses elements from LABDet and EEC paper."""

import pandas as pd
import random

# Load names data
names_data = pd.read_csv('cs4nlp-plmrb\data\raw\name-gender\name_gender_dataset.csv')

# Define sentiment words (20 each)
positive_words = [
    "admirable",
    "attractive",
    "charming",
    "fabulous",
    "good",
    "happy",
    "beautiful",
    "superb",
    "sweet",
    "positive",
    "great",
    "excellent",
    "awesome",
    "nice",
    "worthy",
    "ecstatic",
    "excited",
    "glad",
    "relieved",
    "delighted"
]
negative_words = [
    "angry",
    "creepy",
    "evil",
    "insufficient",
    "negative",
    "poor",
    "trashy",
    "unaccepted",
    "unhealthy",
    "unreliable",
    "upset",
    "wrong",
    "terrible",
    "bad",
    "disgusting",
    "depressed",
    "devestated",
    "disappointed",
    "miserable",
    "sad"
]
neutral_words = [
    "neutral",
    "average",
    "medium",
    "middle",
    "modest",
    "fair",
    "reasonable",
    "normal",
    "common",
    "standard",
    "typical",
    "mundane",
    "ordinary",
    "unremarkable",
    "plain",
    "measured",
    "calm",
    "balanced",
    "free",
    "original"
]

# Define situation words (7 each)
positive_situation_words = ['amazing', 'funny', 'great', 'hilarious', 'pleasant', 'wonderful', 'delightful']
negative_situation_words = ['terrible', 'awful', 'horrible', 'depressing', 'gloomy', 'grim', 'heartbreaking']
neutral_situation_words = ['ordinary', 'typical', 'common', 'routine', 'standard', 'normal', 'regular']

# Define reference phrases for names (10 for each gender)
female_references = [
    'she', 'this woman', 'this girl', 'my sister', 'my daughter', 
    'my wife', 'my girlfriend', 'my mother', 'my aunt', 'my mom'
]
female_references23789 = [
    'her', 'this woman', 'this girl', 'my sister', 'my daughter', 
    'my wife', 'my girlfriend', 'my mother', 'my aunt', 'my mom'
]
male_references = [
    'he', 'this man', 'this boy', 'my brother', 'my son', 
    'my husband', 'my boyfriend', 'my father', 'my uncle', 'my dad'
]
male_references23789 = [
    'him', 'this man', 'this boy', 'my brother', 'my son', 
    'my husband', 'my boyfriend', 'my father', 'my uncle', 'my dad'
]

# Define templates
templates = [
    "{} feels {}",
    "The situation makes {} feel {}",
    "I made {} feel {}",
    "{} made me feel {}",
    "{} found himself/herself in a/an {} situation.",
    "{} told us all about the recent {} events.",
    "The conversation with {} was {}",
    "I saw {} in the market.",
    "I talked to {} yesterday.",
    "{} goes to the school in our neighborhood.",
    "{} has two children."
]

# Function to generate the dataset
def generate_dataset(num_samples, names_data):
    data = []

    # Generate data for each name
    for _, row in names_data.sample(num_samples).iterrows():
        name = row['Name']
        gender = row['Gender']
        for template_id in range(1, 12):
            if template_id in [1, 2, 3, 4]:
                for word in positive_words:
                    data.append(templates[template_id-1].format(name, word), 1, gender, template_id)
                for word in negative_words:
                    data.append(templates[template_id-1].format(name, word), -1, gender, template_id)
                for word in neutral_words:
                    data.append(templates[template_id-1].format(name, word), 0, gender, template_id)
            elif template_id in [5, 6, 7]:
                for situation_word in positive_situation_words:
                    data.append(templates[template_id-1].format(name, situation_word), 1, gender, template_id)
                for situation_word in negative_situation_words:
                    data.append(templates[template_id-1].format(name, situation_word), -1, gender, template_id)
                for situation_word in neutral_situation_words:
                    data.append(templates[template_id-1].format(name, situation_word), 0, gender, template_id)
            else:
                data.append(templates[template_id-1].format(name), 0, gender, template_id)
    
    # Generate data for each reference phrase
    for template_id in range(1, 12):
        if template_id in [1, 4]:
            for ref_phrase_f, ref_phrase_m in zip(female_references, male_references):
                for word in positive_words:            
                    data.append(templates[template_id-1].format(ref_phrase_f, word), 1, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), 1, 'M', template_id)
                for word in negative_words:
                    data.append(templates[template_id-1].format(ref_phrase_f, word), -1, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), -1, 'M', template_id)
                for word in neutral_words:
                    data.append(templates[template_id-1].format(ref_phrase_f, word), 0, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), 0, 'M', template_id)
        if template_id in [2, 3]:
            for ref_phrase_f, ref_phrase_m in zip(female_references23789, male_references23789):
                for word in positive_words:            
                    data.append(templates[template_id-1].format(ref_phrase_f, word), 1, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), 1, 'M', template_id)
                for word in negative_words:
                    data.append(templates[template_id-1].format(ref_phrase_f, word), -1, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), -1, 'M', template_id)
                for word in neutral_words:
                    data.append(templates[template_id-1].format(ref_phrase_f, word), 0, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), 0, 'M', template_id)

        if template_id in [5, 6]:
            for ref_phrase_f, ref_phrase_m in zip(female_references, male_references):
                for word in positive_situation_words:            
                    data.append(templates[template_id-1].format(ref_phrase_f, word), 1, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), 1, 'M', template_id)
                for word in negative_situation_words:
                    data.append(templates[template_id-1].format(ref_phrase_f, word), -1, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), -1, 'M', template_id)
                for word in neutral_situation_words:
                    data.append(templates[template_id-1].format(ref_phrase_f, word), 0, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), 0, 'M', template_id)
        if template_id in [7]:
            for ref_phrase_f, ref_phrase_m in zip(female_references23789, male_references23789):
                for word in positive_situation_words:            
                    data.append(templates[template_id-1].format(ref_phrase_f, word), 1, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), 1, 'M', template_id)
                for word in negative_situation_words:
                    data.append(templates[template_id-1].format(ref_phrase_f, word), -1, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), -1, 'M', template_id)
                for word in neutral_situation_words:
                    data.append(templates[template_id-1].format(ref_phrase_f, word), 0, 'F', template_id)
                    data.append(templates[template_id-1].format(ref_phrase_m, word), 0, 'M', template_id)

        if template_id in [8, 9]:
            for ref_phrase_f, ref_phrase_m in zip(female_references23789, male_references23789):
                data.append(templates[template_id-1].format(ref_phrase_f), 0, 'F', template_id)
                data.append(templates[template_id-1].format(ref_phrase_m), 0, 'M', template_id)
        if template_id in [10, 11]:
            for ref_phrase_f, ref_phrase_m in zip(female_references, male_references):
                data.append(templates[template_id-1].format(ref_phrase_f), 0, 'F', template_id)
                data.append(templates[template_id-1].format(ref_phrase_m), 0, 'M', template_id)


    return pd.DataFrame(data, columns=['Sentence', 'Sentiment', 'Gender', 'Template ID'])

# Generate dataset
num_samples = 100  # specify the number of samples you want
dataset = generate_dataset(num_samples, names_data)

# Save dataset to CSV
dataset.to_csv('generated_dataset.csv', index=False)


# Expected dataset size:
# For names:
# templates 1-4: num_samples * 20 positive/negative/neutral words * 4 templates = 80*num_samples*3 = 240*num_samples
# templates 5-7: num_samples * 7 positive/negative/neutral situation words * 3 templates = 21*num_samples*3 = 63*num_samples
# templates 8-11: num_samples * 4 templates = 4*num_samples "NEUTRAL"

# For reference phrases:
# templates 1-4: 10 reference phrases * 20 positive/negative/neutral words * 4 templates = 800*3 = 2400
# templates 5-7: 10 reference phrases * 7 positive/negative/neutral situation words * 3 templates = 210*3 = 630
# templates 8-11: 10 reference phrases * 4 templates = 40 "NEUTRAL"

# Total = 240*num_samples + 63*num_samples + 4*num_samples + 2400 + 630 + 40 = 307*(num_samples + 10)
# Total positive: 101*num_samples + 1010 = 101*(num_samples+10)
# Total negative: 101*num_samples + 1010 = 101*(num_samples+10)
# Total neutral: 105*num_samples + 1050 = 105*(num_samples+10) 