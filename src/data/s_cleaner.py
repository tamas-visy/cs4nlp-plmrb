import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

religion_filters = {"Christianity", "Islam", "Judaism", "Hinduism", "Buddhism"}
race_filters = {"Asian", "Black", "White", "Hispanic", "Latino", "Native American"}
gender_filters = {"male", "female", "non-binary", "transgender"}

def string_match_filter(text, filters):
    matches = []
    for filter_word in filters:
        if re.search(r'\b{}\b'.format(re.escape(filter_word)), text, re.IGNORECASE):
            matches.append(filter_word)
    return matches

def ner_filter(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)
    processed_sentence = []
    for entity in named_entities:
        if isinstance(entity, nltk.tree.Tree):
            if entity.label() == 'PERSON':
                processed_sentence.append('[NAME]')
            elif entity.label() == 'GPE':  # Filter out geopolitical entities
                processed_sentence.append('')  # Replace with empty string
            else:
                # Check for NORP-like entities based on keywords or patterns
                entity_text = ' '.join(word for word, tag in entity.leaves())
                if any(keyword in entity_text.lower() for keyword in ['nation', 'country', 'religion', 'religious']):
                    processed_sentence.append('')  # Replace with empty string
                else:
                    processed_sentence.extend([word for word, tag in entity])
        else:
            processed_sentence.append(entity[0])
    return ' '.join(processed_sentence)


def process_sentence(sentence):
    processed_sentence = ner_filter(sentence)
    # print("Processed sentence:", processed_sentence) # used just for personal name masking checks
    
    religion_matches = string_match_filter(processed_sentence, religion_filters)
    race_matches = string_match_filter(processed_sentence, race_filters)
    gender_matches = string_match_filter(processed_sentence, gender_filters)

    religion_ner_matches = ner_filter(processed_sentence)
    race_ner_matches = ner_filter(processed_sentence)
    gender_ner_matches = ner_filter(processed_sentence)

    print("String matching for religion:", religion_matches)
    print("NER for religion:", religion_ner_matches)
    print("String matching for race:", race_matches)
    print("NER for race:", race_ner_matches)
    print("String matching for gender:", gender_matches)
    print("NER for gender:", gender_ner_matches)
    print()

example_sentences = [
    "The Christian community celebrated Easter.",
    "Asian and Black individuals participated in the event.",
    "A female transgender person gave a speech.",
    "John Smith attended the conference.",
    "The Hindu festival is celebrated with great enthusiasm.",
    "Sheikh Mohammed donated a large sum of money to the mosque.",
    "White supremacists marched through the streets.",
    "The Latino population is growing rapidly.",
    "Transgender rights activists protested outside the courthouse."
]

for sentence in example_sentences:
    print("Processing sentence:", sentence)
    process_sentence(sentence)
