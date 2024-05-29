import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

gender_filters = {"male", "female", "non-binary", "transgender", "transsexual", "LGBTQ", "queer"}

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
            else:
                processed_sentence.extend([word for word, tag in entity])
        else:
            processed_sentence.append(entity[0])
    return ' '.join(processed_sentence)

def process_sentence(sentence):
    processed_sentence = ner_filter(sentence)
    
    gender_matches = string_match_filter(processed_sentence, gender_filters)
    
    gender_ner_matches = ner_filter(processed_sentence)

    if '[NAME]' in processed_sentence:
        print("Processed sentence:", processed_sentence)
        print("String matching for gender:", gender_matches)
        print("NER for gender:", gender_ner_matches)
        print()

example_sentences = [
    "A female transgender person gave a speech.",
    "John Smith attended the conference.",
    "Transgender rights activists protested outside the courthouse.",
    "The non-binary community celebrated Pride Month with various events.",
    "She identifies as queer and is an advocate for LGBTQ rights.",
    "The male participants were enthusiastic about the workshop.",
    "A transsexual woman shared her story at the event.",
    "The LGBTQ community held a march in support of equal rights.",
    "He came out as gay during the meeting."
]

for sentence in example_sentences:
    print("Processing sentence:", sentence)
    process_sentence(sentence)
