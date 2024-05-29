import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Word list taken from:
# https://github.com/gender-bias/gender-bias/tree/master/genderbias: job-posting-specific male and female attributes
# https://github.com/davidemiceli/gender-detection/: personal names
# https://github.com/microsoft/responsible-ai-toolbox-genbit/tree/main: gendered professions and adjectives related to cis, non-binary and trans people
# https://github.com/tolga-b/debiaswe/tree/master: general male and female coded words (implicit gender bias descriptors) and professions
# TO DEBUG: superlatives only used for women/men, expand on unnecessarly citing personal infomration or gender references when not required, expand on jobs dataset, replace to neutral counterpart
# WHAT DOESN'T MAKE SENSE: only ocnsidering adjectives that are present in more than 2 lists - too occlusive

def read_names(file_path):
    names = set()
    with open(file_path, 'r') as file:
        for line in file:
            matches = re.findall(r'"([^"]+)"', line)
            if matches:
                names.update(matches)
    return [name.lower() for name in names]

female_names = read_names("/workspaces/cs4nlp-plmrb/data/raw/female.js")
male_names = read_names("/workspaces/cs4nlp-plmrb/data/raw/male.js")

def read_wordlist(file_path):
    with open(file_path, 'r') as file:
        return {word.strip() for word in file}

female_attributes_filters = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/female_adjectives.wordlist")
male_attributes_filters = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/male_adjectives.wordlist")
female_jobs_filters = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/female.txt")
male_jobs_filters = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/male.txt")

def string_match_filter(text, filters):
    matches = []
    for filter_word in filters:
        pattern = re.compile(r'\b{}\w*\b'.format(re.escape(filter_word)), re.IGNORECASE)
        if re.search(pattern, text):
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
                entity_words = []
                for word, tag in entity.leaves():
                    if word.lower() in female_names:
                        entity_words.append('[FEMALE NAME]')
                    elif word.lower() in male_names:
                        entity_words.append('[MALE NAME]')
                    elif word.lower() in female_jobs_filters:
                        entity_words.append('[FEMALE JOB]')
                    elif word.lower() in male_jobs_filters:
                        entity_words.append('[MALE JOB]')
                    for attribute_filter in female_attributes_filters:
                        pattern = re.compile(r'\b{}\w*\b'.format(re.escape(attribute_filter)), re.IGNORECASE)
                        if re.match(pattern, word):
                            entity_words.append('[FEMALE ATTRIBUTES]')
                            break
                    for attribute_filter in male_attributes_filters:
                        pattern = re.compile(r'\b{}\w*\b'.format(re.escape(attribute_filter)), re.IGNORECASE)
                        if re.match(pattern, word):
                            entity_words.append('[MALE ATTRIBUTES]')
                            break
                    else:
                        entity_words.append(word)
                processed_sentence.extend(entity_words)
        else:
            processed_sentence.append(entity[0])
    return ' '.join(processed_sentence)

def process_sentence(sentence):
    processed_sentence = ner_filter(sentence)
    
    gender_matches = string_match_filter(processed_sentence, female_attributes_filters)
    
    gender_ner_matches = ner_filter(processed_sentence)

    if '[NAME]' or '[FEMALE ATTRIBUTES]' or '[MALE ATTRIBUTES]' or '[FEMALE NAME]' or '[MALE NAME]' or '[FEMALE JOB]' or '[MALE JOB]' in processed_sentence:
        #print("Processed sentence:", processed_sentence)
        print("String matching for gender:", gender_matches)
        #print("NER for gender:", gender_ner_matches)
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
    "He came out as gay during the meeting.",
    "Anna is gentle and caring towards others.",
    "His hardworking nature led to his success in the project.",
    "She is honest and forthright in her opinions.",
    "He has excellent interpersonal skills, making him a great team player.",
    "The success of the project is due to their interdependence.",
    "vidya is a kind and compassionate bro."
]

for sentence in example_sentences:
    print("Processing sentence:", sentence)
    process_sentence(sentence)
