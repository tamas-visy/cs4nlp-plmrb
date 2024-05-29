# Word list taken from:
# https://github.com/gender-bias/gender-bias/tree/master/genderbias: job-posting-specific male and female attributes
# https://github.com/uclanlp/gn_glove/: gendered titles
# https://github.com/davidemiceli/gender-detection/: personal names
# https://github.com/microsoft/responsible-ai-toolbox-genbit/tree/main: gendered professions and adjectives related to cis, non-binary and trans people
# https://github.com/tolga-b/debiaswe/tree/master: general male and female coded words (implicit gender bias descriptors) and professions
# https://github.com/amity/gender-neutralize/: gender neutral job counterparts
# TO DEBUG: superlatives only used for women/men, expand on unnecessarly citing personal infomration or gender references when not required, mask personal names with gender neutralnames?

import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from collections import defaultdict

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

def read_wordlist(file_path):
    with open(file_path, 'r') as file:
        return {word.strip() for word in file}

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
female_jobs_filters = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/female.txt")
male_jobs_filters = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/male.txt")
cis = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/cis.txt")
trans = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/trans.txt")
non_binary = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/non-binary.txt")
female_attributes_filters = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/female_adjectives.wordlist")
male_attributes_filters = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/male_adjectives.wordlist")
male_jobs_filters = [word for word in male_jobs_filters if word not in cis and word not in trans and word not in non_binary]
female_jobs_filters = [word for word in female_jobs_filters if word not in cis and word not in trans and word not in non_binary]

def string_match_filter(text, filters, mask):
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in filters) + r')\b', re.IGNORECASE)
    return re.sub(pattern, '['+mask+']', text)

def ner_filter(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)
    processed_sentence = []
    for entity in named_entities:
        if isinstance(entity, nltk.tree.Tree):
                entity_words = []
                for word, tag in entity.leaves():
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
    
    gender_matches_attributes = string_match_filter(processed_sentence, set(female_attributes_filters).union(male_attributes_filters), 'GENDERED ATTRIBUTES')
    gender_matches_jobs = string_match_filter(processed_sentence, set(female_jobs_filters).union(male_jobs_filters), 'GENDERED JOBS')
    gender_matches_names = string_match_filter(processed_sentence, set(female_names).union(male_names), 'GENDERED NAMES')
    gender_matches_orientation = string_match_filter(processed_sentence, set(trans).union(cis).union(non_binary), 'GENDERED ORIENTATION')

    
    gender_ner_matches = ner_filter(processed_sentence)

    if '[NAME]' or '[FEMALE ATTRIBUTES]' or '[MALE ATTRIBUTES]' or '[FEMALE NAME]' or '[MALE NAME]' or '[FEMALE JOB]' or '[MALE JOB]' in processed_sentence:
        #print("Processed sentence:", processed_sentence)
        print("String matching for gender attributes:", gender_matches_attributes)
        print("String matching for gender jobs:", gender_matches_jobs)
        print("String matching for gender names:", gender_matches_names)
        print("String matching for gender orientation:", gender_matches_orientation)
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
    "vidya is a kind and compassionate waiter.",
    "drag queens pay more taxes to marco"
]

for sentence in example_sentences:
    print("Processing sentence:", sentence)
    process_sentence(sentence)

def common_words(*lists):
    word_counts = defaultdict(int)
    for idx, lst in enumerate(lists):
        for word in lst:
            word_counts[word] += 1
    
    common_words_lists = {word: [idx for idx, lst in enumerate(lists) if word in lst] for word, count in word_counts.items() if count > 1}
    return common_words_lists


common_words_lists = common_words(female_names, male_names, female_jobs_filters, male_jobs_filters, cis, trans, non_binary, female_attributes_filters, male_attributes_filters)
