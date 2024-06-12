# Word list taken from:
# https://github.com/gender-bias/gender-bias/tree/master/genderbias: job-posting-specific male and female attributes
# https://github.com/uclanlp/gn_glove/: gendered titles
# https://github.com/davidemiceli/gender-detection/: personal names
# https://github.com/microsoft/responsible-ai-toolbox-genbit/tree/main: gendered professions and adjectives related to cis, non-binary and trans people
# https://github.com/tolga-b/debiaswe/tree/master: general male and female coded words (implicit gender bias descriptors) and professions
# https://github.com/amity/gender-neutralize/: gender neutral job counterparts
# TO DEBUG: superlatives only used for women/men, expand on unnecessarly citing personal infomration or gender references when not required, gendered attirbutews to neutral like 'ok'?, mask personal names with gender neutralnames?

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

female_names = read_names(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\female.js")
male_names = read_names(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\male.js")
male_titles = read_wordlist(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\male_word_file.txt")
female_titles = read_wordlist(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\female_word_file.txt")
female_jobs_filters = read_wordlist(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\female.txt")
male_jobs_filters = read_wordlist(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\male.txt")
cis = read_wordlist(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\cis.txt")
trans = read_wordlist(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\trans.txt")
non_binary = read_wordlist(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\non-binary.txt")
female_attributes_filters = read_wordlist(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\female_adjectives.wordlist")
male_attributes_filters = read_wordlist(r"C:\Users\susan\.csnlp\cs4nlp-plmrb\data\raw\male_adjectives.wordlist")
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
    gender_matches_titles = string_match_filter(processed_sentence, set(female_titles).union(male_titles), 'GENDERED TITLES')

    processed_sentence = string_match_filter(processed_sentence, set(trans).union(non_binary), 'GENDERED ORIENTATION')
    processed_sentence = string_match_filter(processed_sentence, set(female_titles).union(male_titles), 'GENDERED TITLES')
    processed_sentence = string_match_filter(processed_sentence, set(female_attributes_filters).union(male_attributes_filters), 'GENDERED ATTRIBUTES')
    processed_sentence = string_match_filter(processed_sentence, set(female_jobs_filters).union(male_jobs_filters), 'GENDERED JOBS')
    processed_sentence = string_match_filter(processed_sentence, set(female_names).union(male_names), 'GENDERED NAMES')
    


    if '[NAME]' or '[FEMALE ATTRIBUTES]' or '[MALE ATTRIBUTES]' or '[FEMALE NAME]' or '[MALE NAME]' or '[FEMALE JOB]' or '[MALE JOB]' in processed_sentence:
        print("Processing sentence:", sentence)
        print("Processed sentence:", processed_sentence)
        #print("String matching for gender attributes:", gender_matches_attributes)
        #print("String matching for gender jobs:", gender_matches_jobs)
        #print("String matching for gender names:", gender_matches_names)
        #print("String matching for gender orientation or pronouns:", gender_matches_orientation)
        #print("String matching for gender titles:", gender_matches_titles)



def common_words(*lists):
    word_counts = defaultdict(int)
    for idx, lst in enumerate(lists):
        for word in lst:
            word_counts[word] += 1
    
    common_words_lists = {word: [idx for idx, lst in enumerate(lists) if word in lst] for word, count in word_counts.items() if count > 1}
    return common_words_lists


common_words_lists = common_words(female_names, male_names, female_jobs_filters, male_jobs_filters, cis, trans, non_binary, female_attributes_filters, male_attributes_filters)
