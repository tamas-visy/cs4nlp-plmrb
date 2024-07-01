import json
import re
import nltk
import spacy
from collections import defaultdict
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk

nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")

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
    return re.sub(pattern, f'[{mask}]', text)

def ner_filter(text):
    tokens = word_tokenize(text)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)
    processed_sentence = []
    for entity in named_entities:
        if isinstance(entity, nltk.tree.Tree):
            entity_words = []
            for word, tag in entity.leaves():
                if word.lower() in female_names or word.lower() in male_names:
                    entity_words.append('[GENDERED NAME]')
                else:
                    entity_words.append(word)
            processed_sentence.extend(entity_words)
        else:
            processed_sentence.append(entity[0])
    return ' '.join(processed_sentence)

def dependency_parse_filter(text, filters, mask):
    doc = nlp(text)
    for token in doc:
        if token.text.lower() in filters:
            unnecessary = True
            for child in token.children:
                if child.dep_ in ('nsubj', 'dobj', 'pobj'):
                    unnecessary = False
                    break
            if unnecessary:
                text = text.replace(token.text, f'[{mask}]')
    return text

def process_sentence(sentence):
    processed_sentence = ner_filter(sentence)
    
    processed_sentence = dependency_parse_filter(processed_sentence, female_attributes_filters.union(male_attributes_filters), 'GENDERED ATTRIBUTES')
    processed_sentence = string_match_filter(processed_sentence, female_jobs_filters.union(male_jobs_filters), 'GENDERED JOBS')
    processed_sentence = string_match_filter(processed_sentence, female_names.union(male_names), 'GENDERED NAMES')
    processed_sentence = string_match_filter(processed_sentence, trans.union(cis).union(non_binary), 'GENDERED ORIENTATION')
    processed_sentence = string_match_filter(processed_sentence, female_titles.union(male_titles), 'GENDERED TITLES')

    if '[GENDERED NAME]' in processed_sentence or '[GENDERED ATTRIBUTES]' in processed_sentence or '[GENDERED JOBS]' in processed_sentence:
        print("Processing sentence:", sentence)
        print("Processed sentence:", processed_sentence)

def common_words(*lists):
    word_counts = defaultdict(int)
    for idx, lst in enumerate(lists):
        for word in lst:
            word_counts[word] += 1
    
    common_words_lists = {word: [idx for idx, lst in enumerate(lists) if word in lst] for word, count in word_counts.items() if count > 1}
    return common_words_lists

common_words_lists = common_words(female_names, male_names, female_jobs_filters, male_jobs_filters, cis, trans, non_binary, female_attributes_filters, male_attributes_filters)

# Example usage
test_sentence = "Mary is an excellent actress. John is a brilliant doctor. He is a transgender person."
process_sentence(test_sentence)
