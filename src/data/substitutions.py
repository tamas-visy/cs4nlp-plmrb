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
male_titles = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/male_word_file.txt")
female_titles = read_wordlist("/workspaces/cs4nlp-plmrb/data/raw/female_word_file.txt")
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
    return re.sub(pattern, '[' + mask + ']', text)

def get_gender_neutral_job(job_title):
    substitutions = [
        (r'\bwait(er|ress)\b', 'server'),
        (r'\bwait(ers|resses)\b', 'servers'),
        (r'\bactor\b', 'performer'),
        (r'\bactors\b', 'performers'),
        (r'\bactress\b', 'performer'),
        (r'\bactresses\b', 'performers'),
        (r'\bboy\b', 'young person'),
        (r'\bboys\b', 'young people'),
        (r'\bboyfriend\b', 'partner'),
        (r'\bboyfriends\b', 'partners'),
        (r'\bbride\b', 'spouse'),
        (r'\brides\b', 'spouses'),
        (r'\bbrother\b', 'sibling'),
        (r'\bbrothers\b', 'siblings'),
        (r'\bbusiness(wo)?men\b', 'businesspeople'),
        (r'\bbusiness(wo)?man\b', 'businessperson'),
        (r'\bchair(wo)?man\b', 'chair'),
        (r'\bcleaning lady\b', 'cleaner'),
        (r'\bdude\b', 'person'),
        (r'\bdudes\b', 'people'),
        (r'\bfire(wo)?men\b', 'firefighters'),
        (r'\bfire(wo)?man\b', 'firefighter'),
        (r'\bfresh(wo)?men\b', 'first-years'),
        (r'\bfresh(wo)?man\b', 'first-year'),
        (r'\bgarbage(wo)?man\b', 'garbage collector'),
        (r'\bgarbage(wo)?men\b', 'garbage collectors'),
        (r'\bgirl\b', 'young person'),
        (r'\bgirls\b', 'young people'),
        (r'\bgirlfriend\b', 'partner'),
        (r'\bgirlfriends\b', 'partners'),
        (r'\bgrandmother\b', 'grandparent'),
        (r'\bgrandmothers\b', 'grandparents'),
        (r'\bgrandfather\b', 'grandparent'),
        (r'\bgrandfathers\b', 'grandparents'),
        (r'\bgrandson\b', 'grandchild'),
        (r'\bgrandsons\b', 'grandchildren'),
        (r'\bgranddaughter\b', 'grandchild'),
        (r'\bgranddaughters\b', 'grandchildren'),
        (r'\bgrandparent\b', 'elder'),
        (r'\bgrandparents\b', 'elders'),
        (r'\bgrandchild\b', 'young relative'),
        (r'\bgrandchildren\b', 'young relatives'),
        (r'\bgrandkid\b', 'young relative'),
        (r'\bgrandkids\b', 'young relatives'),
        (r'\bgrandbaby\b', 'young relative'),
        (r'\bgrandbabies\b', 'young relatives'),
        (r'\bgrandbaby\b', 'young relative'),
        (r'\bgrandbabies\b', 'young relatives'),
        (r'\bgrandma\b', 'elder'),
        (r'\bgrandmas\b', 'elders'),
        (r'\bgrandpa\b', 'elder'),
        (r'\bgrandpas\b', 'elders'),
        (r'\bgrandma\b', 'elder'),
        (r'\bgrandmas\b', 'elders'),
        (r'\bgrandpa\b', 'elder'),
        (r'\bgrandpas\b', 'elders'),
        (r'\bhe\b', 'they'),
        (r'\bhis\b', 'their'),
        (r'\bher\b', 'their'),
        (r'\bhers\b', 'theirs'),
        (r'\bhim\b', 'them'),
        (r'\bhimself\b', 'themselves'),
        (r'\bhusband\b', 'spouse'),
        (r'\bhusbands\b', 'spouses'),
        (r'\bking\b', 'monarch'),
        (r'\bkings\b', 'monarchs'),
        (r'\bmaid\b', 'domestic worker'),
        (r'\bmaids\b', 'domestic workers'),
        (r'\bman\b', 'person'),
        (r'\bmen\b', 'people'),
        (r'\bmankind\b', 'humankind'),
        (r'\bmarksman\b', 'sharpshooter'),
        (r'\bmarks(wo)?men\b', 'sharpshooters'),
        (r'\bmother\b', 'parent'),
        (r'\bmothers\b', 'parents'),
        (r'\bmom\b', 'parent'),
        (r'\bmoms\b', 'parents'),
        (r'\bnewlywed\b', 'newlywed couple'),
        (r'\bnewlyweds\b', 'newlywed couples'),
        (r'\bniece\b', 'nibling'),
        (r'\bnieces\b', 'nieces and nephews'),
        (r'\bnephew\b', 'nibling'),
        (r'\bnephews\b', 'nieces and nephews'),
        (r'\bpolice(wo)?man\b', 'police officer'),
        (r'\bpolice(wo)?men\b', 'police officers'),
        (r'\bpost(wo)?man\b', 'mail carrier'),
        (r'\bpost(wo)?men\b', 'mail carriers'),
        (r'\bseamstress\b', 'sewer'),
        (r'\bseamstresses\b', 'sewers'),
        (r'\bsir\b', 'person'),
        (r'\bsirs\b', 'people'),
        (r'\bson\b', 'child'),
        (r'\bsons\b', 'children'),
        (r'\bstewardess\b', 'flight attendant'),
        (r'\bstewardesses\b', 'flight attendants'),
        (r'\buncle\b', 'auncle'),
        (r'\buncles\b', 'auncles'),
        (r'\bwidow\b', 'surviving spouse'),
        (r'\bwidows\b', 'surviving spouses'),
        (r'\bwidower\b', 'surviving spouse'),
        (r'\bwidowers\b', 'surviving spouses'),
        (r'\bwife\b', 'spouse'),
        (r'\bwives\b', 'spouses'),
        (r'\bwoman\b', 'person'),
        (r'\bwomen\b', 'people'),
    ]

    # Apply substitutions
    original_title = job_title
    for pattern, replacement in substitutions:
        job_title = re.sub(pattern, replacement, job_title, flags=re.IGNORECASE)
        if job_title != original_title:
            break

    return job_title

def process_sentence(sentence):
    # Step 1: Apply NER filter to replace gendered attributes with placeholders
    tokens = word_tokenize(sentence)
    tagged_tokens = pos_tag(tokens)
    named_entities = ne_chunk(tagged_tokens)
    processed_sentence = []
    for entity in named_entities:
        if isinstance(entity, nltk.Tree):
            entity_words = []
            for word, tag in entity.leaves():
                if any(re.match(r'\b{}\w*\b'.format(re.escape(attr)), word, re.IGNORECASE) for attr in female_attributes_filters):
                    entity_words.append('[FEMALE ATTRIBUTES]')
                elif any(re.match(r'\b{}\w*\b'.format(re.escape(attr)), word, re.IGNORECASE) for attr in male_attributes_filters):
                    entity_words.append('[MALE ATTRIBUTES]')
                else:
                    entity_words.append(word)
            processed_sentence.extend(entity_words)
        else:
            processed_sentence.append(entity[0])
    processed_sentence = ' '.join(processed_sentence)

    # Step 2: Replace gendered jobs with neutral counterparts or [GENDERED JOB] if no substitution found
    def replace_jobs(match):
        job_title = match.group(0)
        neutral_job = get_gender_neutral_job(job_title)
        return neutral_job if neutral_job != job_title else '[GENDERED JOB]'

    gendered_jobs_pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in set(female_jobs_filters).union(male_jobs_filters)) + r')\b', re.IGNORECASE)
    processed_sentence = gendered_jobs_pattern.sub(replace_jobs, processed_sentence)

    # Step 3: Apply other filters
    processed_sentence = string_match_filter(processed_sentence, set(trans).union(cis).union(non_binary), 'GENDERED ORIENTATION')
    processed_sentence = string_match_filter(processed_sentence, set(female_titles).union(male_titles), 'GENDERED TITLES')
    processed_sentence = string_match_filter(processed_sentence, set(female_attributes_filters).union(male_attributes_filters), 'GENDERED ATTRIBUTES')
    processed_sentence = string_match_filter(processed_sentence, set(female_names).union(male_names), 'GENDERED NAMES')

    print("Processed sentence:", processed_sentence)

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
    "annabel is a kind and compassionate waiter.",
    "drag queens pay more taxes to marco"
]

for sentence in example_sentences:
    print("Processing sentence:", sentence)
    process_sentence(sentence)
