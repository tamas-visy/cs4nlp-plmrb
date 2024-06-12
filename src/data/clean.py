import logging
import re

from src.data.iohandler import IOHandler
from src.data.datatypes import TextDataset

logger = logging.getLogger(__name__)


def clean_dataset(dataset: TextDataset, dummy=False) -> TextDataset:
    """Cleans the dataset using cleaning measures such as filtering via string matching or Named Entity Recognition."""
    # Can use NLTK or spaCy
    if not dummy:
        count = 0
        dataset_clean = dataset.map(lambda row: {"input": process_sentence(row["input"]), "label": row["label"]})
        for i,j in zip(dataset, dataset_clean):
            if i!=j:
                count += 1
        logger.info(f"Masked {count} rows")
    else:
        # Dummy version drops rows with "London" in them
        dataset_clean = dataset.filter(lambda row: "London".lower() not in row['input'])
        logger.warning("This is a dummy implementation")
    return dataset_clean


# ######################################################################################################################
# ############################################ IMPLEMENTATIONS #########################################################
# ######################################################################################################################

# TODO DEBUG:
#   superlatives only used for women/men,
#   expand on unnecessarily citing personal information or gender references when not required,
#   gendered attributes to neutral like 'ok'?, mask personal names with gender neutral names?

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


# Word list taken from:
# https://github.com/gender-bias/gender-bias/tree/master/genderbias
#   job-posting-specific male and female attributes
# https://github.com/uclanlp/gn_glove/
#   gendered titles
# https://github.com/davidemiceli/gender-detection/
#   personal names
# https://github.com/microsoft/responsible-ai-toolbox-genbit/tree/main
#   gendered professions and adjectives related to cis, non-binary and trans people
# https://github.com/tolga-b/debiaswe/tree/master
#   general male and female coded words (implicit gender bias descriptors) and professions
# https://github.com/amity/gender-neutralize/
#   gender-neutral job counterparts

female_names = read_names(IOHandler.raw_path_to("dropping/female_names.js"))
male_names = read_names(IOHandler.raw_path_to("dropping/male_names.js"))

male_titles = read_wordlist(IOHandler.raw_path_to("dropping/male_titles.txt"))
female_titles = read_wordlist(IOHandler.raw_path_to("dropping/female_titles.txt"))

female_jobs_filters = read_wordlist(IOHandler.raw_path_to("dropping/female_jobs.txt"))
male_jobs_filters = read_wordlist(IOHandler.raw_path_to("dropping/male_jobs.txt"))

cis = read_wordlist(IOHandler.raw_path_to("dropping/cis.txt"))
trans = read_wordlist(IOHandler.raw_path_to("dropping/trans.txt"))
non_binary = cis # A workaround to avoid using non-binary wordlist without changing too much code
# non_binary = read_wordlist(IOHandler.raw_path_to("dropping/non-binary.txt"))

# female_attributes_filters = read_wordlist(IOHandler.raw_path_to("dropping/female_adjectives.wordlist"))
# male_attributes_filters = read_wordlist(IOHandler.raw_path_to("dropping/male_adjectives.wordlist"))

extra = read_wordlist(IOHandler.raw_path_to("dropping/extra.txt"))

male_jobs_filters = [word for word in male_jobs_filters if
                     word not in cis and word not in trans and word not in non_binary]
female_jobs_filters = [word for word in female_jobs_filters if
                       word not in cis and word not in trans and word not in non_binary]


def string_match_filter(text, filters, mask="MASK"):
    pattern = re.compile(r'\b(?:' + '|'.join(re.escape(word) for word in filters) + r')\b', re.IGNORECASE)
    return re.sub(pattern, '[' + mask + ']', text)


def process_sentence(sentence: str) -> str:
    # Disabled ner_filter as it's only working on gendered attributes
    processed_sentence = sentence

    processed_sentence = string_match_filter(processed_sentence, set(trans).union(cis).union(non_binary), 'ORIENTATION')

    processed_sentence = string_match_filter(processed_sentence, set(female_titles).union(male_titles), 'TITLE/PRONOUN')

    # Not replacing gendered attributes
    # processed_sentence = string_match_filter(processed_sentence,
    #                                          set(female_attributes_filters).union(male_attributes_filters),
    #                                          'GENDERED ATTRIBUTES')
    processed_sentence = string_match_filter(processed_sentence, set(female_jobs_filters).union(male_jobs_filters), 'JOB')

    processed_sentence = string_match_filter(processed_sentence, set(female_names).union(male_names), 'NAME')

    processed_sentence = string_match_filter(processed_sentence, extra, 'ORIENTATION') # Mask name will change. For now, this is fitting

    return processed_sentence


def test():
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
        print("                    ", process_sentence(sentence))


if __name__ == '__main__':
    test()
