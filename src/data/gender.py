import spacy
from collections import defaultdict

nlp = spacy.load("en_core_web_sm")

male_words = ["he", "him", "his", "himself", "Mr."]
female_words = ["she", "her", "hers", "herself", "Ms.", "Mrs."]


def detect_gender_bias(text):
    doc = nlp(text)
    gender_bias_phrases = []
    entities = defaultdict(list)

    for ent in doc.ents:
        if ent.label_ == "PERSON":
            entities[ent.text].append(ent)

    sentences = list(doc.sents)
    for sent in sentences:
        sent_text = sent.text
        male_count = sum(sent_text.lower().count(word) for word in male_words)
        female_count = sum(sent_text.lower().count(word) for word in female_words)

        if male_count > 0 or female_count > 0:
            for word in male_words + female_words:
                if word in sent_text:
                    gender_bias_phrases.append(sent_text)
                    break

    return gender_bias_phrases, entities


def test():
    text = """
        Mr. John Doe is the CEO of the company. He has been leading the company towards success.
        Mrs. Jane Smith, on the other hand, takes care of the office administration and her efforts are often overlooked.
        """

    gender_bias_phrases, entities = detect_gender_bias(text)

    print("Detected gender bias phrases:")
    for phrase in gender_bias_phrases:
        print(phrase)

    print("\nNamed Entities:")
    for entity, occurrences in entities.items():
        print(f"{entity}: {[ent.label_ for ent in occurrences]}")


if __name__ == "__main__":
    test()
