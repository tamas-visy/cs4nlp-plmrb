import nltk


class Tokenizer:
    @classmethod
    def tokenize(cls, text):
        return nltk.word_tokenize(text)
