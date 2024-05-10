import logging
from enum import Enum

import numpy as np
from tqdm import tqdm
import transformers
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, LlamaTokenizer, LlamaModel

from src.data.datatypes import TextData, EncodingData
from src.data.download import Downloader
from src.data.iohandler import IOHandler
from src.models.tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class LanguageModel:
    def encode(self, texts: TextData) -> EncodingData:
        """Returns the encodings of texts"""
        logger.debug(f"{self.__class__.__name__} is encoding {len(texts)} sentences")
        return self._encode(texts)

    def _encode(self, texts: TextData) -> EncodingData:
        raise NotImplementedError


class DummyLanguageModel(LanguageModel):
    def __init__(self):
        self.embedding_size = 16

    def _encode(self, texts: TextData) -> EncodingData:
        return [np.random.random(self.embedding_size) for _ in texts]


class GloveLanguageModel(LanguageModel):
    """See https://nlp.stanford.edu/projects/glove/."""

    class GloveVersion(Enum):
        Wikipedia6B = "https://nlp.stanford.edu/data/glove.6B.zip"
        CommonCrawl42B = "https://nlp.stanford.edu/data/glove.42B.300d.zip"
        CommonCrawl840B = "https://nlp.stanford.edu/data/glove.840B.300d.zip"
        Twitter27B = "https://nlp.stanford.edu/data/glove.twitter.27B.zip"

    def __init__(self, version=GloveVersion.Wikipedia6B, embedding_dim=100, aggregation=np.mean):
        Downloader.download_glove(version)
        self.embedding_dim = embedding_dim
        self.embeddings = IOHandler.load_glove_embeddings(version, self.embedding_dim)
        self.aggregation = aggregation

    def _encode(self, texts: TextData) -> EncodingData:
        embeddings = []
        for text in tqdm(texts):
            text_embedding = []
            for token in Tokenizer.tokenize(text):
                # For out-of-dictionary words we use the zero vector
                embedding = self.embeddings.get(token.lower(), np.zeros(self.embedding_dim))
                text_embedding.append(embedding)
            embeddings.append(self.aggregation(text_embedding, axis=0))
        return embeddings

#---------------------Transformers---------------------

class BertLanguageModel(LanguageModel):
    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)

    def _encode(self, texts: TextData) -> EncodingData:
        encodings = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            encodings.append(output.last_hidden_state.mean(dim=1).detach().numpy())
        return encodings

class GPT2LanguageModel(LanguageModel):
    def __init__(self, model_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2Model.from_pretrained(model_name)

    def _encode(self, texts: TextData) -> EncodingData:
        encodings = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            encodings.append(output.last_hidden_state.mean(dim=1).detach().numpy())
        return encodings

class LlamaLanguageModel(LanguageModel):
    def __init__(self, model_name='Llama-2-7b-chat-hf'): #no access to repository yet
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaModel.from_pretrained(model_name)

    def _encode(self, texts: TextData) -> EncodingData:
        encodings = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt')
            output = self.model(**encoded_input)
            encodings.append(output.last_hidden_state.mean(dim=1).detach().numpy())
        return encodings

