import logging
from enum import Enum

import numpy as np
from tqdm import tqdm
import torch
from huggingface_hub import login
from transformers import BertTokenizer, BertModel, GPT2Tokenizer, GPT2Model, LlamaTokenizer, LlamaForCausalLM, \
    RobertaTokenizer, RobertaModel, ElectraTokenizer, ElectraModel, T5Tokenizer, T5Model, XLNetTokenizer, XLNetModel

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


# ---------------------Transformers---------------------

class BERTLanguageModel:
    def __init__(self, model_name='bert-base-uncased', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _encode(self, texts):
        encodings = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
                self.device)
            with torch.no_grad():
                output = self.model(**encoded_input)
            encodings.append(output.last_hidden_state.mean(dim=1).cpu().numpy())
        return encodings


class GPT2LanguageModel:
    def __init__(self, model_name='gpt2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

        # Add a padding token if not already present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})

        self.model = GPT2Model.from_pretrained(model_name).to(self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.eval()

    def _encode(self, texts):
        encodings = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
                self.device)
            with torch.no_grad():
                output = self.model(**encoded_input)
            encodings.append(output.last_hidden_state.mean(dim=1).cpu().numpy())
        return encodings


class LLaMALanguageModel:
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Add a padding token if not already present
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _encode(self, texts):
        login()
        encodings = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True).to(self.device)
            with torch.no_grad():
                output = self.model(**encoded_input)
            encodings.append(output.last_hidden_state.mean(dim=1).cpu().numpy())
        return encodings


class RoBERTaLanguageModel:
    def __init__(self, model_name='roberta-base', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = RobertaTokenizer.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _encode(self, texts):
        encodings = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
                self.device)
            with torch.no_grad():
                output = self.model(**encoded_input)
            encodings.append(output.last_hidden_state.mean(dim=1).cpu().numpy())
        return encodings


class ELECTRALanguageModel:
    def __init__(self, model_name='google/electra-base-discriminator',
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = ElectraTokenizer.from_pretrained(model_name)
        self.model = ElectraModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _encode(self, texts):
        encodings = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
                self.device)
            with torch.no_grad():
                output = self.model(**encoded_input)
            encodings.append(output.last_hidden_state.mean(dim=1).cpu().numpy())
        return encodings


class T5LanguageModel:
    def __init__(self, model_name='t5-base', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5Model.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _encode(self, texts):
        encodings = []
        for text in texts:
            # Encode input text
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
                self.device)
            # Create decoder input ids
            decoder_input_ids = self.tokenizer('<pad>', return_tensors='pt').input_ids.to(self.device)
            # Get model output
            with torch.no_grad():
                output = self.model(input_ids=encoded_input.input_ids, decoder_input_ids=decoder_input_ids)
            # Take the mean of the encoder's last hidden state
            encodings.append(output.encoder_last_hidden_state.mean(dim=1).cpu().numpy())
        return encodings


class XLNetLanguageModel:
    def __init__(self, model_name='xlnet-base-cased', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.tokenizer = XLNetTokenizer.from_pretrained(model_name)
        self.model = XLNetModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _encode(self, texts):
        encodings = []
        for text in texts:
            encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
                self.device)
            with torch.no_grad():
                output = self.model(**encoded_input)
            encodings.append(output.last_hidden_state.mean(dim=1).cpu().numpy())
        return encodings
