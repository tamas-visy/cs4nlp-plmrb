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

class TransformerModel(LanguageModel):
    def __init__(self, model_name, model_class, tokenizer_class, device=None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.model = model_class.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _encode(self, texts):
        encodings = []
        # TODO do batching?
        for text in texts:
            inputs = self._create_model_inputs(text=text)
            with torch.no_grad():
                output = self.model(**inputs)
            encodings.append(output.last_hidden_state.mean(dim=1).cpu().numpy()[0])
        return encodings

    def _create_model_inputs(self, text: str) -> dict:
        encoded_input = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=512
        ).to(self.device)
        return encoded_input


class BERTLanguageModel(TransformerModel):
    def __init__(self, model_name='bert-base-uncased', device=None):
        super().__init__(model_name, model_class=BertModel, tokenizer_class=BertTokenizer, device=device)


class GPT2LanguageModel(TransformerModel):
    def __init__(self, model_name='gpt2', device=None):
        super().__init__(model_name, model_class=GPT2Model, tokenizer_class=GPT2Tokenizer, device=device)
        # Add a padding token if not already present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.model.resize_token_embeddings(len(self.tokenizer))


class LLaMALanguageModel(TransformerModel):
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf', device=None):
        login()  # login might be needed for tokenizer creation
        super().__init__(model_name, LlamaForCausalLM, LlamaTokenizer, device)
        # Add a padding token if not already present

        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})  # why not self.tokenizer.eos_token?
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _encode(self, texts):
        super()._encode(texts)


class RoBERTaLanguageModel(TransformerModel):
    def __init__(self, model_name='roberta-base', device=None):
        super().__init__(model_name, model_class=RobertaModel, tokenizer_class=RobertaTokenizer, device=device)


class ELECTRALanguageModel(TransformerModel):
    def __init__(self, model_name='google/electra-base-discriminator', device=None):
        super().__init__(model_name, ElectraModel, ElectraTokenizer, device)


class T5LanguageModel(TransformerModel):
    def __init__(self, model_name='t5-base', device=None):
        super().__init__(model_name, model_class=T5Model, tokenizer_class=T5Tokenizer, device=device)

    def _create_model_inputs(self, text: str) -> dict:
        encoded_input = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(
            self.device)
        # Create decoder input ids
        decoder_input_ids = self.tokenizer('<pad>', return_tensors='pt').input_ids.to(self.device)
        return dict(input_ids=encoded_input.input_ids, decoder_input_ids=decoder_input_ids)


class XLNetLanguageModel(TransformerModel):
    def __init__(self, model_name='xlnet-base-cased', device=None):
        super().__init__(model_name, model_class=XLNetModel, tokenizer_class=XLNetTokenizer, device=device)
