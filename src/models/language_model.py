import logging
import os
from enum import Enum
from typing import Literal, List

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
from src.utils.misc import stable_hash

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

    def encode(self, texts: TextData, result_type: int | Literal["initial", "final", "middle"] = "final",
               agg_func=None, caching=True) -> EncodingData:
        """Returns the encodings of texts. The result_type parameter is ignored for GloVe embeddings."""
        return self._encode(texts)

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
    def __init__(self, model_name, model_class, tokenizer_class,
                 half_precision=False, device=None, batch_size: int | None = 16):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        kwargs = {}
        if half_precision:
            kwargs["torch_dtype"] = torch.float16
            logger.debug("FP16 enabled")
        self.device = device
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        self.model = model_class.from_pretrained(model_name, **kwargs).to(self.device)
        self.model.eval()
        self.batch_size = batch_size  # about 2-3x speedup compared to no batching or large batches

        self._get_initial_can_be_batched = True
        """A flag to disable batch inference due to 'attention_mask' not being supported in get_initial"""

        self._cache = IOHandler.load_cache_of(self)

    @property
    def num_encoder_layers(self):
        return len(self.model.encoder.layer)

    def encode(self, texts: TextData, result_type: int | Literal["initial", "final", "middle"] = "final",
               agg_func=torch.mean,
               caching=True) -> EncodingData:
        """Returns the encodings of texts using the aggregation function over a sentence.
        Depending on result type - "initial", "final", "middle", or an integer the vectors after
        the appropriate layer are returned"""
        logger.debug(f"{self.__class__.__name__} is encoding {len(texts)} sentences - {result_type, agg_func, caching}")
        if not caching:
            raise NotImplementedError

        cache_key = (agg_func.__name__, str(result_type), str(stable_hash('='.join(texts)))[:9])
        if cache_key in self._cache:
            logger.debug(f"Found {cache_key} in cache, using that instead of encoding again")
            return self._cache[cache_key]

        encodings: List[np.ndarray] = self._encode(texts, result_type=result_type, agg_func=agg_func)
        IOHandler.save_cache_of(self, encodings, *cache_key)
        return encodings

    def _encode(self, texts, result_type=None, agg_func=None):
        if result_type is None or agg_func is None:
            raise NotImplementedError("You bypassed encode() and did not provide required arguments")
        if result_type == "middle":
            result_type = self.num_encoder_layers // 2

        outputs = []
        with tqdm(total=len(texts)) as bar:
            batch_size = self.batch_size if self.batch_size is not None else 1
            if result_type == "initial" and not self._get_initial_can_be_batched:
                logger.warning(f"Force disabling batching for result type 'initial' of {self.__class__.__name__}")
                batch_size = 1

            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                x = self._create_model_inputs(batch_texts)
                with torch.no_grad():
                    if result_type == "initial":
                        output = self._get_initial(**x)
                    elif result_type == "final":
                        output = self._get_final(**x)
                    else:
                        output = self._get_at(at=result_type, **x)
                # I removed .tolist() from after .numpy(), which might cause issues
                outputs.extend(agg_func(output, dim=1).cpu().numpy())  # should work with batches
                bar.update(len(output))
        return outputs

    def _get_initial(self, **kwargs):
        if not self._get_initial_can_be_batched:
            # If not supported, we have to manually remove the attention mask
            if 'attention_mask' in kwargs:
                # verify that all values are one in the mask before removing
                assert kwargs['attention_mask'].all()
                kwargs.pop("attention_mask")
        try:
            return self.model.embeddings(**kwargs)
        except TypeError as te:
            if "got an unexpected keyword argument 'attention_mask'" in str(te):
                raise NotImplementedError(
                    f"Batch inference with {self.__class__.__name__} is not supported for _get_initial, "
                    + "set _get_initial_can_be_batched to False")

    def _get_final(self, **kwargs):
        return self.model(**kwargs).last_hidden_state

    def _get_at(self, at: int, **kwargs):
        return self.model(**kwargs, output_hidden_states=True).hidden_states[at]

    def _create_model_inputs(self, text: str | List[str]) -> dict:
        encoded_input = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=512
        ).to(self.device)

        # Attention mask is necessary when batch encoding
        # If any model actually uses the attention mask, we can push this down one class
        # if 'attention_mask' in encoded_input.data:
        #     assert encoded_input.data['attention_mask'].all()
        #     encoded_input.data.pop("attention_mask")
        return encoded_input


class BERTLanguageModel(TransformerModel):
    def __init__(self, model_name='bert-base-uncased', device=None):
        super().__init__(model_name, model_class=BertModel, tokenizer_class=BertTokenizer, device=device)
        self._get_initial_can_be_batched = False


class GPT2LanguageModel(TransformerModel):
    def __init__(self, model_name='gpt2', device=None):
        super().__init__(model_name, model_class=GPT2Model, tokenizer_class=GPT2Tokenizer, device=device)
        # Add a padding token if not already present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self._get_initial_can_be_batched = False

    @property
    def num_encoder_layers(self):
        return len(self.model.h)

    def _get_initial(self, input_ids, attention_mask=None, **kwargs):
        # Only arg that is accepted
        if attention_mask is not None:
            # verify that all values are one in the mask even if we don't use it
            assert attention_mask.all()
        return self.model.wte(input=input_ids)

    # def _create_model_inputs(self, text: str) -> dict:
    #     encoded_input = super()._create_model_inputs(text)
    #     encoded_input.data['input'] = encoded_input.data.pop("input_ids")
    #     return encoded_input


class LLaMALanguageModel(TransformerModel):
    """LLaMA implementation modified to work with decoder-only architecture.
    Unlike encoder-decoder models, LLaMA uses a single stack of transformer layers,
    so we need to handle layer access differently."""
    
    def __init__(self, model_name='meta-llama/Llama-2-7b-hf', half_precision=False, device=None):
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
        login(token=token, write_permission=False)
        logger.debug("Logged in")
        super().__init__(model_name, LlamaForCausalLM, LlamaTokenizer,
                         half_precision=half_precision, device=device, batch_size=8)
        
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    @property
    def num_encoder_layers(self):
        """Get number of transformer layers in LLaMA.
        Fixed to use model.model.layers instead of looking for an encoder,
        as LLaMA is decoder-only and doesn't have an encoder component.
        This correctly counts the number of transformer layers in the model."""
        return len(self.model.model.layers)

    def _get_initial(self, input_ids, attention_mask=None, **kwargs):
        """Get initial embeddings directly from LLaMA's embedding layer.
        Now uses embed_tokens to get initial embeddings directly,
        which is simpler and more reliable than using _get_at(0)."""
        return self.model.model.embed_tokens(input_ids)

    def _get_final(self, **kwargs):
        """Get final hidden states from the model.
        Part of proper hidden states handling, configured to get
        states from the final layer of the model."""
        return self.model(**kwargs).hidden_states[-1]

    def _get_at(self, at: int, **kwargs):
        """Get hidden states from a specific layer.
        Part of proper hidden states handling, configured to get
        states from any specified layer in the model."""
        return self.model(**kwargs, output_hidden_states=True).hidden_states[at]


class RoBERTaLanguageModel(TransformerModel):
    def __init__(self, model_name='roberta-base', device=None):
        super().__init__(model_name, model_class=RobertaModel, tokenizer_class=RobertaTokenizer, device=device)
        self._get_initial_can_be_batched = False


class ELECTRALanguageModel(TransformerModel):
    def __init__(self, model_name='google/electra-base-discriminator', device=None):
        super().__init__(model_name, ElectraModel, ElectraTokenizer, device=device)
        self._get_initial_can_be_batched = False


class T5LanguageModel(TransformerModel):
    # Could have access to decoder hidden states with output.decoder_hidden_states
    def __init__(self, model_name='t5-base', device=None):
        super().__init__(model_name, model_class=T5Model, tokenizer_class=T5Tokenizer, device=device)

    @property
    def num_encoder_layers(self):
        return len(self.model.encoder.block)  # T5 uses 'block' instead of 'layer'

    def _get_initial(self, input_ids, decoder_input_ids=None, **kwargs):
        # Get embeddings directly from the shared embedding layer
        return self.model.shared(input_ids)

    def _get_final(self, **kwargs):
        # Only use encoder outputs
        return self.model.encoder(**kwargs).last_hidden_state

    def _get_at(self, at: int, **kwargs):
        # Only use encoder hidden states
        return self.model.encoder(**kwargs, output_hidden_states=True).hidden_states[at]

    def _create_model_inputs(self, text: str) -> dict:
        encoded_input = super()._create_model_inputs(text)
        # We only need input_ids for encoder-only operation
        return dict(input_ids=encoded_input.input_ids)


class XLNetLanguageModel(TransformerModel):
    def __init__(self, model_name='xlnet-base-cased', device=None):
        super().__init__(model_name, model_class=XLNetModel, tokenizer_class=XLNetTokenizer, device=device)
        self._get_initial_can_be_batched = False

    @property
    def num_encoder_layers(self):
        return len(self.model.layer)

    def _get_initial(self, input_ids, attention_mask=None, **kwargs):
        # Remove attention mask for initial embeddings
        return self.model.word_embedding(input=input_ids)
