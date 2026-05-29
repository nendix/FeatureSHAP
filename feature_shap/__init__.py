from .feature_shap import FeatureSHAP

from .comparators.comparator_base import ComparatorBase
from .comparators.embeddings.huggingface_comparator import HuggingFaceComparator
from .comparators.embeddings.openai_comparator import OpenAIComparator
from .comparators.embeddings.tfidf_text_comparator import TfidfTextComparator
from .comparators.embeddings.bertscore_comparator import BertScoreComparator

from .comparators.metrics.bleu_comparator import BLEUComparator
from .comparators.metrics.codebleu_comparator import CodeBLEUComparator

from .models.model_base import ModelBase
from .models.huggingface_model import HuggingFaceModel
try:
    from .models.vllm_model import VLLMModel
except ImportError:
    VLLMModel = None
from .models.openai_model import OpenAIModel

from .splitters.splitter_base import SplitterBase
from .splitters.points_splitter import PointsSplitter
from .splitters.llm_splitter import LLMSplitter
from .splitters.tokenizer_splitter import TokenizerSplitter
from .splitters.block_splitter import BlocksSplitter
from .splitters.word_splitter import WordSplitter

from .modifiers.modifier_base import ModifierBase
from .modifiers.masker_modifier import MaskerModifier
from .modifiers.remover_modifier import RemoverModifier
from .modifiers.removal_modifier import RemovalModifier


__all__ = [
    # Module
    "FeatureSHAP",

    # Splitters
    "SplitterBase",
    "PointsSplitter",
    "LLMSplitter",
    "TokenizerSplitter",
    "BlocksSplitter",
    "WordSplitter",

    # Modifiers
    "ModifierBase",
    "MaskerModifier",
    "RemoverModifier",
    "RemovalModifier",

    # Models
    "ModelBase",
    "HuggingFaceModel",
    "VLLMModel",
    "OpenAIModel",

    # Comparators
    "ComparatorBase",
    ## Embedding-based
    "BertScoreComparator",
    "HuggingFaceComparator",
    "OpenAIComparator",
    "TfidfTextComparator",

    ## Metric-based
    "BLEUComparator",
    "CodeBLEUComparator",
]
