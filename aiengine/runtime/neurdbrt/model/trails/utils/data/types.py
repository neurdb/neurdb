from typing import Union, Optional, Dict
from torch_frame.config import (
    ImageEmbedderConfig,
    TextEmbedderConfig,
    TextTokenizerConfig,
)

TextEmbedderCFG = Optional[TextEmbedderConfig]
TextTokenizerCFG = Union[dict[str, TextTokenizerConfig],
                         TextTokenizerConfig, None]
ImageEmbedderCFG = Union[dict[str, ImageEmbedderConfig],
                         ImageEmbedderConfig, None]