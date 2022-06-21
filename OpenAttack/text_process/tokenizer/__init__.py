from .base import Tokenizer
from .jieba_tokenizer import JiebaTokenizer
from .punct_tokenizer import PunctTokenizer
from .transformers_tokenizer import TransformersTokenizer
from transformers import BertTokenizer

def get_default_tokenizer(lang):
    from ...tags import TAG_English, TAG_Chinese
    if lang == TAG_English:
        return PunctTokenizer()
        # return BertTokenizer()
    if lang == TAG_Chinese:
        return JiebaTokenizer()
    return PunctTokenizer()

def get_bert_tokenizer(lang):
    from ...tags import TAG_English, TAG_Chinese
    if lang == TAG_English:
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        return TransformersTokenizer(tokenizer,TAG_English)
    else:
        return get_default_tokenizer(lang)