import copy
from typing import List, Optional, Union
import numpy as np
from transformers import BertConfig, BertTokenizerFast, BertForMaskedLM
import torch


from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...tags import TAG_English, Tag
from ...exceptions import WordNotInDictionaryException
from ...attack_assist.substitute.word import get_default_substitute, WordSubstitute
from ...attack_assist.filter_words import get_default_filter_words

class Networkbug(ClassificationAttacker):
    @property
    def TAGS(self):
        return { self.__lang_tag, Tag("get_pred", "victim"), Tag("get_prob", "victim")}

    def __init__(self):
        