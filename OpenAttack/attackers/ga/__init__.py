import numpy as np
from copy import deepcopy
from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...data_manager import DataManager
from ...tags import TAG_English, Tag
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from typing import List, Optional
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
from ...exceptions import WordNotInDictionaryException
import torch

DEFAULT_CONFIG = {
    "sst": False,
}


class GAAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return {self.__lang_tag, Tag("get_pred", "victim")}  # hard_label

    def __init__(self,
                 tokenizer: Optional[Tokenizer] = None,
                 substitute: Optional[WordSubstitute] = None,
                 token_unk="<unk>",
                 lang=None):
        """
        Generating Natural Language Attacks in a Hard Label Black Box Setting
        `[pdf] https://ojs.aaai.org/index.php/AAAI/article/view/17595/17402`
        `[code]https://github.com/RishabhMaheshwary/hard-label-attack`

        Args:

        :Language: English
        :Classifier Capacity:
            * get_pred
        """

        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if substitute is not None:
            lst.append(substitute)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language %s" % lang)

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute

        check_language([self.tokenizer, self.substitute], self.__lang_tag)

        self.token_unk = token_unk

        synonym_words = [
            self.get_neighbours(word,pos) # TODO
            # for idx,word,pos in x_orig
        ]

    def attack(self, victim: Classifier, sentence: str, goal: ClassifierGoal):
        """

        """
        x_orig = sentence.lower()
        orig_probs = victim.get_prob([x_orig])[0]
        orig_label = orig_probs.argmax()
        orig_prob = orig_probs.max()

        x_orig = self.tokenizer.tokenize(x_orig)
        x_pos = list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        len_text = len(x_orig)

    def get_neighbours(self, word, pos):
        try:
            return list(
                filter(
                    lambda x: x != word,
                    map(
                        lambda x: x[0],
                        self.substitute(word,pos),
                    )
                )
            )
        except WordNotInDictionaryException:
            return []

    def pos_filter(self, ori_pos, new_pos_list):
        same = [True if ori_pos == new_pos or (set([ori_pos, new_pos]) <= set(['noun', 'verb']))  # 集合比较大小其实就是在比较集合的包含关系。？这个地方不明白为什么是名词或动词都可以
                else False for new_pos in new_pos_list]
        return same
