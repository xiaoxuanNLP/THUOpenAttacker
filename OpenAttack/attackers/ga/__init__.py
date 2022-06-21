import numpy as np
from copy import deepcopy
from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...data_manager import DataManager
from ...tags import TAG_English, Tag
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer, get_bert_tokenizer
from ...metric.algorithms.base import AttackMetric
from typing import List, Optional
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
from ...exceptions import WordNotInDictionaryException
import random
from ...metric import UniversalSentenceEncoder
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
                 sentence_encoder:Optional[AttackMetric] = None,
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
        if sentence_encoder is not None:
            lst.append(sentence_encoder)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language %s" % lang)

        if tokenizer is None:
            # tokenizer = get_bert_tokenizer(self.__lang_tag)
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        if substitute is None:
            substitute = get_default_substitute(
                self.__lang_tag)  # 这个东西它下载不动，得去wordnet官网下载，还得是2.0版本以下的，放在根目录的data下，按照他的报错格式命名
        self.substitute = substitute

        if sentence_encoder is None:
            sentence_encoder = UniversalSentenceEncoder()
        self.sentence_encoder = sentence_encoder

        check_language([self.tokenizer, self.substitute], self.__lang_tag)

        self.token_unk = token_unk

    def attack(self, victim: Classifier, sentence: str, goal: ClassifierGoal):
        """

        """
        x_orig = sentence.lower()
        orig_probs = victim.get_prob([x_orig])[0]
        orig_label = orig_probs.argmax()
        orig_prob = orig_probs.max()

        x_orig = self.tokenizer.tokenize(x_orig)
        # print("x_orig = ",x_orig)
        x_pos = list(map(lambda x: x[1], x_orig))
        x_orig = list(map(lambda x: x[0], x_orig))

        len_text = len(x_orig)
        # print("len_text = ",len_text)

        words_perturb = [(x_orig[i], x_pos[i]) for i in range(len_text)]

        synonym_words = [
            self.get_neighbours(word, pos)  # TODO
            for word, pos in words_perturb
        ]
        # print("synonym_words = ", synonym_words)
        substitutable_nums = sum([1 if item != [] else 0 for item in synonym_words])
        substitutable_words = []
        for i in range(len(synonym_words)):
            if synonym_words[i] != []:
                substitutable_words.append((i, synonym_words[i]))

        if substitutable_nums / len(synonym_words) > 0.3:
            indices = random.sample(substitutable_words, int(0.3 * len(synonym_words)))
        else:
            indices = substitutable_words
        random.shuffle(indices)

        # print("x_orig = ",x_orig)
        x_disturb = x_orig.copy()
        x_adv = ""
        replaced = []
        for item in indices:
            replaced_word = random.sample(item[1], 1)[0]
            x_disturb[item[0]] = replaced_word
            x_adv = self.tokenizer.detokenize(x_disturb)
            print("x_adv = ",x_adv)
            pred = victim.get_pred([x_adv])[0]
            replaced.append((item[0],x_orig[item[0]],replaced_word))
            if goal.check(x_adv, pred):
                break

        pred = victim.get_pred([x_adv])[0]
        if not goal.check(x_adv, pred):
            return None

        # random.shuffle(replaced)
        scores = []
        for item in replaced:
            _disturb = x_disturb.copy()
            _disturb[item[0]] = item[1]
            x_adv = self.tokenizer.detokenize(_disturb)
            if goal.check(x_adv, pred):
                scr = self.get_sim(x_orig,_disturb)
                scores.append((scr,item[1],item[0]))

        scores = sorted(scores,key=lambda x:-x[0])
        # print("scores = ",scores)
        _disturb = x_disturb.copy()
        for xi in scores:
            _disturb[xi[2]] = xi[1]
            if not goal.check(x_adv, pred):
                break
            x_disturb = _disturb

        x_star = x_disturb


        # print(indices)
        return x_orig

    def get_neighbours(self, word, pos):
        try:
            return list(
                filter(
                    lambda x: x != word,
                    map(
                        lambda x: x[0],
                        self.substitute(word, pos),
                    )
                )
            )
        except WordNotInDictionaryException:
            return []

    def pos_filter(self, ori_pos, new_pos_list):
        same = [True if ori_pos == new_pos or (
                    set([ori_pos, new_pos]) <= set(['noun', 'verb']))  # 集合比较大小其实就是在比较集合的包含关系。？这个地方不明白为什么是名词或动词都可以
                else False for new_pos in new_pos_list]
        return same

    def get_sim(self, ori_text, replaced_text):
        ori_text = self.tokenizer.detokenize(ori_text)
        replaced_text = self.tokenizer.detokenize(replaced_text)
        return self.sentence_encoder.calc_score(replaced_text, ori_text)
