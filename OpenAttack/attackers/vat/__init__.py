from typing import List, Optional
import datasets
from tqdm import tqdm

from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...utils import get_language, check_language, language_by_name
from ...tags import Tag
from OpenAttack.victim.classifiers import TransformersClassifier
import torch
import numpy as np

# TODO failed 发现该文章的重点是防御而非攻击
class VATAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return {self.__lang_tag, Tag("get_pred", "victim")}

    def __init__(self,
                 vocab,
                 word_embed,
                 use_attn_d=0,
                 online_nn=0,
                 lang=None,
                 top_k=15,
                 tokenizer: Optional[Tokenizer] = None,
                 ):
        """
                `[pdf] <https://arxiv.org/pdf/1805.02917>`__
                `[code] <https://github.com/aonotas/interpretable-adv>`__

                Args:

                    lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
                :Classifier Capacity:
                    * get_pred

        """
        self.vocab_inv = dict([(widx, w) for w, widx in vocab.items()])
        print('vocab_inv:', len(self.vocab_inv))
        lst = []
        if tokenizer is not None:
            lst.append(tokenizer)
        if len(lst) > 0:
            self.__lang_tag = get_language(lst)
        else:
            self.__lang_tag = language_by_name(lang)
            if self.__lang_tag is None:
                raise ValueError("Unknown language `%s`" % lang)

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        check_language([self.tokenizer], self.__lang_tag)
        self.use_attn_d = use_attn_d
        self.online_nn = online_nn
        self.top_k = top_k
        self.word_embed = word_embed

    # TODO failed 发现该文章的重点是防御而非攻击
    def attack(self, victim: TransformersClassifier, sentence: str, goal: ClassifierGoal):
        word_embs = victim.get_embedding()
        self.compute_all_nearest_words(self.word_embed.weight)


    def compute_all_nearest_words(self, word_embs, offsent=0,batch_size=100 ):
        nearest_ids = None
        vocab_size = len(self.vocab_inv)
        word_embs_T = word_embs.T
        iteration_list = range(0, word_embs.shape[0], batch_size)
        score_list = []
        top_idx_list = []
        for index in iteration_list:
            emb = word_embs[index:index + batch_size]
            scores = torch.dot(emb, word_embs_T)
            top_idx = torch.argsort(-scores, axis=1)

            if offsent >= 0:
                top_idx = top_idx[:, offsent:self.top_k + offsent]
            else:
                top_idx = top_idx[:, -self.top_k:]
            top_idx_list.append(top_idx.cpu)

        nearest_ids = np.concatenate(top_idx_list, axis=0)
        nearest_ids = np.array(nearest_ids, dtype=np.int32)
        nearest_ids = nearest_ids
        self.nearest_ids = nearest_ids
        return nearest_ids

    def get_nearest_words(self,x_data, offsent=0,noise=None):
        if noise is None:
            return self.nearest_ids[x_data]#TODO
        xs_var = self.word_embed(x_data).weight
        if noise is not None:
            xs_var += noise
        vocab_size = len(self.vocab_inv)
        word_embs = self.word_embed.weight[:vocab_size]
        scores = torch.dot(xs_var, word_embs.T)
        top_idx = torch.argsort(-scores, axis=1)
        if offsent >= 0:
            top_idx = top_idx[:, offsent:self.top_k + offsent]
        else:
            top_idx = top_idx[:, :self.top_k]
        return top_idx.long()

    def most_sims(self,word):
        if word not in self.vocab_inv:
            print('[not found]:{}'.format(word))
            return False
        idx = self.vocab_inv[word]
        idx_gpu = np.array([idx], dtype=int)
        top_idx = self.get_nearest_words(idx_gpu)
        sim_ids = top_idx[0]
        words = [self.vocab_inv[int(i)] for i in sim_ids]
        word_line = ','.join(words)
        print('{}\t\t{}'.format(word, word_line))