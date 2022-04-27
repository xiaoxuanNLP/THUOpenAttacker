from typing import List, Optional
import random
import numpy as np

from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...attack_assist.substitute.word import WordSubstitute, get_default_substitute
from ...utils import get_language, check_language, language_by_name
from ...exceptions import WordNotInDictionaryException
from ...tags import Tag
from ...attack_assist.filter_words import get_default_filter_words


class TextBuggerAttacker(ClassificationAttacker):

    @property
    def TAGS(self):
        ret = {self.__lang_tag, Tag("get_pred", "victim")}
        if self.blackbox:
            ret.add(Tag("get_prob", "victim"))  # 得到重要性
        else:
            ret.add(Tag("get_grad", "victim"))  # 得到梯度
        return ret

    def __init__(self,
                 blackbox=True,
                 tokenizer: Optional[Tokenizer] = None,
                 substitute: Optional[WordSubstitute] = None,
                 filter_words: List[str] = None,
                 lang=None
                 ):
        """
        TEXTBUGGER: Generating Adversarial Text Against Real-world Applications. Jinfeng Li, Shouling Ji, Tianyu Du, Bo Li, Ting Wang. NDSS 2019.
        `[pdf] <https://arxiv.org/pdf/1812.05271.pdf>`__

        Args:
            blackbox: If is true, the attacker will perform a black-box attack.
            tokenizer: A tokenizer that will be used during the attack procedure. Must be an instance of :py:class:`.Tokenizer`
            substitute: A substitute that will be used during the attack procedure. Must be an instance of :py:class:`.WordSubstitute`
            lang: The language used in attacker. If is `None` then `attacker` will intelligently select the language based on other parameters.
            filter_words: A list of words that will be preserved in the attack procesudre.

        :Classifier Capacity:
            * get_pred
            * get_prob `if blackbox = True`
            * get_grad `if blackbox = False`


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
                raise ValueError("Unknown language `%s`" % lang)

        if substitute is None:
            substitute = get_default_substitute(self.__lang_tag)
        self.substitute = substitute

        if tokenizer is None:
            tokenizer = get_default_tokenizer(self.__lang_tag)
        self.tokenizer = tokenizer

        check_language([self.tokenizer, self.substitute], self.__lang_tag)

        self.glove_vectors = None
        self.blackbox = blackbox

        if filter_words is None:
            filter_words = get_default_filter_words(self.__lang_tag)
        self.filter_words = set(filter_words)

    def attack(self, victim: Classifier, sentence, goal: ClassifierGoal):
        x = self.tokenizer.tokenize(sentence, pos_tagging=False)

        if self.blackbox:
            ranked_words = self.get_word_importances(x, victim, goal)
        else:
            ranked_words = self.get_w_word_importances(x, victim, goal)
        for word_idx in ranked_words:
            word = x[word_idx]
            if word in self.filter_words:
                continue
            bug = self.selectBug(word, word_idx, x, victim, goal)
            x = self.replaceWithBug(x, word_idx, bug)
            x_prime_sentence = self.tokenizer.detokenize(x)
            prediction = victim.get_pred([x_prime_sentence])[0]

            if goal.check(x_prime_sentence, prediction):
                return x_prime_sentence

        return None

    def get_word_importances(self, sentence_tokens, clsf, goal: ClassifierGoal):  # clsf是victim模型
        word_losses = {}
        for i in range(len(sentence_tokens)):
            sentence_tokens_without = sentence_tokens[:i] + sentence_tokens[i + 1:]  # 可以看出，这个地方就是直接把对应位置的单词摘出来，然后组成一个句子
            sentence_without = self.tokenizer.detokenize(sentence_tokens_without)
            tempoutput = clsf.get_prob([sentence_without])[0]
            word_losses[i] = tempoutput[goal.target]
        word_losses = [k for k, _ in sorted(word_losses.items(), key=lambda item: item[1], reverse=goal.targeted)]
        return word_losses

    def get_w_word_importances(self, sentence_tokens, clsf, y_orig):  # white
        _, grad = clsf.get_grad([sentence_tokens], [y_orig])
        grad = grad[0]
        if grad.shape[0] != len(sentence_tokens):
            raise RuntimeError("Sent %d != Gradient %d" % (len(sentence_tokens), grad.shape[0]))
        dist = np.linalg.norm(grad, axis=1)

        return [idx for idx, _ in sorted(enumerate(dist.tolist()), key=lambda x: -x[1])]

    def selectBug(self, original_word, word_idx, x_prime, clsf, goal):
        bugs = self.generateBugs(original_word, self.glove_vectors)
        max_score = float('-inf')
        best_bug = original_word
        for bug_type, b_k in bugs.items():
            candidate_k = self.replaceWithBug(x_prime, word_idx, b_k)
            score_k = self.getScore(candidate_k, clsf, goal)  # 获得干扰预测的分数
            if score_k > max_score:
                best_bug = b_k
                max_score = score_k
        return best_bug

    def getScore(self, candidate, clsf, goal):  # 这个地方就是，如果是有目标的，那就离目标尽可能近，否则就离正确值尽可能远即可
        candidate_sentence = self.tokenizer.detokenize(candidate)
        tempoutput = clsf.get_prob([candidate_sentence])[0]
        if goal.targeted:
            return tempoutput[goal.target]
        else:
            return - tempoutput[goal.target]

    def replaceWithBug(self, x_prime, word_idx, bug):
        return x_prime[:word_idx] + [bug] + x_prime[word_idx + 1:]

    def generateBugs(self, word, glove_vectors, sub_w_enabled=False, typo_enabled=False):  # 这个函数是生成干扰策略的决策器
        bugs = {"insert": word, "delete": word, "swap": word, "sub_C": word, "sub_W": word}
        if len(word) <= 2:  # 如果整个句子（包含标点）小于2，那不操作
            return bugs
        bugs["insert"] = self.bug_insert(word)  # 当单词小于等于6的时候不插，否则在单词之间随便找个位置，插入歌空格
        bugs["delete"] = self.bug_delete(word)  # 随便删一个
        bugs["swap"] = self.bug_swap(word)  # 小于4的句子不换
        bugs["sub_C"] = self.bug_sub_C(word)  # 字母的键盘临近替换和形似替换
        bugs["sub_W"] = self.bug_sub_W(word)  # 找意思最相近的词
        return bugs

    def bug_sub_W(self, word):
        try:
            res = self.substitute(word, None)
            if len(res) == 0:
                return word
            return res[0][0]
        except WordNotInDictionaryException:
            return word

    def bug_insert(self, word):
        if len(word) >= 6:
            return word
        res = word
        point = random.randint(1, len(word) - 1)
        res = res[0:point] + " " + res[point:]
        return res

    def bug_delete(self, word):
        res = word
        point = random.randint(1, len(word) - 2)
        res = res[0:point] + res[point + 1:]
        return res

    def bug_swap(self, word):
        if len(word) <= 4:
            return word
        res = word
        points = random.sample(range(1, len(word) - 1), 2)
        a = points[0]
        b = points[1]

        res = list(res)
        w = res[a]
        res[a] = res[b]
        res[b] = w
        res = ''.join(res)
        return res

    def bug_sub_C(self, word):
        res = word
        key_neighbors = self.get_key_neighbors()
        point = random.randint(0, len(word) - 1)

        if word[point] not in key_neighbors:
            return word
        choices = key_neighbors[word[point]]
        subbed_choice = choices[random.randint(0, len(choices) - 1)]
        res = list(res)
        res[point] = subbed_choice
        res = ''.join(res)

        return res

    def get_key_neighbors(self):
        ## TODO: support other language here
        # By keyboard proximity
        neighbors = {
            "q": "was", "w": "qeasd", "e": "wrsdf", "r": "etdfg", "t": "ryfgh", "y": "tughj", "u": "yihjk",
            "i": "uojkl", "o": "ipkl", "p": "ol",
            "a": "qwszx", "s": "qweadzx", "d": "wersfxc", "f": "ertdgcv", "g": "rtyfhvb", "h": "tyugjbn",
            "j": "yuihknm", "k": "uiojlm", "l": "opk",
            "z": "asx", "x": "sdzc", "c": "dfxv", "v": "fgcb", "b": "ghvn", "n": "hjbm", "m": "jkn"
        }  # 键盘上的位置
        # By visual proximity # 形似
        neighbors['i'] += '1'
        neighbors['l'] += '1'
        neighbors['z'] += '2'
        neighbors['e'] += '3'
        neighbors['a'] += '4'
        neighbors['s'] += '5'
        neighbors['g'] += '6'
        neighbors['b'] += '8'
        neighbors['g'] += '9'
        neighbors['q'] += '9'
        neighbors['o'] += '0'

        return neighbors
