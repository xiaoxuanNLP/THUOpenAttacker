from typing import List, Optional
import datasets
from tqdm import tqdm

from ..classification import ClassificationAttacker, Classifier, ClassifierGoal
from ...text_process.tokenizer import Tokenizer, get_default_tokenizer
from ...utils import get_language, check_language, language_by_name
from ...tags import Tag
from OpenAttack.victim.classifiers import TransformersClassifier, TransformersLearnable
import torch
import torch.nn.functional as F
import jiwer
import numpy as np
from transformers import AutoModelForCausalLM, BertModel


class GBDAAttacker(ClassificationAttacker):
    @property
    def TAGS(self):
        return {self.__lang_tag, Tag("get_pred", "victim")}

    def __init__(self,
                 ref_model,
                 tokenizer: Optional[Tokenizer] = None,
                 lang=None,
                 num_labels=2):
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
        self.model = BertModel.from_pretrained(ref_model, num_labels).cuda()
        self.batch_size = 10

    def attack(self, victim: TransformersLearnable, sentence: str, goal: ClassifierGoal):
        with torch.no_grad():
            embeddings = self.model.get_input_embeddings()(torch.arange(0, self.tokenizer.vocab_size).long().cuda())
            ref_embeddings = victim.get_input_embeddings()(torch.arange(0, self.tokenizer.vocab_size).long().cuda())

        ids = self.tokenizer(sentence)

        forbidden = np.zeros(len(ids)).astype('bool')
        forbidden[0] = True
        forbidden[-1] = True
        forbidden_indices = np.arange(0, len(ids))[forbidden]
        forbidden_indices = torch.from_numpy(forbidden_indices).cuda()

        ids = ids.long().cuda()
        clean_logit = self.model(input_ids=ids)

        adv_logits, adv_texts = [], []

        with torch.no_grad():
            orig_output = victim.get_learnable_logits(sentence).hidden_states[-1]
            orig_output = orig_output[:, -1]

            log_coeffs = torch.zeros(len(ids), embeddings.size(0))
            indices = torch.arange(log_coeffs.size(0)).long()
            log_coeffs[indices, torch.LongTensor(ids)] = 15
            log_coeffs = log_coeffs.cuda()
            log_coeffs.requires_grad = True

        optimizer = torch.optim.Adam([log_coeffs], lr=3e-1)
        label = goal.get_label()
        label = torch.tensor(label, dtype=torch.long)
        for i in range(100):
            optimizer.zero_grad()
            coeffs = F.gumbel_softmax(log_coeffs.unsqueeze(0).repeat(self.batch_size, 1, 1), hard=False)  # B x T x V
            inputs_embeds = (coeffs @ embeddings[None, :, :])  # B x T x D
            pred = self.model(inputs_embeds=inputs_embeds).logits

            adv_loss = -F.cross_entropy(pred, label * torch.ones(self.batch_size).long().cuda())

            ref_embeds = (coeffs @ ref_embeddings[None, :, :])
            pred = victim.get_pred_by_embeds(inputs_embeds=ref_embeds)

            output = pred.hidden_states[-1]
            output = output[:, -1]

            cosine = (output * orig_output).sum(1) / output.norm(2, 1) / orig_output.norm(2, 1)
            ref_loss = - cosine.mean()

            perp_loss = self.log_perplexity(pred.logits, coeffs)

            total_loss = adv_loss + ref_loss + perp_loss
            total_loss.backward()

            entropy = torch.sum(-F.log_softmax(log_coeffs, dim=1) * F.softmax(log_coeffs, dim=1))

            # Gradient step
            log_coeffs.grad.index_fill_(0, forbidden_indices, 0)
            optimizer.step()

        with torch.no_grad():
            for j in range(100):
                adv_ids = F.gumbel_softmax(log_coeffs, hard=True).argmax(1)

                adv_ids = adv_ids[1:len(adv_ids) - 1].cpu().tolist()
                adv_text = self.tokenizer.decode(adv_ids)
                x = self.tokenizer(adv_text, max_length=256, truncation=True, return_tensors='pt')

                adv_logit = self.model(input_ids=x['input_ids'].cuda(), attention_mask=x['attention_mask'].cuda(),
                                       token_type_ids=(x['token_type_ids'].cuda() if 'token_type_ids' in x else None)).logits.data.cpu()
                if adv_logit.argmax() != label or j == 100 - 1:
                    adv_texts.append(adv_text)
                    print(adv_text)
                    adv_logits.append(adv_logit)
                    break
        if len(adv_texts) != 0:
            return adv_texts[0]
        return None

    def wer(self, x, y):
        x = " ".join(["%d" % i for i in x])
        y = " ".join(["%d" % i for i in y])

        return jiwer.wer(x, y)

    def bert_score(self, refs, cands, weights=None):
        refs_norm = refs / refs.norm(2, -1).unsqueeze(-1)
        if weights is not None:
            refs_norm *= weights[:, None]
        else:
            refs_norm /= refs.size(1)
        cands_norm = cands / cands.norm(2, -1).unsqueeze(-1)
        cosines = refs_norm @ cands_norm.transpose(1, 2)
        cosines = cosines[:, 1:-1, 1:-1]
        R = cosines.max(-1)[0].sum(1)
        return R

    def log_perplexity(self, logits, coeffs):
        shift_logits = logits[:, :-1, :].contiguous()
        shift_coeffs = coeffs[:, 1:, :].contiguous()
        shift_logits = shift_logits[:, :, :shift_coeffs.size(2)]
        return -(shift_coeffs * F.log_softmax(shift_logits, dim=-1)).sum(-1).mean()
