import OpenAttack
import numpy as np
import transformers
from transformers import BertTokenizer,AutoModelForSequenceClassification,BertForSequenceClassification
from OpenAttack.victim.classifiers import TransformersClassifier
import pandas as pd
import datasets

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

def main():
    bert_model = BertForSequenceClassification.from_pretrained('./data/Victim.BERT.SST')
    tokenizer = BertTokenizer.from_pretrained('./data/Victim.BERT.SST')
    classifer = TransformersClassifier(bert_model, tokenizer, bert_model.get_input_embeddings())
    vocab = pd.read_csv('./data/Victim.BERT.SST/vocab.txt')
    dataset = datasets.load_dataset("sst", split="train[:100]").map(function=dataset_mapping)
    attacker = OpenAttack.attackers.VATAttacker(vocab,bert_model.get_input_embeddings())
    attack_eval = OpenAttack.AttackEval(attacker, classifer)
    attack_eval.eval(dataset,visualize=True,progress_bar=True)

if __name__ == "__main__":
    main()