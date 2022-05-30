import OpenAttack
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import datasets
import transformers
from transformers import BertModel

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

def main():
    victim = BertModel.from_pretrained('data/Victim.BERT.SST')

    attacker = OpenAttack.attackers.GBDAAttacker(victim)

    dataset = datasets.load_dataset("sst", split="train[:100]").map(function=dataset_mapping)

    attacker_eval = OpenAttack.AttackEval(attacker,victim)

    attacker_eval.eval(dataset,visualize=True,progress_bar=True)


if __name__ == "__main__":
    main()