import OpenAttack
import datasets
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import datasets


def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }


def main():
    print("New Attacker")
    attacker = OpenAttack.attackers.PWWSAttacker(lang="english")

    print("Building model")
    clsf = OpenAttack.loadVictim("BERT.SST")

    print("Loading dataset")
    dataset = datasets.load_dataset("sst").map(function=dataset_mapping)

    print("start attack")
    attacker_eval = OpenAttack.AttackEval(attacker,clsf,metrics=[
        OpenAttack.metric.Fluency(),
        OpenAttack.metric.GrammaticalErrors(),
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attacker_eval.eval(dataset, visualize=True, progress_bar=True)

if __name__ == "__main__":
    main()