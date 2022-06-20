import OpenAttack
import datasets

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

def main():
    print("New Attacker")
    attacker = OpenAttack.attackers.GAAttacker()

    print("build Attacker")
    clsf = OpenAttack.DataManager.loadVictim("BERT.SST")

    print("build dataset")
    dataset = datasets.load_dataset("sst", split="train[:100]").map(function=dataset_mapping)

    print("Start attack")
    attack_eval = OpenAttack.AttackEval(attacker, clsf, metrics=[
        OpenAttack.metric.Fluency(),
        OpenAttack.metric.GrammaticalErrors(),
        OpenAttack.metric.EditDistance(),
        OpenAttack.metric.ModificationRate()
    ])
    attack_eval.eval(dataset, visualize=True, progress_bar=True)

if __name__ == "__main__":
    main()