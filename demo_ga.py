import OpenAttack
import datasets
from OpenAttack.data_manager import DataManager
from transformers import BertTokenizer
from OpenAttack.text_process.tokenizer import get_bert_tokenizer
from OpenAttack.tags import TAG_English

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
    tokenizer = get_bert_tokenizer(TAG_English)
    # attack_eval = OpenAttack.AttackEval(attacker, clsf,tokenizer=tokenizer)
    attack_eval = OpenAttack.AttackEval(attacker, clsf)

    attack_eval.eval(dataset, visualize=True, progress_bar=True)

if __name__ == "__main__":
    # sent_tokenizer = DataManager.load("TProcess.NLTKSentTokenizer")
    main()
    from transformers import BertTokenizer
    # x = "you are"
