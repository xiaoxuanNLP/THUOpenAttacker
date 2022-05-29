import OpenAttack
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import datasets

def make_model():
    class MyClassifier(OpenAttack.Classifier):
        def __init__(self):
            try:
                self.model = SentimentIntensityAnalyzer()
            except LookupError:
                nltk.download('vader_lexicon')
                self.model = SentimentIntensityAnalyzer()
        
        def get_pred(self, input_):
            return self.get_prob(input_).argmax(axis=1)

        def get_prob(self, input_):
            ret = []
            for sent in input_:
                res = self.model.polarity_scores(sent)
                prob = (res["pos"] + 1e-6) / (res["neg"] + res["pos"] + 1e-6)
                ret.append(np.array([1 - prob, prob]))
            return np.array(ret)
    return MyClassifier()

def dataset_mapping(x):
    return {
        "x": x["sentence"],
        "y": 1 if x["label"] > 0.5 else 0,
    }

def main():

    print("New Attacker")
    # attacker = OpenAttack.attackers.PWWSAttacker()
    # attacker = OpenAttack.attackers.BAEAttacker()

    # print("Build model")
    # clsf = make_model()
    # clsf = OpenAttack.loadVictim("ALBERT.IMDB")
    # print(clsf)
    # print(type(clsf))
    clsf = OpenAttack.loadVictim("BERT.SST")

    # dataset = datasets.load_dataset("sst", split="train[:100]").map(function=dataset_mapping)

    # print("Start attack")
    # attack_eval = OpenAttack.AttackEval( attacker, clsf, metrics=[
    #     OpenAttack.metric.Fluency(),
    #     OpenAttack.metric.GrammaticalErrors(),
    #     OpenAttack.metric.SemanticSimilarity(),
    #     OpenAttack.metric.EditDistance(),
    #     OpenAttack.metric.ModificationRate()
    # ] )
    # attack_eval = OpenAttack.AttackEval(attacker, clsf)
    # attack_eval.eval(dataset, visualize=True, progress_bar=True)



if __name__ == "__main__":
    # main()
    clsf = OpenAttack.loadVictim("BERT.SST")
    print(type(clsf))

#---------------------------------------------------------------------------
    from transformers import AlbertModel, AlbertTokenizer, BertModel,BertTokenizer
    import pytorch_pretrained_bert
    import torch
    import json
    import logging

    logging.getLogger().setLevel(logging.WARNING)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    text = "The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal ."
    tokenizer = BertTokenizer.from_pretrained("./data/Victim.BERT.SST")
    words = ["[CLS]"] + tokenizer.tokenize(text) + ["[SEP]"]

    model = BertModel.from_pretrained("./data/Victim.ALBERT.IMDB").to(device)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    tokenized_ids = tokenizer.convert_tokens_to_ids(words)
    segment_ids = [0 for _ in range(len(words))]
    token_tensor = torch.tensor([tokenized_ids], device=device)
    segment_tensor = torch.tensor([segment_ids], device=device)
    x = model.embeddings(token_tensor, segment_tensor)[0]
    from Interpreter import Interpreter, calculate_regularization



    def Phi(x):
        global     model
        x = x.unsqueeze(0)
        attention_mask = torch.ones(x.shape[:2]).to(x.device)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        # extract the 3rd layer
        model_list = model.encoder.layer[:6]
        # print(type(model_list))
        hidden_states = x
        for layer_module in model_list:
            hidden_states = layer_module(hidden_states, extended_attention_mask)
        return hidden_states[0]


    regularization = calculate_regularization(sampled_x=x, Phi=Phi, device=device)

    interpreter = Interpreter(x=x, Phi=Phi, regularization=regularization, words=words).to(
        device
    )
    interpreter.optimize(iteration=5000, lr=0.01, show_progress=False)
    interpreter.get_sigma()
    interpreter.visualize()
#-----------------------------------------
    from transformers import BertLayer
    # from pytorch_pretrained_bert import BertModel
    # model = BertModel.from_pretrained("bert-base-uncased")