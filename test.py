# import necessary packages

from simpletransformers.classification import ClassificationModel
import pandas as pd
import json
from emot.emo_unicode import UNICODE_EMO
from gensim.parsing.preprocessing import remove_stopwords

# Initialize variables

DATA_PATH = './data/test.jsonl'
FINAL_RESULTS = 'answer.txt'
MODEL_LOCATION = 'outputs'
MODEL_TYPE = 'bert'

word_dist = []
pred = []

# Converts emojis into text
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
    return text


def predict_sarcasm(data_path, results, model_loc, model):
    # Bringing in the test data
    with open(data_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        pred.append(json.loads(json_str))

    pred_response = [remove_stopwords(convert_emojis(pred[i]['response'])) for i in range(len(pred))]
    pred_id = [pred[i]['id'] for i in range(len(pred))]

    model = ClassificationModel(model, model_loc, use_cuda=False)

    predictions, raw_outputs = model.predict(pred_response)

    pred_bert = pd.DataFrame({
        'id': pred_id,
        'label': predictions
    })

    pred_bert['label'] = pred_bert['label'].replace([1, 0], ['SARCASM', 'NOT_SARCASM'])
    pd.DataFrame(pred_bert).to_csv(results, header=False, sep=',', index=False)


predict_sarcasm(DATA_PATH, FINAL_RESULTS, MODEL_LOCATION, MODEL_TYPE)
