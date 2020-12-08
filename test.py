from simpletransformers.classification import ClassificationModel
import pandas as pd
import json
from emot.emo_unicode import UNICODE_EMO
from gensim.parsing.preprocessing import remove_stopwords

# Initialize variables

DATA_PATH = './data/test.jsonl'

word_dist = []
pred = []

# Converts emojis into text
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
    return text


def predict_sarcasm(data_path):
    # Bringing in the test data
    with open(data_path, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        pred.append(json.loads(json_str))

    pred_response = [remove_stopwords(convert_emojis(pred[i]['response'])) for i in range(len(pred))]
    pred_id = [pred[i]['id'] for i in range(len(pred))]

    model = ClassificationModel("electra", "outputs", use_cuda=False)

    predictions, raw_outputs = model.predict(pred_response)

    pred_bert = pd.DataFrame({
        'id': pred_id,
        'label': predictions
    })

    pred_bert['label'] = pred_bert['label'].replace([1, 0], ['SARCASM', 'NOT_SARCASM'])


predict_sarcasm(DATA_PATH)
