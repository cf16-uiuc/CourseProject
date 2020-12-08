from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import json
from emot.emo_unicode import UNICODE_EMO
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import remove_stopwords

# Initialize variables

DATA_INPUT = './data/train.jsonl'
EARLY_STOP = True
EARLY_STOP_DELTA = 0.01
OVERWRITE = True
EPOCHS = 3
BATCH_SIZE = 100
LEARNING_RATE = 0.00004
MODEL_TYPE = 'bert'
MODEL_BASE = 'bert-base-cased'

word_dist = []
train = []
test = []
pred = []

# Converts emojis into text
def convert_emojis(text):
    for emot in UNICODE_EMO:
        text = text.replace(emot, "_".join(UNICODE_EMO[emot].replace(",", "").replace(":", "").split()))
    return text


# Train BERT model
def bert_training(model_type, model_base, train_data, early_stop,
                  early_stop_delta, overwrite, epoch, batch_size,
                  learning_rate):
    # Bringing in the training data
    with open(train_data, 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        train.append(json.loads(json_str))

    train_labels = [train[i]['label'] for i in range(len(train))]

    train_response = [remove_stopwords(convert_emojis(train[i]['response'])) for i in range(len(train))]

    labels_train, labels_test, response_train, response_test = train_test_split(train_labels,
                                                                                train_response,
                                                                                test_size=0.2,
                                                                                random_state=42)

    # Bringing in the testing data
    with open('./data/test.jsonl', 'r') as json_file:
        json_list = list(json_file)

    for json_str in json_list:
        pred.append(json.loads(json_str))

    pred_response = [convert_emojis(pred[i]['response']) for i in range(len(pred))]

    labels_train_pd = (pd.DataFrame(labels_train) == 'SARCASM').astype(int)
    labels_test_pd = (pd.DataFrame(labels_test) == 'SARCASM').astype(int)
    response_train_pd = pd.DataFrame(response_train)
    response_test_pd = pd.DataFrame(response_test)

    train_bert = pd.DataFrame({
        'text': response_train_pd[0].replace(r'\n', ' ', regex=True),
        'label': labels_train_pd[0]
    })

    eval_bert = pd.DataFrame({
        'text': response_test_pd[0].replace(r'\n', ' ', regex=True),
        'label': labels_test_pd[0]
    })

    model_args = ClassificationArgs()
    model_args.use_early_stopping = early_stop
    model_args.early_stopping_delta = early_stop_delta
    model_args.overwrite_output_dir = overwrite
    model_args.num_train_epochs = epoch
    model_args.train_batch_size = batch_size
    model_args.learning_rate = learning_rate

    # Create a TransformerModel
    model = ClassificationModel(model_type, model_base, use_cuda=False,
                                args=model_args)

    # Train the model
    model.train_model(train_bert)

    # Evaluate the model
    result, model_outputs, wrong_predictions = model.eval_model(eval_bert)


bert_training(MODEL_TYPE, MODEL_BASE, DATA_INPUT, EARLY_STOP,
              EARLY_STOP_DELTA, OVERWRITE, EPOCHS, BATCH_SIZE, LEARNING_RATE)
