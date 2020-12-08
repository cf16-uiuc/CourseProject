# Text Classification Competition: Twitter Sarcasm Detection 

This project is for CS410 during the fall 2020 semester at UIUC. More details about this competition can be found at:
https://github.com/CS410Fall2020/ClassificationCompetition

# Implementation Overview

This classifier relies on the Simple Transformer package, based on the Transformer package from HuggingFace. The documentation for Simple Transformer can be foundh here: https://simpletransformers.ai/docs/classification-models/#classificationmodel

The code is broken into two files - train.py and test.py. In train.py the focus of the file is training the model. Prior to training the model we do a little bit of preprocessing of the data. First, we remove all stop words, then we replace all emoticons with text that may be able to provide information to the model that is trained. For the training we rely on the Simple Transformer package. From that package we use the binary classifier. Their setup allows us to bring in any pretrained model and adjust to our data. We looked at multiple pretrained models from HuggingFace found here: https://huggingface.co/models. In the end we discovered that simply using the BERT model yielded the best results. 

To apply labels to a new set of tweets we can run the test.py file. This does a similar process of converting emoticons to text and removing stop words to preprocess the data. We then read in our trained model from train.py and can use the predict() function to predict the new labels. 

# Running the Code

To run the code you need to ensure that all the required packages are installed. The code can be run from the command line by running either python train.py or python test.py, depending on which file you want to run. The different variables, such as file name for training data, number of epochs, or learning rate can be adjusted within the file at the beginning of the file. The end result of running train.py will be folder titled outputs. The test.py file reads from the outputs folder and will output and an answers.txt file. 

# Useful Links

https://huggingface.co/models
https://simpletransformers.ai/docs/binary-classification/
https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3
https://github.com/ThilinaRajapakse/simpletransformers
