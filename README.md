# Detecting Insults in Social Commentary
## Introduction
This project is based on the following Kaggle competition:
https://www.kaggle.com/c/detecting-insults-in-social-commentary
The challenge is to classify whether a comment is insulting or not.
## Approach
I implement a simple Bidirectional Recurrent Neural Network with GRU cell in Keras to perform the task.
<br/>
In order to improve the model's ROC-AUC, which can be interpreted as the probability of a randomly chosen positive datapoint is ranked higher than a randomly chosen negative datapoint, I implement a pairwise ranking loss for the model. This combines with the GRU-RNN allow my model to reach a ROC-AUC of over 0.87, higher than any model on the leaderboard of the competition.
<br/>
I also attempt to implement a StackedBiRNN and augment the data through translating existing comments to another language and then back to English, but these tricks do not seem to consistently help.
## Final Result (GRU-BiRNN + Translation Augmentation + Pairwise Loss)
Test Accuracy: 0.835663 (Decision Threshold: 0.99)
<br/>
ROC AUC: 0.8790744113159672
