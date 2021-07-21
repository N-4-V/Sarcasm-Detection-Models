# SARCASM DETECTION MODELS

# Approach
Pre-processing using conventional approach is done to minimize the content of the text so that it's easier to convert to numeric data for the various models. However, doing the same to Sarcastic text removes the essence of what makes it Sarcastic, thereby greatly reducing the potential highest accuracies of the models. In this code, I've attempted to maximize the structure of data so as to maximize accuracy. This is the code that goes along with the research I did, showing that the proposed pre-processing can increase the accuracies of all models by upto 11%, with an average of 6% for the Machine Learning models. For Neural Networks however, the change is minimal.

# Resources
Dataset used : https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection?select=Sarcasm_Headlines_Dataset.json

Feature Extractors used : 
Count Vectorizer, TF-IDF Vectorizer, Hashing Vectorizer for balanced subset of 10k, containing 5000 sarcastic and 5000 non-sarcastic news headlines.
Pre-trained GloVe (100d, based on 6 Billion words) for full unbalanced dataset of 55,328 entries, taken from Kaggle.                                           

Machine Learning Models used : Gaussian Naive Bayes, Decision Tree, Logistic Regression, Support Vector Machine, K Nearest Neighbors, Random Forest. All with Count, TF-IDF and Hashing Vectorizers except for Random Forest which gives error with Hashing Vectorizer.

Neural Networks used : Convoluted Neural Network and Bidirectional Long Short Term Memory. Both used with Count and TF-IDF Vectorizers for the 10k dataset, and with 
pre-trained GLoVe for the 55k dataset.

# Results
For ML models on 10k : 82.89% accuracy for Logistic Regression with Count Vectorizer.

For Neural Networks on 55k : 91.94% accuracy for Bidirectional Long Short Term Memory with pre-trained GloVe
