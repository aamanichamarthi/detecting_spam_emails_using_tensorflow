# detecting_spam_emails_using_tensorflow

This project implements a spam email detection system using deep learning with TensorFlow. The goal is to classify emails as either "spam" or "ham" (not spam) based on their content.

# Dataset
The dataset used for this project is spam_ham_dataset.csv.
This CSV file contains a collection of emails, each labeled as either spam or ham. The relevant columns are:

label: Indicates whether the email is 'ham' or 'spam'.

text: The actual content of the email.

label_num: A numerical representation of the label (0 for ham, 1 for spam).

# Features
The detecting_spam_emails_using_tensorflow.ipynb notebook covers the following key aspects:


Data Loading: Reads the spam_ham_dataset.csv file into a Pandas DataFrame.


Data Exploration: Includes basic checks on the dataset's shape and distribution of spam vs. ham emails.

Data Preprocessing:

Removal of "Subject:" prefixes from email texts.

Removal of punctuation from the email content.

Removal of common English stopwords to reduce noise and focus on meaningful terms.


Text Vectorization: Likely uses tensorflow.keras.preprocessing.text.Tokenizer to convert text into numerical sequences and tensorflow.keras.preprocessing.sequence.pad_sequences for uniform input length (inferred from imports).


Model Building: Utilizes tensorflow.keras for constructing a deep learning model, probably for text classification (inferred from imports).


Model Training and Evaluation: Employs sklearn.model_selection.train_test_split for splitting data and keras.callbacks.EarlyStopping, ReduceLROnPlateau for optimized training (inferred from imports).

# Installation
To run this project, you will need the following Python libraries:

numpy

pandas

matplotlib

seaborn

nltk

wordcloud

tensorflow

scikit-learn

keras (usually installed with TensorFlow)
