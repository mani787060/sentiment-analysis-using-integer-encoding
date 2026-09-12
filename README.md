# Sentiment Analysis Using Integer Encoding

## Overview

This project demonstrates the basic workflow of **Sentiment Analysis using Integer Encoding** with the IMDB movie review dataset available through Keras.

The main focus of this notebook is to understand how textual reviews can be converted into numerical sequences using **integer encoding** so that they can be processed by a neural network for sentiment classification.

---

## Objective

The objectives of this project are to:

* Understand the IMDB movie review dataset.
* Learn how text data is represented numerically.
* Understand **integer encoding** of words.
* Convert movie reviews into sequences of integer IDs.
* Prepare encoded text data for a neural network.
* Understand the basic workflow of sentiment classification.

---

## Dataset

The project uses the built-in **IMDB dataset** provided by Keras.

```python
from keras.datasets import imdb

(X_train, y_train), (X_test, y_test) = imdb.load_data()
```

The dataset contains movie reviews along with their sentiment labels.

* `X_train` → Training reviews represented as integer sequences
* `y_train` → Training sentiment labels
* `X_test` → Testing reviews represented as integer sequences
* `y_test` → Testing sentiment labels

The sentiment labels represent the review category:

* `0` → Negative
* `1` → Positive

---

## What Is Integer Encoding?

Neural networks cannot directly process raw text. Therefore, words need to be converted into numbers.

In integer encoding, each word in the vocabulary is assigned a unique integer ID.

For example:

```text
"I loved this movie"
        ↓
[10, 25, 7, 42]
```

Here, each number represents a particular word according to the vocabulary mapping.

This converts a text review into a numerical sequence that can be used as input to a neural network.

---

## Project Workflow

```text
IMDB Movie Reviews
        ↓
Load Dataset
        ↓
Vocabulary / Word Index
        ↓
Integer Encoding
        ↓
Numerical Review Sequences
        ↓
Prepare Data for Neural Network
        ↓
Sentiment Classification
```

---

## Key Concepts Covered

### 1. Natural Language Processing

NLP allows computers to work with human language. Sentiment analysis is an NLP task used to determine the emotional or opinion-based meaning of text.

### 2. Sentiment Analysis

Sentiment analysis classifies a review based on its sentiment.

For this project:

```text
Movie Review → Positive / Negative
```

### 3. Tokenization

Text is divided into smaller units called tokens, commonly words.

```text
"I love this movie"
        ↓
["I", "love", "this", "movie"]
```

### 4. Integer Encoding

Each word is represented by an integer ID.

```text
Word → Integer ID
```

This transforms textual information into numerical sequences.

### 5. Sequential Data

The encoded review is represented as a sequence of integers, preserving the order of words in the original review.

This makes the data suitable for sequence-based neural network architectures.

---

## Why Integer Encoding Is Important

Machine learning and deep learning models require numerical input. Integer encoding provides a simple way to convert textual information into numbers.

However, the integer IDs themselves do **not** represent the semantic meaning or similarity between words.

For example:

```text
word A → 10
word B → 11
```

This does not mean that word A and word B are semantically similar.

For this reason, integer-encoded sequences are commonly followed by an **Embedding layer**, which learns meaningful vector representations of words.

---

## Technologies Used

* Python
* TensorFlow
* Keras
* IMDB Dataset
* Natural Language Processing
* Integer Encoding
* Deep Learning

---

## Key Learnings

Through this project, I learned:

* How text data is represented numerically.
* How the Keras IMDB dataset works.
* The concept of integer encoding.
* How movie reviews are represented as integer sequences.
* Why neural networks require numerical input.
* The difference between integer IDs and meaningful word representations.
* Why integer encoding is an important preprocessing step for NLP models.

---

## Future Improvements

This project can be extended by adding:

* Sequence padding using `pad_sequences`.
* An Embedding layer.
* SimpleRNN for sentiment classification.
* LSTM or GRU models.
* Model evaluation using accuracy, precision, and recall.
* Training and validation visualization.
* Comparison between different sequence models.

---

## Conclusion

This project provides a foundation for understanding how textual movie reviews are converted into numerical sequences using **integer encoding**.

It is an important preprocessing step toward building complete NLP and sentiment-analysis systems using deep learning. The concepts learned here can be further extended with Embedding layers and recurrent architectures such as **SimpleRNN, LSTM, and GRU**.
