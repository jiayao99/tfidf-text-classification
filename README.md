# TUTORIAL: Text Classification with TF-IDF

This tutorial provides a guide on utilizing TF-IDF (Term Frequency-Inverse Document Frequency) for the classification of textual data.

**Implementation**: [resume_classification_tfidf.ipynb](https://github.com/jiayao99/tfidf-text-classification/blob/main/resume_classification_tfidf.ipynb)

## What is \*TF-IDF\*?

TF-IDF stands for Term Frequency-Inverse Document Frequency.

It measures **importance of a word in a document, which is part of a corpus**.

Despite its simplicity and lower computational footprint as compared to sophisticated pretrained word embedding models (such as BERT, Word2Vec, etc.), TF-IDF can often yield surprisingly effective results in various classification tasks.


### Here’s how it works:

**Term Frequency (TF)**:

This measures how frequently a term appears in a document. Since every document is different in length, it is possible that a term would appear much more times in longer documents than shorter ones. Thus, the term frequency is often divided by the document length (i.e., the total number of terms in the document) as a way of normalization.

$TF(t, d) = \frac{\text{Number of times term } t \text{ appears in a document } d}{\text{Total number of terms in the document } d}$

**Inverse Document Frequency (IDF)**:

This measures how important a term is. However, it is known that certain terms, like "is", "of", and "that", may appear a lot of times but have little importance. Thus, we need to weigh down the frequent terms while scaling up the rare ones, by computing the following.

$IDF(t, D) = \log\left(\frac{\text{Total number of documents in the corpus } D}{\text{Number of documents with term } t \text{ in them}}\right)$

**Term Frequency-Inverse Document Frequency**:

$TF-IDF(t, d, D) = TF(t, d) \times IDF(t, D)$

where $t$ represents the term or word, $d$ is a document in the corpus, and $D$ is the entire corpus of documents.

*\*Note that TF-IDF is a **SPARSE EMBEDDING**: there are a lot of 0s in the vector.*
