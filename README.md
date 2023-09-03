# Overview
In AmazonReviewsSentimentAnalysisML, we will do a sentiment analysis in python using two different techniques: VADER (Valence Aware Dictionary and sEntiment Reasoner) - Bag of words approach and Roberta Pretrained Model from 🤗 (HuggingFace)

## File

* https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

## Installation

* Python
* Pandas
  * open source data analysis and manipulation tool
* Numpy
  * Python library that adds support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
* Matplotlib
  * Python library for creating high-quality 2D and limited 3D data visualizations
* Seaborn
  * Python data visualization library built on top of Matplotlib, designed for creating attractive, informative statistical graphics with minimal code
* NLTK (Natural Language Toolkit) 
  * Python library used for natural language processing (NLP) and text analysis, providing tools and resources for tasks such as tokenization, stemming, lemmatization, part-of-speech tagging, and more
* SentimentIntensityAnalyzer
  * component provided by the Natural Language Toolkit (NLTK) library in Python, designed for analyzing and determining the sentiment of text data, assigning a numerical sentiment score that indicates the text's overall sentiment polarity, ranging from negative to positive
* tqdm
  * Python library that provides a fast and extensible progress bar for loops and iterable operations, making it easy to monitor the progress of tasks that may take some time to complete by displaying a dynamic progress bar with information about the percentage completed and the estimated time remaining.
* AutoTokenizer
  * Hugging Face Transformers library, a popular library for natural language processing (NLP) tasks. The AutoTokenizer class is designed to automatically select and load the appropriate tokenizer for a given pretrained model architecture, allowing you to tokenize text data in a consistent and efficient manner regardless of the underlying model. It simplifies the process of working with various pretrained models by abstracting the tokenizer selection and configuration.
* AutoModelForSequenceClassification
  * Hugging Face Transformers library that automatically loads pretrained transformer-based models fine-tuned for sequence classification tasks, making it easier to perform tasks like sentiment analysis and text classification in natural language processing.
* Softmax
  * SciPy library that computes the an input array, which scales the values in the array to represent a probability distribution.
