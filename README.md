# Sentiment Analysis of Customer Reviews using NLP

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: YATIN GOGIA
*NAME*: DIWAKAR VERMA

*INTERN ID*: CT04DH1053

*DOMAIN*: MACHINE LEARNING

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTOSH

## Overview

This project addresses the challenge of automatically classifying customer reviews as positive or negative using Natural Language Processing (NLP). By leveraging a rich dataset of IMDB movie reviews, the system demonstrates the complete workflow from raw data to a deployable sentiment analysis model. The aim is to equip businesses, researchers, and practitioners with scalable tools for understanding public opinion and extracting actionable insights from vast amounts of customer feedback.

## Dataset

- **IMDB Dataset of 50K Movie Reviews**
    - 50,000 movie reviews labeled as **positive** or **negative**
    - Balanced: 25,000 positive and 25,000 negative reviews for both training and testing
    - Download link: [IMDB Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

## Table of Contents

1. [Introduction](#introduction)
2. [Project Workflow](#project-workflow)
    - [Step 1: Environment Setup](#step-1-environment-setup)
    - [Step 2: Exploratory Data Analysis (EDA)](#step-2-exploratory-data-analysis-eda)
    - [Step 3: Data Preprocessing](#step-3-data-preprocessing)
    - [Step 4: Feature Engineering & Vectorization](#step-4-feature-engineering--vectorization)
    - [Step 5: Model Selection and Building](#step-5-model-selection-and-building)
    - [Step 6: Model Training & Evaluation](#step-6-model-training--evaluation)
    - [Step 7: Analysis & Insights](#step-7-analysis--insights)
    - [Step 8: Sentiment Prediction on New Data](#step-8-sentiment-prediction-on-new-data)
    - [Step 9: Reporting & Visualization](#step-9-reporting--visualization)
3. [Conclusion](#conclusion)
4. [References](#references)

## Introduction

Understanding customer sentiment is crucial for making data-driven business decisions. Manual analysis of large text corpora is unfeasible, so this project uses machine learning and deep learning techniques to automate sentiment classification.

## Project Workflow

### Step 1: Environment Setup

- Use **Google Colab** for simple setup and GPU support.  
- Required Libraries:  
  - `pandas`, `numpy` for data handling  
  - `sklearn` for machine learning tools  
  - `nltk` for NLP tasks  
  - `keras`, `tensorflow` for deep learning  
  - `matplotlib`, `seaborn` for visualizations

### Step 2: Exploratory Data Analysis (EDA)

- **Inspect the Data:** Check for missing/duplicate values and confirm data integrity.
- **Review Samples:** Examine random reviews to assess their content and structure.
- **Class Distribution:** Plot the balance between positive and negative reviews.
- **Text Statistics:** Analyze review length, vocabulary size, and frequent terms for insights into dataset composition.

### Step 3: Data Preprocessing

Prepare the raw text for modeling:
- **Lowercase Conversion:** Ensures uniformity.
- **Remove Punctuation/Special Characters:** Cleans the text for better tokenization.
- **Tokenization:** Break reviews into words.
- **Stopword Removal:** Eliminates common, non-informative words.
- **Stemming/Lemmatization:** Reduces words to their root forms.

### Step 4: Feature Engineering & Vectorization

Transform text into a machine-understandable format:
- **Bag of Words (BoW):** Counts word occurrences.
- **TF-IDF:** Assigns weights to words based on their importance.
- **Word Embeddings:** (Optional for advanced modeling) Uses pre-trained models like Word2Vec or GloVe to capture semantic meaning.

### Step 5: Model Selection and Building

Test and compare:
- **Classical Machine Learning Models:** Logistic Regression, Naive Bayes, Support Vector Machines, etc.
- **Deep Learning Models:** 
    - **LSTM (Long Short-Term Memory):** For sequential analysis of text.
    - **CNN (Convolutional Neural Network):** For extracting local features and n-gram patterns.
- Models use `sigmoid` output for binary classification and are optimized using algorithms like `Adam`.

### Step 6: Model Training & Evaluation

- **Splitting Data:** Train-test split to ensure unbiased evaluation.
- **Metrics Tracked:** Accuracy, precision, recall, F1-score.
- **Confusion Matrix:** Visualizes model predictions vs. ground truth.
- **Training Curves:** Assess overfitting or underfitting.

### Step 7: Analysis & Insights

- **Compare Models:** Highlight the best-performing approach.
- **Inspect Errors:** Review misclassified samples to improve understanding.
- **Most Influential Words:** Visualize via word clouds or feature importance.

### Step 8: Sentiment Prediction on New Data

Deploy the trained model for real-world use:
- **Pipeline:** New reviews undergo identical preprocessing, vectorization, and prediction steps.
- **Output:** Classifies new input reviews as positive or negative sentiment.

### Step 9: Reporting & Visualization

Communicate findings:
- **Graphs:** E.g., accuracy/loss curves, confusion matrices, word clouds.
- **Summary Reports:** Present model strengths, limitations, and overall performance to stakeholders.

## Conclusion

This project delivers a practical and comprehensive sentiment analysis pipeline, applicable not only to movie reviews but any domain involving customer feedback. By automating text classification, organizations can quickly respond to customer attitudes, spot trends, and improve offerings through actionable analytics.

## References

- IMDB Dataset: [Kaggle - IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- Tools: Python, scikit-learn, NLTK, TensorFlow/Keras, Google Colab  
- Example Guides: Machine learning & NLP tutorials

## Project Code and Notebook

The code accompanying this project, including all of the steps above, can be found in the Google Colab notebook (provide your own link if making public).

**Feel free to fork, contribute, or raise issues in the repository for improvements!**
## OUTPUT
## Model Evaluation Results

### 1. Classification Report

The classification report reflects the effectiveness of the model in predicting positive and negative sentiments, reporting high precision, recall, and F1-score.

<img width="590" height="160" alt="Image" src="https://github.com/user-attachments/assets/0a2df31e-0692-4d4f-8615-06ee6a8b512a" />

## 2. Confusion Matrix

The confusion matrix provides a visual representation of correct and incorrect predictions, showing how many positive and negative reviews were classified correctly or incorrectly.

<img width="741" height="474" alt="Image" src="https://github.com/user-attachments/assets/95d183ff-886c-4d36-914f-f4da3813ef38" />
