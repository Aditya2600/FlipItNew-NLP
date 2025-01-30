# News Classification using NLP

## Introduction
FlipItNews, a Gurugram-based company, aims to revolutionize financial literacy in India by leveraging AI and ML for smart content discovery and contextual engagement. This project focuses on classifying news articles into various categories like Politics, Technology, Sports, Business, and Entertainment using Natural Language Processing (NLP).

## Objective
The objective of this project is to extract news articles from FlipItNews’s internal database and categorize them using NLP techniques. We will explore and compare at least three different machine learning models for this classification task.

## Approach
The project follows these steps:

### 1. Importing Libraries
Required libraries such as Pandas, NumPy, Sklearn, NLTK, and TensorFlow are imported.

### 2. Loading the Dataset
The dataset is loaded from FlipItNews’s internal database and stored for processing.

### 3. Data Exploration
- Checking the shape of the dataset
- Visualizing the distribution of news articles across categories

### 4. Text Processing
- **Removing Non-Letter Characters**: Cleaning text data
- **Tokenization**: Splitting text into individual words
- **Removing Stopwords**: Eliminating common words like "the" and "is"
- **Lemmatization**: Reducing words to their base forms

### 5. Data Transformation
- **Encoding the Target Variable**: Converting categorical labels into numerical form
- **Feature Extraction**:
  - Bag of Words (BoW)
  - Term Frequency-Inverse Document Frequency (TF-IDF)
- **Train-Test Split**: Splitting data into training and testing sets

### 6. Model Training & Evaluation
We implement and compare the following models:

#### 6.1 Simple Approach
- **Naïve Bayes Classifier**: A probabilistic classifier based on Bayes' theorem
- **Decision Tree Classifier**: A tree-based model for classification
- **K-Nearest Neighbors (KNN)**: A distance-based classification method
- **Random Forest Classifier**: An ensemble learning method using multiple decision trees

### 7. Functionalized Code
Code is modularized for better reusability and efficiency.

## Results & Comparisons
Each model is evaluated based on accuracy, precision, recall, and F1-score to determine the best-performing classifier for news categorization.

## Installation & Usage
### Prerequisites
Ensure you have the following libraries installed:
```bash
pip install pandas numpy sklearn nltk tensorflow
