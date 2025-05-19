# Fake News Detection using NLP in Jupyter Notebook
This project aims to detect fake news articles using Natural Language Processing (NLP) and Machine Learning techniques within a Jupyter Notebook environment. The model is trained to classify news as FAKE or REAL based on the article content.

## Environment
Jupyter Notebook (Recommended)

## Technologies Used
Pandas – data handling

NumPy – numerical operations

Matplotlib / Seaborn – data visualization

NLTK – text preprocessing

Scikit-learn – machine learning tools


##  Dataset

This project uses the **Fake and Real News Dataset** from Kaggle.

##### [Click here to view/download the dataset on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

After downloading, extract the zip and place the `Fake.csv` file in your project directory.

**The dataset used is Fake.csv, which contains**:

title – Headline of the news article

text – Full content of the article

subject – Category or topic (e.g., News, Politics)

date – Date of publication

(Assumption: Dataset includes a label column with FAKE or REAL values)

## How to Use (Jupyter Notebook)
Open your terminal or Anaconda Prompt and launch Jupyter Notebook:

bash
Copy
Edit
jupyter notebook
Open the notebook file:
fake_news_detection.ipynb

Run each cell step-by-step:

Import libraries

Load dataset

Preprocess text

Vectorize text using TF-IDF

Train model

Evaluate results

##Output
Accuracy Score of the model

Classification Report with precision, recall, F1-score

Confusion Matrix Heatmap for visual evaluation

## Steps Followed in Notebook
Importing Libraries

Loading the Dataset

Text Cleaning – Lowercasing, removing punctuation and stopwords

TF-IDF Vectorization

Model Training – Logistic Regression

Evaluation – Accuracy, Confusion Matrix, Report

##Future Scope
Experiment with other models (SVM, Naive Bayes, Random Forest)

Add LSTM or Transformer-based Deep Learning models

Build a web interface using Flask or Streamlit for live news testing

##Folder Structure
python
Copy
Edit
Fake-News-Detection/
│
├── fake_news_detection.ipynb
├── Fake.csv
├── README.md

## Sample Output

Accuracy: 0.985
Classification Report:
              precision    recall  f1-score   support

       Real       0.98      0.99      0.99       780
       Fake       0.99      0.98      0.98       776

    accuracy                           0.98      1556
   macro avg       0.98      0.98      0.98      1556
weighted avg       0.98      0.98      0.98      1556
Confusion Matrix:

Predicted Real	Predicted Fake
Real	770	10
Fake	14	762

#### Output Visualization
A heatmap showing the confusion matrix:

X-axis: Predicted labels

Y-axis: True labels

Annotated values

