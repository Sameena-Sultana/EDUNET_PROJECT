# EDUNET_PROJECT
#  Fake News Detection using AI & NLP

This project focuses on building a machine learning model that can **detect whether a news article is fake or real** based on its title and content.

---

## 📌 What This Project Does

- Loads and combines real (`True.csv`) and fake (`Fake.csv`) news data
- Cleans the news text (lowercasing, removing punctuation)
- Extracts text features using **TF-IDF Vectorizer**
- Trains a **Multinomial Naive Bayes AI model**
- Evaluates the model’s performance using accuracy and classification metrics
- Visualizes results using **word clouds**, **bar charts**, and a **confusion matrix**

---

##  What AI Is Used?

### ✔️ Multinomial Naive Bayes
- A **supervised machine learning** algorithm based on **Bayes Theorem**
- Great for **text classification tasks** like spam detection and fake news
- Assumes each word contributes independently to the final classification (Naive)
- It’s lightweight and fast, ideal for early-stage NLP tasks

### ✔️ TF-IDF (Term Frequency-Inverse Document Frequency)
- An **NLP feature extraction technique**
- Converts text into numeric vectors based on word importance
- Gives higher weight to words that are unique in an article but less common overall

Together, **TF-IDF + Naive Bayes** forms a classic AI pipeline for document classification.

---

## 🔍 Requirements

- Python 3.x
- Jupyter Notebook
- Libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, wordcloud

```bash
pip install pandas numpy scikit-learn matplotlib seaborn wordcloud
