# Fake Job Posting Detector

A machine learning model that flags fraudulent job postings from their text.

[Open in Colab](https://colab.research.google.com/github/somyadev725-create/fake-job-detector-/blob/main/fake_job_detector.ipynb)

## Dataset
[Real / Fake Job Posting Prediction (EMSCAD)](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction) by Shivam Bansal on Kaggle: 17,880 job postings, of which 866 (about 5%) are fraudulent.

## Method
1. Combined title, company profile, description, requirements and benefits into one text field
2. Split the data 80/20 with stratification to keep the fake-posting ratio
3. TF-IDF features (unigrams and bigrams, 50,000 features)
4. Logistic Regression with `class_weight='balanced'` to handle the class imbalance

## Results (test set, fraudulent class)
| Metric | Score |
|--------|-------|
| Precision | 0.81 |
| Recall | 0.91 |
| F1-score | 0.86 |

The model caught 157 of 173 fake postings, with 36 false alarms among 3,403 real postings.

Accuracy is about 99%, but it is not a good measure here because the data is imbalanced, so the metrics above are the ones that matter.

## How to run
1. Download the dataset from the Kaggle link above (it downloads as `archive.zip`)
2. Open the notebook in Colab using the badge above
3. Upload `archive.zip` when the upload cell asks for it, then run all cells

## Future work
- Compare with Linear SVM and Random Forest
- Add structured features such as `has_company_logo` and `telecommuting`
- Build a Streamlit app where users paste a job post and get a verdict
