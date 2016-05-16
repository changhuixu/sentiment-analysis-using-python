# sentiment-analysis-using-python
--- Large Data Analysis Course Project ---

This folder is a set of simplified python codes which use sklearn package 
to classify movie reviews.

Two classifiers were used: Naive Bayes and SVM.
SVM gives an accuracy of about 87.5%, which is slightly higher than 86% given by Naive Bayes.

## usage
`imdbReviews.py` generates `*.pkl` files which are the training and testing datasets.
First, set the dataset directory in the `imdbReviews.py`, then run the code.
```bash
python imdbReviews.py
```

You will get two `*.pkl` files which are needed for `naive.py` and `svm.py`.
To do prediction, run the following command.
```bash
python naive.py
```

## screenshots
![L1_1](https://github.com/changhuixu/sentiment-analysis-using-python/blob/master/naive/naive-demo-screenshot.png "Naive Bayes code")


![L1_2](https://github.com/changhuixu/sentiment-analysis-using-python/blob/master/svm/svm-demo-screenshot.png "SVM code")

### End
last modified 04/16/2016
