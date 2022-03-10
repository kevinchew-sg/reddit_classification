# Project 3: Web APIs & NLP

## Introduction

In this project, we aim to collect posts from two subreddits of our choosing using Pushshift's API. Then, we will use NLP (Natural Language Processing) to train a classifier on which subreddit a given post came from.

## Problem Statement

We aim to answer whether or not we can build a classification model to accurately predict if a post belongs to one subredit or the other. Should we be able to, we next seek to determine which model works best and why.

The two subreddits chosen are

[PremierLeague](https://www.reddit.com/r/PremierLeague/): This subreddit is where users can post information regarding the English Premier League, which is the top level of the English football league system. Contested by 20 clubs, it operates on a system of promotion and relegation with the English Football League.

[nba](https://www.reddit.com/r/nba/): This subreddit is where users can post information regarding the NBA (National Basketball Association). The NBA is a professional basketball league in North America. The league is composed of 30 teams and is the premier men's professional basketball league in the world.

## Outline

* **Section 1: Problem Statement**
    * We define our proble statement here
* **Section 2: Data Collection**
    * We perform webscrapping using Pushshift's API to collect data from the 2 chosen subreddits
* **Section 3: Preprocessing (Data Cleaning & EDA)**
    * We do preliminary data cleaning and preprocessing to clean the text. We then do EDA (Exploratory Data Cleaning) to see what insights we can get of our dataset
* **Section 4: Modeling the Data**
    * The models that we have chosen are Logistic Regression with CountVectorizer, Logistic Regression with TF-IDF Vectorizer, Random Forest with CountVectorizer, Random Forest with TF-IDF Vectorizer, Naive Bayes with CountVectorizer and Naive Bayes with TF-IDF Vectorizer
    * We initiate a pipeline and conduct GridSearchCV for hyperparameter tuning. We put the models' results into a table and select which models to use for further evaluation
* **Section 5: Evaluation of our Models**
    * We take a closer look at our model results, including top predictors and false postives and false negatives
* **Section 6: Conclusion**
    * A summary of our findings is shared and we select the best model to use



## Models Results

|Model|Accuracy Score|Training Score|Test Score|Overfitting|Cross_val_score|
|---|---|---|---|---|---|
|**Logistic Regression with Count Vectorizer**|0.954|0.985|0.958|0.027|0.951|
|**Logistic Regression with TF-IDF Vectorizer**|0.956|0.983|0.961|0.022|0.956|
|**Random Forest with Count Vectorizer**|0.722|0.684|0.682|0.002|0.718|
|**Random Forest with TF-IDF Vectorizer**|0.725|0.726|0.723|0.003|0.729|
|**Naives Bayes with Count Vectorizer**|0.960|0.975|0.955|0.02|0.960|
|**Naives Bayes with TF-IDF Vectorizer**|0.956|0.972|0.951|0.021|0.954|


## Model Findings

* Random Forest performs poorly
* Use TF-IDF Vectorizer instead of Count Vectorizer
* Logistic Regression and Naive Bayes achieved similar results, though Logistic Regression performs slightly better in Training Score and Testing Score

## Conclusion

Circling back to our problem statement, we aim to answer whether or not we can build a classification model to accurately predict if a post belongs to one subredit or the other. Then, should we be able to, we next seek to determine which model works best and why.

From the results of our Logistic Regression and Naive Bayes model, we can confidently say that they are able to accurately classify the posts, with both having an accuracy score of ~0.96. The Random Foreat model, however, performed poorly with an accuracy score of ~0.70 and we dropped that model for further evaluation.

Now that we answered the first part of the problem statement, we move on to the next, which is what model works best and why.

In terms of which vectorizer to use, TF-IDF Vectorizer would be our choice instead of Count Vectorizer. Apart from Naive Bayes, in which the results were more or less the same, the model results are better when it is run in conjunction with TF-IDF. This is to be expected as TF-IDF is a statistical measure said to have fixed the issues with CountVectorizer. In simple terms, it helps us to tease out the relevance of words instead of just using word count in the analysis.

Now then, should we use Naive Bayes with TF-IDF Vectorizer, or Logistic Regression with TF-IDF Vectorizer?

From just looking at the model results, they have very similar results across all the metrics that we had used, including overfitting. The only noticeable difference, albeit it being minute, is that Logistic Regression did better in the Training and Test Score.

Let us now look at the difference between the Naive Bayes and Logistic Regression models.

Naïve Bayes is a classification method based on Bayes’ theorem that derives the probability of the given feature vector being associated with a label. Naïve Bayes has a naive assumption of conditional independence for every feature, which means that the algorithm expects the features to be independent which not always is the case.
Logistic regression, on the other hand, learns the probability of a sample belonging to a certain class. Logistic regression tries to find the optimal decision boundary that best separates the classes. [Source: Comparison between Naive Bayes and Logistic Regression](https://dataespresso.com/en/2017/10/24/comparison-between-naive-bayes-and-logistic-regression/)

Thus, one clear difference is that, Naive Bayes makes a conditional independence assumption, which is violated when you have correlated/repetive features. 

Given this, the output for Logistic Regression gives us more context and given that some words might be correlated to others, the top words for Logistic Regression are the ones that are the best predictors with the context of other words. 

Also, in a paper written by Professor Andrew Ng and Professor Michael I Jordan, which provides a mathematical proof of error properties of both the logistic regression and naive bayes, they concluded that when the training size reaches infinity, the discriminative model (Logistic Regression) performs better than the generative model (Naive Bayes). [Source](http://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)
Thus, with a larger dataset, for example, increase the data interval from 1 year to 2 years, we could expect the Logistic Regression model to score even better.

With the above in mind, we conclude that Logistic Regression with TF-IDF Vectorizer works best.






