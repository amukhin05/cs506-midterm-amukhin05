[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Lqxbrt0o)
# CS506 Midterm

## IMPORTANT NOTE: I completed this assignment with 2 submissions, as I ran most of my tests locally. Both submitted iterations are separate documents.

## <a id="explore">Data Exploration</a>

Before even attempting to build any model, I started by looking over the contents of the data. The first thing I noticed was that there were a lot more 5's than any other score. This meant that my model should attempt to be extremely good at predicting a score of 5, as simply predicting all 5's already gave a benchmark score within the 10% benchmark. There was not that much other data to explore, other than noticing that 'UserID' and 'ProductID' were not unique. 

Second Submission: All of this was done prior to the first submission. Did not alter this section for the second submission.

## <a id="feature">Feature Extraction</a>

First, I started with intuitive tasks. Since 'UserID' and 'ProductID' were not unique, I decided to combine them into two new features: 'Prod_Count' and 'User_Count', by simply grouping the same IDs together. 

Next, I created a temporary feature that was aimed at helping with further text analysis, called All_Text. This simply contained both the summary and the text combined.

From this, I extracted 2 direct features and 3 sentiment features. 
- The direct features were 'Exclamations' and 'Questions', which counted the total number of exclamation points and question marks, respectively, in each review.
- The sentiment features were 'Negation_Count', 'Positive_Count', and 'Sentiment': 
  - 'Negation_Count' used the [opinion_lexicon](https://www.nltk.org/_modules/nltk/corpus/reader/opinion_lexicon.html) library to count the total number of words in a review that were also contained in opinion_lexicon.negative().
  - 'Positive_Count' did the exact same, but instead counted words that were also in opinion_lexicon.positive(). I found this library via a Google search, and the only way I used it was to load a list of negative and positive words, to avoid creating such myself.
  - 'Sentiment' came from a different library, called [Textblob](https://textblob.readthedocs.io/en/dev/). This is one of the libraries that our group originally reviewed for our group final project, but ended up rejecting as it was very simple and slightly underperforming compared to more complex models. However, complex models would take an exorbitant amount of time if run on such a large dataset, so I opted to use the simpler TextBlob. As can be seen in the code, this library had a very direct implementation of sentiment analysis using TextBlob(x).sentiment.polarity.
 
Finally, the last and most important feature(s) that I added was TF-IDF. This was covered partially in lab, but I will elaborate more on the implementation side of this process. I used [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) from scikit-learn, a library used extensively in labs. At first, I created a tfidf_matrix from the 'All_Text' column, then I converted it into a dataframe, indexed it based on the current df, and merged the two using pd.concat. The parameters I used for TF-IDF were: max_features=1000, ngram_range=(1, 2), stop_words='english'(default). max_features=1000 made TF-IDF select the 1000 most common words, and ngram_range=(1, 2) allowed one and two-word pairs to be considered. 

Note that TF-IDF takes quite a long time to run, so for convenience purposes, if an X_train file already exists, it is simply loaded (instead of recalculating all of the features from scratch). If it does not exist, then the proper files are created. 

Second Submission: Changed the TF-IDF max_features from 1000 to 5000

## <a id="model_make">Model Creation, Assumptions, and Tuning</a>

There were three more preprocessing steps that I completed before diving into creating a model: 
- First, I sampled only 100,000 rows, as working with the entire dataframe proved to be extremely challenging and time-consuming for my machine.
- Then, I created a train-test-split on my data, which I did in the ratio of 1/4.0. This was already provided, but I will still link the library used at [train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html) from scikit-learn
- Finally, I removed the unnecessary features that could not be used for modeling, such as non-float columns, the ID column, and the Score column.
All of these steps were quite intuitive, as the model simply could not run without them (or run efficiently without sampling).

Second Submission: removed sampling and ran the model on the entire dataset.

Finding the correct model: 
Throughout the process of finding a model, I tried numerous different models, which I will only briefly list: GaussianNB, MultinomialNB, RandomForest, SVM, and more
I also tried numerous dimensionality reduction techniques, briefly listed: NMF, LSA, LDA

However, the model+reduction pairing that seemed to work by far the best was TruncatedSVD + LogisticRegression. This also included using a scaler. I will explain each of these approaches one by one:
- StandardScaler - scales all the columns to a similar scale. This prevents overpredicting on columns with a scale much greater than the others, and is very useful for dealing with the count columns, as they range significantly more than all of the TF-IDF columns. No parameters needed. This was covered in labs, so no documentation.
- TruncatedSVD - SVD was covered in class, and [TruncatedSVD](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html) is a very close variant of it which comes, once again, from the scikit-learn library. The only parameter that I used for this was n_components, which I initially set to 1000. 
- LogisticRegression - this was covered in class and extensively in other math courses that I have previously taken, so implementing this model did not require any Google searches from me. The only parameter that I used for this was max_iter, which I initially set to 100,000.
After all of this, I pipelined the model using [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) from scikit-learn. This simply allowed the model to run/apply all three of these things in order. The results of this particular model were my first submission.

Second Submission: In this submission, I found that the best SVD value was around 350 (otherwise, the model overpredicted), and a higher number of iterations in logistic regression seemed to not yield much benefit, so the new parameters I used were n_components=350, max_iter=100000. Running this modified model on a larger TF-IDF set and on the whole dataset improved the accuracy score by about 1.5% locally, but took very long to run. 

Note: After creating the submission model, I was interested in seeing whether gradient boosting would actually boost the performance of this model. Surprisingly, after running the HistGradientBoostingClassifier for about 90 minutes, it yielded results similar to my first submission, and significantly worse than my second submission! This was pleasantly surprising to me, as it showed that the model I created was better than this particular gradient boosting, and also that I had tuned the features correctly to match my model. 

## <a id="model_eval">Model Evaluation / Performance</a>

With the final model that I stopped on described above, I ended up getting a local test result of about 0.639. This came after a lot of tuning and model selection, but the confusion matrix still did not look as good as I would want it to look. The best-looking confusion matrix (despite yielding rather poor results overall) came when I utilized stratified sampling with the same final model. There was a very clear trend (unlike the confusion matrix of the submission), but unfortunately, the model could not distinguish well enough between neighboring values, resulting in a poor accuracy score. 

Note: I did almost nothing to this section, as it was mostly given, except for normalizing the confusion matrix.

Runtime note: my final submission ended up taking hours to run locally, and TF-IDF ran out of memory on my laptop, so I had to run it on a more powerful PC with 64GB of ram, which seemed to give enough memory for TF-IDF to run. 

## <a id="issues">Struggles / Issues / Open Questions</a>

The main issue that I ran into with this midterm was memory allocation. It seemed that almost every method I tried to run at scale would crash the kernel due to trying to allocate more memory than my computer had (in extreme cases reaching up to 9.03TB of missing memory). This meant that a lot of my midterm was spent tying to perfect the model on the very harsh memory constraints of my computer. The other struggle was finding a model that worked accurately, as logistic regression was one of the last models I tried. Finally, I found the manual feature selection work to be quite tough, since it seemed that the more I engineered my code, the worse the results got. 

Overall, I am very glad that I ended up making a model that could create an accuracy score well above the minimal threshold for the 10/10 score. 

Note: my last and final issue was that Gradescope could not find the original repo. As such, I have copied the original repo exactly into the new repo that has been submitted. 


