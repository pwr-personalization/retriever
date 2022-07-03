# PerSent 
Personalized sentiment text classification

__Authors__: Michał Kajstura, Joanna Baran

__Mentors__: professor Przemysław Kazienko, PhD. Jan Kocoń

The project was focused on the scientific view only. 
The main aim was to develop a new method to human-centric classification problem
involving a highly subjective task.

## Short description
Generalizing models are often unable to produce a personalized prediction for a individual user.
Personalized methods often yield better results
but at the cost of an increased training complexity,
especially in big-data scenarios, when frequent re-training is necessary.
We propose a __retrieval__ based approach that performs similar to SOTA personalization algorithms,
but does not require any further optimization for new users.

## Methodology
Retriever is a method combining text representations obtained from a language model, like RoBERTa,
and an aggregated user-level score computed by a retrieval module.
Texts previously written or annotated by the user are retrieved from the database.
Then the text similarity scores are computed and used to calculate a user score representing their preferences.
There are various ways of aggregating multiple labels into a single score. 
In the experiments, we used a simple mean, weighted average and 
a KNN based aggregation which averages the labels of the K most similar samples. Textual features are concatenated with a user score.
This personalized representation is then passed to a linear classifier for a person-informed prediction.



### Text similarity
Text similarity scores influence the aggregation of previous users' text labels.
The labels of samples most similar to the current text have the greatest impact on the final user score.
Conversely, labels of unrelated texts should be discarded in the aggregation process.

The simplest method of aggregating multiple targets is a weighted arithmetic mean.
Similarity score $s$ between a pair of texts, play a role of weighting coefficients.
If, therefore, another text is very similar to the sample being evaluated, it will have a weight close to 1, 
and the weight of the differing text will be close to 0.

$$s(t_i, T, L) = 
    \frac{1}{N - 1} 
    \sum_{n=1}^{N}
    \1_{n \neq i} \cdot l_n \cdot \text{similarity($t_i$, $t_n$)}
$$

## Datasets
In this work only open-source freely available data was used.
- [Sentiment140](http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip) - a heuristically annotated dataset of tweets that was collected using a predefined set of queries.
The annotations are based on emoticons contained in the texts

- [IMDB](https://drive.google.com/drive/folders/1do6c_kXC4abMgEhbeJ7ObcEdqm78ghAy?usp=sharing) - a collection of film reviews from the popular IMDB website. 
The user reviews were divided into training, validation and test in the ratio of 0.8, 0.1, 0.1, 
meaning that a single user appears in all splits.

- [Measuring Hate Speech](https://hatespeech.berkeley.edu/) - dataset that offers multiple dimensions like sentiment analysis, insults,
incitement to violence and hate speech detection.
We focused on the latter task because it proved difficult for generalizing, or non-personalized, methods.

## Experiments
### Setup
RoBERTa-base language model was used both as a baseline and in personalized approaches.
For text similarity calculations, we utilized MPNet based _Bi-Encoder_ trained on various 
text-pair datasets.
The _Cross-Encoder_ was based on a RoBERTa trained on Semantic Text Similarity Benchmark.
As other personalized approach we chose [UserIdentifier](https://arxiv.org/abs/2110.00135) 
which focuses on learning user-specific vector representations.

Fine-tuning hyperparameters:
- AdamW optimizer 
- learning rate $1e^{-5}$ with linear warmup schedule
- batch size 16
- max sequence length 512 
- 50000 training steps
The best model was selected according to the validation F1-score.
For the KNN-based aggregation, K was set to 3.
Experiments were repeated 5 times and the mean F1-score was reported.

In order to analyze the performance of the personalized approach,
we created __separate subsets__ of the _Sentiment140_ dataset.
The first subset contained only _highly polarized_ users,
that is, users for whom the fraction of positive or negative tweets was above $70\%$.
We noticed that for many users, the similarities between texts were almost uniform,
resulting in a poor performance of the proposed model.
If the two texts are almost identical and differ in labels, the retriever module cannot fetch relevant tweets.
The second subset was formed from users who wrote _different texts_ and aims to highlight the advantages of
the proposed method,
as it uses the content-aware aggregation of user's scores based on text similarity.
The text diversity was measured as a standard deviation of cosine similarity in a user's text similarity matrix.
Individuals with a diversity above 0.5 were considered to be writing diverse tweets.

### Results
|                     | Sentiment140 | IMDB     | MHS      |
|---------------------|--------------|----------|----------|
| Baseline            | 85.9.        | 46.7     | 56.3     |
| UserIdentifier      | __87.9__     | __50.4__ | 56.1     |
| Retriever-Mean      | 86.3         | 47.7     | 56.6     |
| Retriever-Bi        | 87.1         | 48.8     | 56.7     |
| Retriever-Cross     | 87.7         | 49.4     | __57.2__ |
| Retriever-Cross-KNN | 87.4         | 49.8     | 56.6     |

The UserIdentifier outperformed other methods for _Sentiment140_ and _IMDB_ datasets but
_Retriever-Cross_'s performance is almost the same without the need to re-train for each new user. 
For the Measuring Hate Speech dataset, UserIdentifier was worse than the standard generalizing RoBERTa model,
while both Retriever variants performed better.
A simple __Retriever-Mean__ aggregation method that uses a basic arithmetic mean to calculate the user score,
offered a slight improvement over the baseline, but performed worse than other alternatives.
__Retriever-Cross__ achieved better results, but at the significantly higher computational cost,
because it required performing expensive forward passes through the network for each text pair.
The __KNN-based__ aggregation performed similarly to the standard weighted average.


| Subset    | Retriever-Mean | Retriever-Cross |
|-----------|----------------|-----------------|
| All       | 86.3           | 87.7            |
| Polarized | 90.6           | 90.5            |
| Diverse   | 87.0           | 89.3            |

The performance of __Retriever-Cross__ model for both _Polarized_ and _Diverse_ data subsets 
was considerably higher than for the standard Sentiment140.
The High F1-score for _Polarized_ split could be attributed to the fact that it consists of users expressing 
opinions with similar sentiment.
Personalized approaches can use the labels of the previously written posts,
so it naturally excels when the user is single-sided.
For the _Diverse_ subset, the text retrieval module was more useful,
because a certain small number of text pairs were more closely related than the rest.
As a result, the aggregated user's score better reflected the individual's preference for the currently 
evaluated text.
