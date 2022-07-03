from sentify.datasets.imdb import IMDBDataModule
from sentify.datasets.measuring_hate_speech import MHSHatespeechDataModule, MHSSentimentDataModule
from sentify.datasets.sentiment140 import Sentiment140DataModule
from sentify.datasets.wiki_detox import AttackWikiDataModule, ToxicityWikiDataModule, AggressionWikiDataModule
from sentify.datasets.yelp import YelpDataModule

DATASETS = {
    'sentiment140': Sentiment140DataModule,
    'imdb': IMDBDataModule,
    'yelp': YelpDataModule,
    'amazon_reviews': ...,
    'MHS_sentiment': MHSSentimentDataModule,
    'MHS_hatespeech': MHSHatespeechDataModule,
    'wiki_attack': AttackWikiDataModule,
    'wiki_toxicity': ToxicityWikiDataModule,
    'wiki_aggression': AggressionWikiDataModule,
    'offensive_language': ...,
    'psmm': ...,
}
