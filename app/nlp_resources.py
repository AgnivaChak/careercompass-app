import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


FALLBACK_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "can", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off",
    "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out",
    "over", "own", "s", "same", "she", "should", "so", "some", "such", "t",
    "than", "that", "the", "their", "theirs", "them", "themselves", "then",
    "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "you", "your",
    "yours", "yourself", "yourselves"
}


def _resource_available(path):
    try:
        nltk.data.find(path)
        return True
    except LookupError:
        return False


def get_stop_words():
    if _resource_available("corpora/stopwords"):
        try:
            return set(stopwords.words("english"))
        except LookupError:
            pass
    return FALLBACK_STOP_WORDS


def tokenize_words(text):
    if _resource_available("tokenizers/punkt"):
        try:
            return word_tokenize(text)
        except LookupError:
            pass
    return re.findall(r"[a-zA-Z][a-zA-Z0-9+#.\-]*", text)


def lemmatize_words(words):
    if not _resource_available("corpora/wordnet"):
        return words

    lemmatizer = WordNetLemmatizer()
    try:
        return [lemmatizer.lemmatize(word) for word in words]
    except LookupError:
        return words
