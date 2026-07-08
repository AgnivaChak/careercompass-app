from sklearn.feature_extraction.text import TfidfVectorizer
from app.nlp_resources import get_stop_words, tokenize_words

stop_words = get_stop_words()

def extract_jd_keywords(text, return_vector=False):
    tokens = tokenize_words(text.lower())
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    keyword_string = " ".join(filtered)

    if return_vector:
        vectorizer = TfidfVectorizer()
        vector = vectorizer.fit_transform([keyword_string]).toarray()[0]
        return filtered, vector

    return filtered
