import re
from sklearn.feature_extraction.text import TfidfVectorizer
import PyPDF2
from app.nlp_resources import get_stop_words, tokenize_words

stop_words = get_stop_words()

def extract_text_from_pdf(pdf_path):
    """Extract plain text from PDF."""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + " "
    return text

def extract_basic_info(text):
    """Extract basic info like email and phone from resume."""
    email_match = re.search(r'[\w\.-]+@[\w\.-]+', text)
    phone_match = re.search(r'\b\d{10}\b', text)

    return {
        "email": email_match.group() if email_match else "Not Found",
        "phone": phone_match.group() if phone_match else "Not Found"
    }

def extract_resume_keywords(text, return_vector=False):
    tokens = tokenize_words(text.lower())
    filtered = [word for word in tokens if word.isalpha() and word not in stop_words]
    
    keyword_string = " ".join(filtered)  # this is what we give to TF-IDF

    if return_vector:
        vectorizer = TfidfVectorizer()
        vector = vectorizer.fit_transform([keyword_string]).toarray()[0]
        return filtered, vector

    return filtered
