import re
import spacy

# Load spaCy model once (fast config)
NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])

URL_RE = re.compile(r"http\S+|www\S+")
MENTION_RE = re.compile(r"@\w+")
WHITESPACE_RE = re.compile(r"\s+")

def normalize_basic(text: str) -> str:
    text = str(text).lower()
    text = URL_RE.sub("", text)
    text = MENTION_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text).strip()
    return text

def clean_doc(doc) -> str:
    tokens = []
    for t in doc:
        if t.is_stop or t.is_punct or t.is_space:
            continue
        if not t.is_alpha:
            continue
        lemma = t.lemma_.strip()
        if lemma:
            tokens.append(lemma)
    return " ".join(tokens)