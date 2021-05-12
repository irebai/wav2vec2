#!/usr/bin/env python3
import re

# Create and save tokenizer
punctuation='[\,\?\.\!]'
chars_to_ignore_regex = '[\,\?\.\!\;\:\"\“\%\”\�\…\·\ǃ\«\‹\»\›“\”\ʿ\ʾ\„\∞\|\;\:\*\—\–\─\―\_\/\:\ː\;\=\«\»\→]'

def collapse_whitespace(text):
    _whitespace_re = re.compile(r'\s+')
    return re.sub(_whitespace_re, ' ', text).strip()

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub('â', 'â', text)
    text = re.sub('à','à',text)
    text = re.sub('á','á',text)
    text = re.sub('ã','à',text)
    text = re.sub('ҫ','ç',text)
    text = re.sub('ê','ê',text)
    text = re.sub('é','é',text)
    text = re.sub('è','è',text)
    text = re.sub('ô','ô',text)
    text = re.sub('û','û',text)
    text = re.sub('ù','ù',text)
    
    text = re.sub("’|´|′|ʼ|‘|ʻ|`", "'", text)
    text = re.sub("'+ ", " ", text)
    text = re.sub(" '+", " ", text)
    text = re.sub("'$", " ", text)
    text = re.sub("' ", " ", text)

    text = re.sub("−|‐", "-", text)
    text = re.sub(" -", "", text)
    text = re.sub("- ", "", text)
    text = re.sub("--+", " ", text)

    text = re.sub(chars_to_ignore_regex, ' ', text)

    text = re.sub(' m\. ', ' monsieur ', text)
    text = re.sub(' mme\. ', ' madame ', text)
    text = re.sub(' mmes\. ', ' mesdames ', text)
    text = re.sub('m\^\{me\}', ' madame ', text)
    text = re.sub('%', ' pourcent ', text)
    text = re.sub(' km ', ' kilomètres ', text)
    text = re.sub('€', ' euro ', text)
    text = re.sub('\$',' dollar ', text)
    text = re.sub('dix nov', 'dix novembre', text)
    text = re.sub('dix déc', 'dix décembre', text)
    text = re.sub('dix fév', 'dix février', text)
    text = re.sub('nº', 'numéro', text)
    text = re.sub('n°', 'numéro', text)
    text = re.sub('onzeº', 'onze degrés', text)
    text = re.sub('α','alpha',text)
    text = re.sub('β','beta',text)
    text = re.sub('γ','gamma',text)
    text = re.sub(' ℰ ',' e ',text)
    text = re.sub(' ℕ ',' n ',text)
    text = re.sub(' ℤ ',' z ',text)
    text = re.sub(' ℝ ',' r ',text)
    text = re.sub('ε','epsilon',text)
    text = re.sub(' ω | Ω ',' oméga ',text)
    text = re.sub(' υ ',' Upsilon ',text)
    text = re.sub(' τ ',' tau ',text)
    text = re.sub('σ|Σ','sigma',text)
    text = re.sub(' π | Π ',' pi ',text)
    text = re.sub(' ν ',' nu ',text)
    text = re.sub(' ζ ',' zeta ',text)
    text = re.sub('δ|Δ|∆',' delta ',text)
    text = re.sub(' ∈ ',' appartient à ',text)
    text = re.sub(' ∅ ',' ',text)
    text = re.sub('☉',' ',text)
    text = re.sub(' ≥ ','supérieur ou égale à',text)
    text = re.sub(' ½ ', ' demi ', text)
    text = re.sub('&', ' et ', text)
    text = re.sub("aujourd' +hui", "aujourd'hui", text)
    text = re.sub(" h' aloutsisme "," haloutsisme ", text)
    text = re.sub(" h' mông ", " hmông ", text)
    text = collapse_whitespace(text)
    return text
