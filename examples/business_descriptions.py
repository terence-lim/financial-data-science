"""
or only keep nouns, since they have the most information:

doc_nouns = nlp(' '.join([str(t) for t in doc if t.pos_ in ['NOUN', 'PROPN']]))
"""
