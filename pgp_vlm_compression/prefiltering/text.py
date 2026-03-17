import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np


def preproc_prompt_clip(prompt: str, strict_mode=False, pre_text_summ=True) -> str:
    # removing the instructions
    phrases_to_remove = [
        " Please answer yes or no.",
        " Answer the question using a single word or phrase.",
        "\nAnswer with the option's letter from the given choices directly.",
        "Please select the correct answer from the options above. \n",
        "Options:\n",
        "Question: ",
        "Hint: ",
        "\nAnswer the question directly.",
        ".", "?", ",",
    ]
    for phrase in phrases_to_remove:
        prompt = prompt.replace(phrase, "")

    prompt = prompt.strip()
    prompt = clean_text(prompt, strict_mode=strict_mode, pre_text_summ=pre_text_summ)

    return prompt.strip()


nltk.download("punkt")                          # tokenizer
nltk.download('punkt_tab')                      # tokenizer
nltk.download("averaged_perceptron_tagger")     # POS tagging
nltk.download("averaged_perceptron_tagger_eng") # POS tagging
nltk.download("wordnet")                        # lemmatizer
nltk.download("maxent_ne_chunker")              # NER chunker
nltk.download("maxent_ne_chunker_tab")          # NER chunker
nltk.download("words")                          # wordlist (NER support)
nltk.download("stopwords")                      # stopwords

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def get_order(tag):
    if tag.startswith('NN'):  # nouns
        return 1
    elif tag.startswith('JJ'):  # adjectives
        return 2
    elif tag.startswith('VB'):  # verbs
        return 3
    else:
        return 4  # least important

def clean_text(text, context_length=72, strict_mode=False, pre_text_summ=True):
    tokens = nltk.word_tokenize(text)
    if pre_text_summ:
        tokens = [t for t in tokens if t.lower() not in stop_words]
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    pos_tags = nltk.pos_tag(tokens)

    # Named Entity Recognition (NER) to force-keep entities
    chunked = nltk.ne_chunk(pos_tags, binary=False)
    ner_tokens = set()
    for subtree in chunked.subtrees():
        if subtree.label() in ["PERSON", "GPE", "ORGANIZATION", "LOCATION"]:
            for leaf in subtree.leaves():
                ner_tokens.add(leaf[0])

    # Priority ordering
    order_list = [get_order(tag) for _, tag in pos_tags]
    if strict_mode:
        # keep only priority < 4 or NERs
        filtered_ids = [
            i for i, order in enumerate(order_list)
            if order < 4 or pos_tags[i][0] in ner_tokens
        ]
        # now enforce context_length if still too long
        sorted_ids = sorted(
            filtered_ids,
            key=lambda i: order_list[i]
        )
        keep_ids = sorted(sorted_ids[:context_length - 2])
    else:
        # normal mode: fill by priority until context_length
        sorted_ids = np.argsort(np.array(order_list))
        keep_ids = sorted(sorted_ids[:context_length - 2])

    # Keep original order in final tokens
    sampled_tokens = np.take(np.array(tokens), keep_ids, axis=0)

    return " ".join(sampled_tokens)
