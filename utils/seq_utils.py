import re
import unicodedata


# Converts the unicode file to ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,¿。，！＠@#＃￥$%％＆&*×（()）])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[?.!,¿。，！＠@#＃￥$%％＆&*×（()）]+", " ", w)

    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = w + ' <EOS>'
    return w


def max_length(tensor):
    return max(len(t) for t in tensor)


def process_result(lang, result):
    result_label = ""
    for i in result:
        if lang.idx2word[i] != '<EOS>':
            result_label += lang.idx2word[i] + " "
        else:
            return result_label
    return result_label
