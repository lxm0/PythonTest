import re
from pattern.text.en import tag
from nltk.corpus import wordnet as wn

from python.test.Test import TypeMinor


def _split_var(self, name):
    if name in self.id2names:
        return self.id2names[name]
    final_names = []
    # try to remove tail digits first
    name = self._remove_tail_digits(name)
    names = map(self._remove_tail_digits, name.split('_'))
    for name in names:
        if self._check_if_all_capitals(name):
            final_names.append(name)
        else:
            subnames = self._split_by_captials(name)
            subnames = map(self._remove_tail_digits, subnames)
            subnames = map(str.lower, subnames)
            final_names.extend(subnames)
    final_names = filter(bool, final_names)
    self.id2names[name] = final_names
    return final_names

def penn_to_wn(tag):
    if is_adjective(tag):
        return "ADJ"
    elif is_noun(tag):
        return "NOUN"
    elif is_adverb(tag):
        return "ADV"
    elif is_verb(tag):
        return "VERB"
    return None
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']


def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']


def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']


def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']


def penn_to_wn(tag):
    if is_adjective(tag):
        return "ADJ"
    elif is_noun(tag):
        return "NOUN"
    elif is_adverb(tag):
        return "ADV"
    elif is_verb(tag):
        return "VERB"
    return None
def _extract_pos_feature(var):
    names = var
    tags = [t for _, t in get_pos(' '.join(names))]
    for t in tags:
        if t == "VERB":
            return 1
        if t == "NOUN":
            return 0
    else:
        return 2

def get_pos(text):
    return [(w, penn_to_wn(p)) for w, p in tag(text)]
if __name__ == '__main__':
    names = (" compu,cc ")
    # cont = _extract_pos_feature(names)
    # print(cont)
    # print(' '.join(names))
    # print(tag(names))
    # [(w, penn_to_wn(p)) for w, p in tag(names)]
    # matchs = re.match('^_+[\d_]*$', names)
    # print(matchs)
    # get_pos(' '.join(names))
    info = names.strip()
    name = TypeMinor._remove_tail_digits("sjsj00")

    print(name)
    print(info)