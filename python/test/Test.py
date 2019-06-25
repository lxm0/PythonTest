import re

import numpy as np
from pattern.text.en import tag
from pynaming.str_utils import find_longest_common_str, is_abbrev_for_multiple, is_singluar, is_adjective, \
    is_verb, is_adverb, is_noun
from sklearn.cluster import KMeans
from nltk.corpus import wordnet as wn
from sklearn.svm import SVC

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
def get_pos(text):
    return [(w, penn_to_wn(p)) for w, p in tag(text)]

class TypeMinor(object):
    def __init__(self, training_file, testing_file, types_file, k=25):
        self.training_filepath = training_file
        self.testing_filepath = testing_file
        self.types_filepath = types_file
        self.type2id = {}
        self.id2type = {}

        # for storing features
        self.id2cluster = {}
        self.id2posfeature = {}
        self.id2pluarfeature = {}
        self.id_type2hasfeature = {}

        self.id2names = {}
        self.raw_training_data = []
        self.raw_testing_data = []
        self.variables = []
        self.n_clusters = 20
        self.dataset = None
        self.classifier = None
        self.type2classifier = {}
        self.cluster = None
        self.results = []
        self.predicts = {}
        self.history = {}
        self.history_probas = {}
        self.k = k

    def _init_raw_data(self):
        # self.types_filepath = "F:\\IntelliJ IDEA\\PythonTest\\python\\test\\Data\\stypes.txt"
        # self.training_filepath = "F:\\IntelliJ IDEA\\PythonTest\\python\\test\Data\\same.txt"
        # self.testing_filepath = "F:\\IntelliJ IDEA\\PythonTest\\python\\test\\Data\\diff.txt"
        self.raw_training_data = self._load_raw_data(self.training_filepath)
        self.raw_testing_data = self._load_raw_data(self.testing_filepath)
        # print("train_data*********")
        # print(self.raw_training_data)
        # print("test_data********")
        # print(self.raw_testing_data)
        self.variables = np.array(list(set(map(lambda d: d['id'], self.raw_training_data) +
                                           map(lambda d: d['id'], self.raw_testing_data))))
        # print("variables********")
        # print(self.variables)
        # init labels
        types = set()
        for info in self.raw_training_data:
            types.update(info['stypes'])
            types.update(info['dtypes'])
        for info in self.raw_testing_data:
            types.update(info['stypes'])
            types.update(info['dtypes'])
        with open(self.types_filepath) as f:
            lines = f.read().splitlines()
            for t in lines:
                if t.strip():
                    types.add(t.strip())
        if '?' in types:
            types.remove('?')
        # add initial builtin types
        self.type2id = {t: i for i, t in enumerate(types)}
        # print("type2id**********")
        # print(self.type2id)

    def _load_raw_data(self,filepath):
        data = []

        with open(filepath, 'rb') as f:
            for line in f.readlines():
                infos = line.strip().split()
                # print(infos)
                if not re.match('^_+[\d_]*$', infos[0]):  # we ignore all variables only consists of char '_'
                    data.append({'id': infos[0], 'stypes': infos[1].split('|'),
                                 'dtypes': infos[2].split('|'), 'context': infos[3],
                                 'location': infos[4]})
        return data

    def _train_cluster_variables(self):
        # variables = np.array(list(set(map(lambda d: d['id'], self.raw_training_data) +
        #                               map(lambda d: d['id'], self.raw_testing_data))))
        variables = self.variables
        distance_matrix = -1 * self._compute_str_distances(variables)
        # clustering = DBSCAN(metric='precomputed', eps=0.8, min_samples=1) # not work
        # clustering = MeanShift()  # not work
        clustering = KMeans(n_clusters=self.k)
        # clustering = AffinityPropagation(affinity='precomputed', damping=0.5)
        # clustering = AgglomerativeClustering(n_clusters=self.n_clusters)
        resukt = clustering.fit(distance_matrix)
        # print(resukt.labels_)

        self.id2cluster = dict(zip(variables, clustering.labels_))
        self.cluster = clustering

        # debuging the clustering
        # for cluster_id in np.unique(clustering.labels_):
        #     cluster = variables[np.nonzero(clustering.labels_ == cluster_id)]
        #     cluster_str = ", ".join(cluster)
        #     print cluster_str
    def _compute_str_distances(self, data):
        _extract_str_distance = self._extract_str_distance
        return np.array([[_extract_str_distance(name1, name2) for name2 in self.variables]
                         for name1 in data])

    def _extract_str_distance(self, var1, var2):
        names1, names2 = self._split_var(var1), self._split_var(var2)
        dis = (self._compute_substr_distance(names1, names2)
               + self._compute_abbrvstr_distance(names1, names2)
               # + self._compute_pos_distance(names1, names2)
               # + self._compute_sp_distance(names1, names2)
               )
        return dis

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

    def _extract_target(self, info):
        return self.type2id[list(info['dtypes'])[0]]

    def _extract_dataset(self):
        data = {tname: [] for tname in self.type2id.iterkeys()}
        targets = []
        for info in self.raw_training_data:
            for tname in self.type2id.iterkeys():
                data[tname].append(self._extract_features(info, tname))
            targets.append(self._extract_target(info))

        id2label = {v: k for k, v in self.type2id.iteritems()}
        self.dataset = {'data': {tname: np.array(d) for tname, d in data.iteritems()},
                        'targets': targets}
        self.id2type = id2label

    def _extract_features(self, info, tname):
        vname = info['id'] if isinstance(info, dict) else str(info)
        # extract cluster id
        if vname in self.id2cluster:
            id_lit_f = self.id2cluster[vname]
        else:
            id_lit_f = self._predict_cluster(vname)
        # extract nouns#, verbs#
        if vname not in self.id2posfeature:
            id_pos_f =  self._extract_pos_feature(vname)
            self.id2posfeature[vname] = id_pos_f
        else:
            id_pos_f = self.id2posfeature[vname]
        # extract has_plural
        if vname not in self.id2pluarfeature:
            id_sp_f = self._extract_sp_feature(vname)
            self.id2pluarfeature[vname] = id_sp_f
        else:
            id_sp_f = self.id2pluarfeature[vname]
        # extract vname ,tname
        if (vname, tname) not in self.id_type2hasfeature:
            has_f = self._extract_has_feature(vname, tname)
            self.id_type2hasfeature[(vname, tname)] = has_f
        else:
            has_f = self.id_type2hasfeature[(vname, tname)]

        return [id_lit_f, id_pos_f, id_sp_f, has_f]

    def _predict_cluster(self, vname):
        distance = -1 * self._compute_str_distances([vname])
        return self.cluster.predict(distance)[0]

    def _extract_has_feature(self, var, tname):
        return self._extract_str_distance(var, tname)

    # def _extract_pos_feature(self, var):
    #     names = self._split_var(var)
    #     tags = [t for _, t in get_pos(' '.join(names))]
    #     for t in tags:
    #         if t == wn.VERB:
    #             return 1
    #         if t == wn.NOUN:
    #             return 0
    #     else:
    #         return 2

    def _extract_pos_feature(self,var):
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
    # def is_noun(tag):
    #     return tag in ['NN', 'NNS', 'NNP', 'NNPS']
    #
    #
    # def is_verb(tag):
    #     return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    #
    #
    # def is_adverb(tag):
    #     return tag in ['RB', 'RBR', 'RBS']
    #
    #
    # def is_adjective(tag):
    #     return tag in ['JJ', 'JJR', 'JJS']

    def _extract_sp_feature(self, var):
        names = self._split_var(var)
        return int(is_singluar(names[-1]))

    def evaluate_all_tests(self):
        self.evaluate_all_ovo()

    def evaluate_all_ovo(self):
        test_data = self._extract_test_dataset()
        predicted_data_proba = self._inner_predict_proba(test_data['data'])
        for i, info in enumerate(self.raw_testing_data):
            #types = {k for k, v in predicted_data_proba[i].iteritems() if v >= 0.5}
            type_probas = sorted(predicted_data_proba[i].items(), key=lambda x: -x[1])
            first_cluster = self._group_data_by_gap(type_probas, 0.1)[0]
            first_cluster_top5 = first_cluster[:min(len(first_cluster), 5)]
            first_cluster_top5 = filter(lambda x: x[1] >= 0.05, first_cluster_top5)
            types = map(lambda x: x[0], first_cluster_top5)
            self.results.append({'id': info['id'], 'context': info['context'],
                                 'location': info['location'], 'dtypes': info['dtypes'],
                                 'stypes': filter(lambda x: x != '?', info['stypes']),
                                 'pred_proba': predicted_data_proba[i], 'ptypes': types})

    def _extract_test_dataset(self):
        data = {tname: [] for tname in self.type2id.iterkeys()}
        targets = []
        for info in self.raw_testing_data:
            for tname in self.type2id.iterkeys():
                data[tname].append(self._extract_features(info, tname))
            targets.append(self._extract_target(info))
        return {'data': {tname: np.array(d) for tname, d in data.iteritems()},
                'targets': targets}

    def _inner_predict_proba(self, data):
        # data[type][records]
        proba = {i: {} for i in range(len(data.itervalues().next()))}
        for tname in self.type2id.iterkeys():
            if tname not in self.type2classifier:
                for i in range(len(data[tname])):
                    proba[i][tname] = 0.0
                continue
            classifier = self.type2classifier[tname]
            assert hasattr(classifier, 'predict_proba'), \
                'The classifier does not support propbabilistic prediction'
            for i, (p0, p1) in enumerate(classifier.predict_proba(data[tname])):
                proba[i][tname] = p1
            # proba.append((tname, classifier.predict_proba(data)[0][1]))
        return proba

    def _train_classify_names_ovo(self):
        for tname, id in self.type2id.iteritems():
            targets = map(lambda x: int(x == id), self.dataset['targets'])
            if not any(targets):
                continue
            # classifier = LogisticRegression(multi_class='multinomial', C=150, solver='lbfgs')
            classifier = SVC(probability=True)
            classifier.fit(self.dataset['data'][tname], targets)
            self.type2classifier[tname] = classifier

    @staticmethod
    def _group_data_by_gap(type_probas, gap):
        elm_wise_data = zip(type_probas, type_probas[1:])
        groups = []
        current_group = [type_probas[0]]
        for (t1, p1), (t2, p2) in elm_wise_data:
            if p1 - p2 <= gap:
                current_group.append((t2, p2))
            else:
                groups.append(current_group)
                current_group = []
        if current_group:
            groups.append(current_group)
        return groups

    @staticmethod
    def _compute_abbrvstr_distance(names1, names2):
        names1, names2 = map(str.lower, names1), map(str.lower, names2)
        name1, name2 = ''.join(names1), ''.join(names2)
        if len(name1) > len(name2):
            name1, names2 = (name2, names1)
        return 1 - is_abbrev_for_multiple(name1, names2)

    @staticmethod
    def _compute_substr_distance(names1, names2):
        name1 = ''.join(map(str.lower, names1))
        name2 = ''.join(map(str.lower, names2))
        min_len = min(len(name1), len(name2))
        substr = list(find_longest_common_str(name1, name2))
        return 1 - (len(substr[0]) if substr else 0) * 1.0 / min_len

    @staticmethod
    def _remove_tail_digits(name):
        return name.rstrip('1234567890')

    @staticmethod
    def _check_if_all_capitals(name):
        return name.upper() == name

    @staticmethod
    def _split_by_captials(name):
        return filter(bool, re.split('([A-Z][^A-Z]*)', name))

if __name__ == '__main__':
    types_filepath = "F:\\IntelliJ IDEA\\PythonTest\\python\\test\\Data\\stypes.txt"
    training_filepath = "F:\\IntelliJ IDEA\\PythonTest\\python\\test\Data\\same.txt"
    testing_filepath = "F:\\IntelliJ IDEA\\PythonTest\\python\\test\\Data\\diff.txt"
    minor = TypeMinor(training_filepath, testing_filepath, types_filepath)
    minor._init_raw_data()
    # Kmean
    minor._train_cluster_variables()

    minor._extract_dataset()

    minor._train_classify_names_ovo()

    minor.evaluate_all_tests()

    print(minor.results)
    # print(minor.dataset)

