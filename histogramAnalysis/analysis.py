"""
Usage:
    make_dataset.py [options]

Options:
    -h --help           Show this screen.
    --dataset NAME      QM9 or ZINC
"""
import json
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from docopt import docopt

current_dir = os.path.dirname(os.path.realpath(__file__))


class HistManager:
    '''
    Constructor
    '''

    def __init__(self, histograms, length_hist, max_valence):
        assert type(histograms[0]) == list
        print("START BUILDING DICTIONARY")
        import time
        start = time.time()
        self.length_hist = length_hist  # lunghezza array istogramma
        self.max_valence = max_valence  # massima y vista nel dataset
        self.n_data = len(histograms)
        self.dict = HistManager.makeDict(histograms, self.max_valence)
        self.keys_ordered = HistManager.getOrderedKeys(self.dict)
        self.n_hist = len(self.keys_ordered)
        self.dict_compatible = HistManager.makeDictCompatible(self.dict, length_hist, max_valence)
        end = time.time()
        print("END in time: " + str(end - start))

    '''
    Return a dictionary where the key is the score, and the value is a list of value. es {7:[7,7,7,7]} 
    Note that all the values are repeated.
    Input: a list of histograms
    Output: the dictionary
    '''

    @staticmethod
    def makeDict(histograms: list, max_valence: int) -> defaultdict:
        diz = defaultdict(list)
        for i in histograms:
            score = HistManager.histToScore(i, max_valence)
            diz[score].append(score)
        return diz

    '''
    Convert the histogram in a score using the length_hist
    '''

    @staticmethod
    def histToScore(hist: list, max_valence: int) -> int:
        n = len(hist)
        score = 0
        for i in range(n):
            inv = n - i - 1
            score += hist[i] * ((max_valence + 1) ** inv)
        return score

    '''
    Return a list of ordered keys belonging to a dictionary
    '''

    @staticmethod
    def getOrderedKeys(data: defaultdict) -> list:
        ord_keys = []
        for i in sorted(data.keys()):
            ord_keys.append(i)
        return ord_keys

    '''
    Given a dictionary of histograms, it create the compatible one. 
    Note that it is a list of list, where for each histogram there are all the compatible histogram.
    Es: {7: [7,7,8,10,33,33,33]} 
    '''

    @staticmethod
    def makeDictCompatible(data: defaultdict, length_hist: int, max_valence: int) -> defaultdict:  # todo verifica
        diz_comp = defaultdict(list)
        ord_keys = HistManager.getOrderedKeys(data)
        for i in range(len(ord_keys)):
            score = ord_keys[i]
            hist = HistManager.scoreToHist(score, length_hist, max_valence)
            # only from i included, in this way it count itself
            for j in range(i, len(ord_keys)):
                current_score = ord_keys[j]
                current_hist = HistManager.scoreToHist(current_score, length_hist, max_valence)
                # if current_hist is compatible with hist
                if HistManager.compatible(hist, current_hist):
                    diz_comp[score].extend(data[current_score])
        return diz_comp

    '''
    Convert the score to the corresponding histogram using length_hist.
    After it adjusts the length according to max_valence
    '''

    @staticmethod
    def scoreToHist(score: int, length_hist: int, max_valence: int) -> list:
        rem = list()
        while score > 0:
            rem.insert(0, score % (max_valence + 1))
            score = score // (max_valence + 1)
        while len(rem) < length_hist:
            rem.insert(0, 0)
        return rem

    '''
    Compare two histograms
    Input: a and b histograms
    Output: True if b is compatible with a
    '''

    @staticmethod
    def compatible(a: list, b: list) -> bool:
        assert len(a) == len(b)
        for i in range(len(a)):
            if a[i] > b[i]:
                return False
        return True

    ''' 
    Returns all the histograms that are compatible with the one in input 
    Input: one histogram
    Output: list of alla the compatible histogram with they're distributions
    '''

    def findAllCompatible(self, hist=None) -> list:
        if hist is not None:
            s = HistManager.histToScore(hist, self.max_valence)
        else:
            hist = [0]
            s = 0

        idx = self.__binary_idx_search(s)

        # find all and add it for the next time
        comp = []
        # check if it exists in the compatible one dictionary
        if self.keys_ordered[idx] == s:
            key = self.keys_ordered[idx]
            comp.extend(self.dict_compatible[key])
        else:
            # add
            for i in range(idx, self.n_hist):
                key = self.keys_ordered[i]
                current_hist = HistManager.scoreToHist(key, self.length_hist, self.max_valence)
                if HistManager.compatible(hist, current_hist):
                    comp.extend(self.dict[key])
            self.dict_compatible[s] = comp

        # convert
        copy = []
        for i in comp:
            transf = HistManager.scoreToHist(i, self.length_hist, self.max_valence)
            copy.append(transf)
        return comp

    ''' 
    Given a score, give the key for the right bucket of compatible hist 
    Input: a score number corresponding to a histogram
    Output: the index of the ordered key array where take all the compatibles one
    '''

    def __binary_idx_search(self, score: int) -> int:
        start = 0
        end = len(self.keys_ordered)
        while True:
            val = start + (end - start) / 2
            idx = int(round(val, 0))
            if score > self.keys_ordered[idx]:
                start = idx
            elif score < self.keys_ordered[idx]:
                end = idx
            else:
                return idx

            if (start + 1) < self.length_hist:
                if start + 1 == end:
                    if score > self.keys_ordered[start]:
                        return end
                    else:
                        return start
            else:
                return start

    '''
    Sample from all the histogram compatible with the one in input
    Input: one histogram
    Output: one histogram
    '''

    def sample(self, hist=None) -> list:
        all_hist = self.findAllCompatible(hist)
        idx = np.random.randint(0, len(all_hist))
        score = all_hist[idx]
        res = HistManager.scoreToHist(score, self.length_hist, self.max_valence)
        return res

    # plotting function
    def plotCompatible(self, name):
        y_cum = []
        for i in self.dict_compatible:
            y_cum.append(len(self.dict_compatible[i]))
        plt.clf()
        # plt.figure(figsize=[30, 20])
        plt.plot(range(self.n_hist), y_cum)
        plt.title(name + " Compatible Histograms")
        plt.savefig(name + "_compatible.png")

    def plotHist(self, name):
        y_hist = []
        for i in self.keys_ordered:
            y_hist.append(len(self.dict[i]))
        plt.clf()
        # plt.figure(figsize=[30, 20])
        plt.plot(range(self.n_hist), y_hist)
        plt.title(name + " Unique Histograms")
        plt.savefig(name + "_hist.png")

    def plotCumulative(self, name):
        y_hist = []
        for i in self.keys_ordered:
            y_hist.append(len(self.dict_compatible[i]))
        plt.clf()
        # plt.figure(figsize=[30, 20])
        plt.plot(range(self.n_hist), sorted(y_hist, reverse=True))
        plt.xlabel("Histogram ordered by compatibility")
        plt.ylabel("Number of molecules")
        plt.title(name + " Sorted Comp. Histograms")
        plt.savefig(name + "_sortedCompatible.png")

    # other tests
    def test1(self):
        for i in range(self.n_hist):
            s = self.keys_ordered[i]
            histConf = HistManager.scoreToHist(s, self.length_hist, self.max_valence)
            all_scores = self.dict_compatible[s]
            for current_score in all_scores:
                hist = HistManager.scoreToHist(current_score, self.length_hist, self.max_valence)
                if not HistManager.compatible(histConf, hist):
                    print("Error")
                    exit(1)
            i += 1

    def test2(self):
        print("START TEST DICTIONARY")
        import time
        start = time.time()
        print("Len data: " + str(self.n_data))
        print("Len histograms: " + str(self.n_hist))
        print("Len max hist: " + str(self.length_hist))
        print("Max value valence: " + str(self.max_valence))
        # print("Ordered keys: " + str(self.keys_ordered))
        # print("Dictionary: " + str(self.dict))
        # print("Dictionary compatible: " + str(self.dict_compatible))
        # hist = HistManager.scoreToHist(0, self.length_hist, self.max_valence)
        # print(self.findAllCompatible(hist))
        # print("Dictionary compatible: " + str(self.dict_compatible))
        end = time.time()
        print("END in time: " + str(end - start))
        print("Self size MB: " + str(HistManager.get_obj_size(self) / (1000 * 1000)))

    @staticmethod
    def get_obj_size(obj):
        import gc
        import sys
        marked = {id(obj)}
        obj_q = [obj]
        sz = 0

        while obj_q:
            sz += sum(map(sys.getsizeof, obj_q))

            # Lookup all the object reffered to by the object in obj_q.
            # See: https://docs.python.org/3.7/library/gc.html#gc.get_referents
            all_refr = ((id(o), o) for o in gc.get_referents(*obj_q))

            # Filter object that are already marked.
            # Using dict notation will prevent repeated objects.
            new_refr = {o_id: o for o_id, o in all_refr if o_id not in marked and not isinstance(o, type)}

            # The new obj_q will be the ones that were not marked,
            # and we will update marked with their ids so we will
            # not traverse them again.
            obj_q = new_refr.values()
            marked.update(new_refr.keys())

        return sz


if __name__ == "__main__":
    args = docopt(__doc__)
    dataset = args.get('--dataset')

    # READ FROM FILES
    full_path = current_dir + "/../data/"
    test_file = full_path + "molecules_test_" + dataset + ".json"
    valid_file = full_path + "molecules_valid_" + dataset + ".json"
    train_file = full_path + "molecules_train_" + dataset + ".json"

    # loading data
    print("Loading data from %s" % test_file)
    with open(test_file, 'r') as f:
        data1 = json.load(f)

    print("Loading data from %s" % valid_file)
    with open(valid_file, 'r') as f:
        data2 = json.load(f)

    print("Loading data from %s" % train_file)
    with open(train_file, 'r') as f:
        data3 = json.load(f)

    # prepare histograms
    mols = []
    mols.extend(data1)
    mols.extend(data2)
    mols.extend(data3)
    all = [mol['hist'] for mol in mols]
    print("Number of molecules: %i" % (len(mols)))

    # default params
    # if dataset == "qm9":
    #     length_hist = 4
    #     max_valence = 9
    # else:
    #     length_hist = 6
    #     max_valence = 34
    length_hist = len(all[0])
    max_valence = max([max(i) for i in all])

    # search
    minSmile = 100000000
    minHist = []  # hist that correspond to the smallest smiles string
    for mol in mols:
        if len(mol['smiles']) < minSmile:
            minSmile = len(mol['smiles'])
            minHist = mol['hist']

    # plotting
    obj = HistManager(all[:], length_hist, max_valence)
    obj.plotCompatible(dataset)
    obj.plotHist(dataset)
    obj.plotCumulative(dataset)

    # check efficiency
    print("START TIME")
    start = time.time()
    vect = np.arange(250000, dtype=np.float32)
    somma = sum(vect)
    np.random.choice(vect, p=(vect / somma).tolist())
    end = time.time()
    print("END in time: " + str(end - start))

    res = obj.findAllCompatible(minHist)
    print("Min smile: " + str(minSmile))
    print("Min hist: " + str(minHist))
    print("len: " + str(len(res)))
    obj.test2()
