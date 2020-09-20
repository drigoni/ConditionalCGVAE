#!/usr/bin/env/python
import numpy as np
import tensorflow as tf


class HistManager:
    # this part is not used, but it is created initially to build the graph
    def __init__(self, n, hist_dim):
        self.placeholders = dict()
        self.weights = dict()
        self.opts = dict()
        self.params = dict()
        self.params['n'] = n
        self.params['hist_dim'] = hist_dim

        self.placeholders['histograms'] = tf.placeholder(tf.int32, (n, hist_dim), name="histograms")  # real histograms
        self.placeholders['n_histograms'] = tf.placeholder(tf.int32, (n, 1),
                                                           name="n_histograms")  # histogram converted value
        self.placeholders['hist'] = tf.placeholder(tf.int32, (hist_dim,), name="hist")  # an histogram
        reshape = tf.reshape(self.placeholders['hist'], (-1, hist_dim))
        m1 = self.placeholders['histograms'] >= reshape
        m2 = tf.reduce_sum(tf.cast(m1, dtype=tf.int32), axis=1)
        m3 = tf.equal(m2, tf.constant(hist_dim))
        m4 = tf.reshape(tf.cast(m3, dtype=tf.int32), (-1, 1))
        m5 = tf.multiply(self.placeholders['n_histograms'], m4)
        mSomma = tf.reduce_sum(m5)

        def __case_sampling():
            m6 = m5 / mSomma
            m7 = tf.squeeze(tf.reshape(m6, (1, -1)))
            m8 = tf.distributions.Categorical(probs=m7).sample(1)[0]
            m9 = self.placeholders['histograms'][m8]
            return m9

        def __case_0():
            return tf.constant(-1)

        self.opts['res'] = tf.cond(tf.equal(tf.constant(0), mSomma), __case_0, __case_sampling)

    def sampleCompatible(self, session, data, n_data, hist):
        feed_dict = {
            self.placeholders['histograms']: data,
            self.placeholders['hist']: hist,
            self.placeholders['n_histograms']: n_data
        }
        return session.run(self.opts['res'], feed_dict=feed_dict)

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
    Calculates the weights for each histograms considering each number of atoms.
    Return a list of A (number of atoms) lists, where Ai is the wights for each histogram according to the number of atoms Ai
    '''

    @staticmethod
    def v_filter(hist, n_hist, number_of_max_atoms):
        assert type(hist) == list  # list of histograms
        assert type(n_hist) == list  # list of frequency for each histogram

        # inizialize the dictionary
        diz = dict()
        diz_prob = dict()

        # for each number of atoms.
        for i in range(number_of_max_atoms + 1):
            # initialize an empty list
            res = np.array([0] * len(hist))

            # for each histogram
            for j in range(len(hist)):
                # counting the number of atoms
                if np.sum(hist[j]) >= i:
                    res[j] = n_hist[j]

            diz[i] = res
            somma = np.sum(res)
            if somma == 0:
                diz_prob[i] = res
            else:
                diz_prob[i] = res / somma

        return diz, diz_prob


def test1():
    hist = []
    n_hist = []
    for i in range(4):
        hist.append([np.random.randint(0, 2, 4)])  # generates number from 0 to 1
        n_hist.append(np.random.randint(0, 100))
    print(hist)
    print(n_hist)
    diz, diz_prob = HistManager.v_filter(hist, n_hist, 4)
    print(diz)
    print(diz_prob)


if __name__ == "__main__":
    test1()
