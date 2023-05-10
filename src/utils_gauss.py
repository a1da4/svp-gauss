import logging
import pickle
import torch
import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import braycurtis, canberra, chebyshev, cityblock, correlation, cosine

from metrics import *

class Gauss:
    def __init__(self, vecs):
        self.mean = np.average(vecs, axis=0) 
        self._cov = np.cov(vecs, rowvar=0)
        diag_cov = np.diag(self._cov)
        self.cov = np.diag(diag_cov)
        self._num_usages = len(vecs)
    def del_fulcov(self):
        self._cov = None


def load_word2vecs(wordvec_path, target_words=None):
    """ 
    convert dict[word] = [tensor, tensor, tensor, ...] into dict[word] = array(len([tensors]), bert_dim)    
    each tensor represents each usage (sentence) of word

    :param wordvec_path: path of dict[word] = [tensor, tensor, tensor, ...] (.pkl, obtained from wordvec_from_bert.py)
    :return: word2vecs
    """
    word2vecs_torch = pickle.load(open(wordvec_path, "rb"))
    word2vecs_np = {}
    if target_words is None:
        target_words = word2vecs_torch.keys()
    for word in tqdm(target_words, desc="[load word2vecs]"):
        logging.debug(f"[load_word2vecs] target word: {word}")
        vecs = word2vecs_torch[word]
        # memory issue
        word2vecs_torch[word] = None
        vecs_np = np.zeros([len(vecs), len(vecs[0])])
        for vec_id in range(len(vecs)):
            vecs_np[vec_id] += vecs[vec_id].numpy()
            if vec_id < 5:
                logging.debug(f" - {vec_id}-th vecs ")
                logging.debug(f"   - vecs(torch): {vecs[vec_id][:5]} ")
                logging.debug(f"   - vecs(np): {vecs_np[vec_id][:5]}")
        word2vecs_np[word] = vecs_np
        logging.debug(f" - dim: {vecs_np.shape}")

    return word2vecs_np


def obtain_word2gauss(word2vecs):
    """
    obtain dict[word] = Gauss

    :param word2vecs: dict[word] = array(len(usages), bert_dim)
    :return: word2gauss
    """
    word2gauss = {}
    for word in tqdm(word2vecs.keys(), desc="[obtain word2gauss]"):
        word2gauss[word] = Gauss(word2vecs[word])
        logging.debug(f"[obtain_word2gauss] {word}: ")
        logging.debug(f" - mean: {word2gauss[word].mean.shape} dim")
        logging.debug(f" - mean: {word2gauss[word].mean[:5]}")
        logging.debug(f" - cov: {word2gauss[word].cov.shape} dim")
        logging.debug(f" - cov: {word2gauss[word].cov[0][:5]}")

    return word2gauss


def write_rank_freq(target_words, word2gauss_c1, word2gauss_c2, output_name):
    """ 
    save word's freq and rank of covariance matrix

    :param target_words: list of words
    :param word2gauss_c1, word2gauss_c2: dict[word] = Gauss()
    """
    if target_words is None:
        target_words = word2gauss_c1.keys()
    with open(f"../results/freq_rank_{output_name}_c1.txt", "w") as fp_c1:
        with open(f"../results/freq_rank_{output_name}_c2.txt", "w") as fp_c2:
            with open(f"../results/freq_rank_{output_name}_c1_c2.txt", "w") as fp_c1_c2:
                fp_c1.write("word\tfreq\trank\n")
                fp_c2.write("word\tfreq\trank\n")
                fp_c1_c2.write("word\tfreq\trank\n")
                for word in target_words:
                    logging.debug(f"[write_rank_freq] word: {word}")

                    freq_c1 = word2gauss_c1[word]._num_usages
                    freq_c2 = word2gauss_c2[word]._num_usages
                    cov_c1 = word2gauss_c1[word]._cov 
                    cov_c2 = word2gauss_c2[word]._cov 
                    rank_c1 = np.linalg.matrix_rank(cov_c1)
                    rank_c2 = np.linalg.matrix_rank(cov_c2)
                    
                    logging.debug(f"- freq(c1): {freq_c1}")
                    logging.debug(f"- rank(c1): {rank_c1}")
                    logging.debug(f"- freq(c2): {freq_c2}")
                    logging.debug(f"- rank(c2): {rank_c2}")

                    fp_c1.write(f"{word}\t{freq_c1}\t{rank_c1}\n")
                    fp_c2.write(f"{word}\t{freq_c2}\t{rank_c2}\n")
                    fp_c1_c2.write(f"{word}_c1\t{freq_c1}\t{rank_c1}\n")
                    fp_c1_c2.write(f"{word}_c2\t{freq_c2}\t{rank_c2}\n")


def delete_fullcov(word2gauss):
    for word in word2gauss.keys():
        word2gauss[word].del_fulcov()

    return word2gauss


def calculate_metrics(target_words, word2gauss_c1, word2gauss_c2, metrics, cov_component="diag"):
    """
    calculate metrics
    
    :param target_words: list of words
    :param word2gauss_c1, word2gauss_c2: dict[word]: Gauss()
    :param metrics: list of metrics
    :param cov_component: "diag" or "full", components of covariance matrix

    :return: word2pred dict[word]: {metric_name: value}
    """
    word2pred = {}
    for word in tqdm(target_words, desc="[calculate metrics]"):
        logging.debug(f"[calculate_metrics] word: {word}")
        word2pred[word] = {}
        gauss_c1 = word2gauss_c1[word]
        gauss_c2 = word2gauss_c2[word]
        if "l2_mean" in metrics:
            l2_mean = l2_norm(gauss_c1.mean, gauss_c2.mean)
            logging.debug(f"[calculate_metrics] - l2(mean): {l2_mean}")
            word2pred[word]["l2_mean"] = l2_mean
        if "l2_cov" in metrics:
            l2_cov = l2_norm(gauss_c1.cov, gauss_c2.cov)
            logging.debug(f"[calculate_metrics] - l2(cov): {l2_cov}")
            word2pred[word]["l2_cov"] = l2_cov
        if "l2_mean_cov" in metrics:
            if cov_component == "diag":
                # diag(cov)
                l2_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1.cov, \
                                                              gauss_c2.mean, gauss_c2.cov, \
                                                              l2_norm, N=1000)
            if cov_component == "full":
                # full(cov)
                l2_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1._cov, \
                                                              gauss_c2.mean, gauss_c2._cov, \
                                                              l2_norm, N=1000)
            logging.debug(f"[calculate_metrics] - l2(mean;cov): {l2_mean_cov}")
            word2pred[word]["l2_mean_cov"] = l2_mean_cov
        if "kl_c1_c2" in metrics:
            kl_c1_c2 = kl_div(gauss_c1.mean, gauss_c1.cov, gauss_c2.mean, gauss_c2.cov)
            logging.debug(f"[calculate_metrics] - kl(c1||c2): {kl_c1_c2}")
            word2pred[word]["kl_c1_c2"] = kl_c1_c2
        if "kl_c2_c1" in metrics:
            kl_c2_c1 = kl_div(gauss_c2.mean, gauss_c2.cov, gauss_c1.mean, gauss_c1.cov)
            logging.debug(f"[calculate_metrics] - kl(c2||c1): {kl_c2_c1}")
            word2pred[word]["kl_c2_c1"] = kl_c2_c1
        if "jeff" in metrics:
            jeff = jeff_div(gauss_c1.mean, gauss_c1.cov, gauss_c2.mean, gauss_c2.cov)
            logging.debug(f"[calculate_metrics] - jeff(c1||c2): {jeff}")
            word2pred[word]["jeff"] = jeff
        if "jsd" in metrics:
            jsd = js_div_with_affine_spherical(gauss_c1.mean, gauss_c1.cov, gauss_c2.mean, gauss_c2.cov)
            logging.debug(f"[calculate_metrics] - js(c1||c2): {jsd}")
            word2pred[word]["jsd"] = jsd
        if "braycurtis_mean" in metrics:
            braycurtis_mean = braycurtis(gauss_c1.mean, gauss_c2.mean)
            logging.debug(f"[calculate_metrics] - braycurtis(mean): {braycurtis_mean}")
            word2pred[word]["braycurtis_mean"] = braycurtis_mean
        if "braycurtis_cov" in metrics:
            braycurtis_cov = braycurtis(np.diag(gauss_c1.cov), np.diag(gauss_c2.cov))
            logging.debug(f"[calculate_metrics] - braycurtis(cov): {braycurtis_cov}")
            word2pred[word]["braycurtis_cov"] = braycurtis_cov
        if "braycurtis_mean_cov" in metrics:
            if cov_component == "diag":
                # diag(cov)
                braycurtis_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1.cov, \
                                                                      gauss_c2.mean, gauss_c2.cov, \
                                                                      braycurtis, N=1000)
            if cov_component == "full":
                # full(cov)
                braycurtis_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1._cov, \
                                                                      gauss_c2.mean, gauss_c2._cov, \
                                                                      braycurtis, N=1000)
            logging.debug(f"[calculate_metrics] - braycurtis(mean;cov): {braycurtis_mean_cov}")
            word2pred[word]["braycurtis_mean_cov"] = braycurtis_mean_cov
        if "canberra_mean" in metrics:
            canberra_mean = canberra(gauss_c1.mean, gauss_c2.mean)
            logging.debug(f"[calculate_metrics] - canberra(mean): {canberra_mean}")
            word2pred[word]["canberra_mean"] = canberra_mean
        if "canberra_cov" in metrics:
            canberra_cov = canberra(np.diag(gauss_c1.cov), np.diag(gauss_c2.cov))
            logging.debug(f"[calculate_metrics] - canberra(cov): {canberra_cov}")
            word2pred[word]["canberra_cov"] = canberra_cov
        if "canberra_mean_cov" in metrics:
            if cov_component == "diag":
                # diag(cov)
                canberra_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1.cov, \
                                                                    gauss_c2.mean, gauss_c2.cov, \
                                                                    canberra, N=1000)
            if cov_component == "full":
                # full(cov)
                canberra_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1._cov, \
                                                                    gauss_c2.mean, gauss_c2._cov, \
                                                                    canberra, N=1000)
            logging.debug(f"[calculate_metrics] - canberra(mean;cov): {canberra_mean_cov}")
            word2pred[word]["canberra_mean_cov"] = canberra_mean_cov
        if "chebyshev_mean" in metrics:
            chebyshev_mean = chebyshev(gauss_c1.mean, gauss_c2.mean)
            logging.debug(f"[calculate_metrics] - chebyshev(mean): {chebyshev_mean}")
            word2pred[word]["chebyshev_mean"] = chebyshev_mean
        if "chebyshev_cov" in metrics:
            chebyshev_cov = chebyshev(np.diag(gauss_c1.cov), np.diag(gauss_c2.cov))
            logging.debug(f"[calculate_metrics] - chebyshev(cov): {chebyshev_cov}")
            word2pred[word]["chebyshev_cov"] = chebyshev_cov
        if "chebyshev_mean_cov" in metrics:
            if cov_component == "diag":
                # diag(cov)
                chebyshev_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1.cov, \
                                                                     gauss_c2.mean, gauss_c2.cov, \
                                                                     chebyshev, N=1000)
            if cov_component == "full":
                # full(cov)
                chebyshev_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1._cov, \
                                                                     gauss_c2.mean, gauss_c2._cov, \
                                                                     chebyshev, N=1000)
            logging.debug(f"[calculate_metrics] - chebyshev(mean;cov): {chebyshev_mean_cov}")
            word2pred[word]["chebyshev_mean_cov"] = chebyshev_mean_cov
        if "cityblock_mean" in metrics:
            cityblock_mean = cityblock(gauss_c1.mean, gauss_c2.mean)
            logging.debug(f"[calculate_metrics] - cityblock(mean): {cityblock_mean}")
            word2pred[word]["cityblock_mean"] = cityblock_mean
        if "cityblock_cov" in metrics:
            cityblock_cov = cityblock(np.diag(gauss_c1.cov), np.diag(gauss_c2.cov))
            logging.debug(f"[calculate_metrics] - cityblock(cov): {cityblock_cov}")
            word2pred[word]["cityblock_cov"] = cityblock_cov
        if "cityblock_mean_cov" in metrics:
            if cov_component == "diag":
                # diag(cov)
                cityblock_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1.cov, \
                                                                     gauss_c2.mean, gauss_c2.cov, \
                                                                     cityblock, N=1000)
            if cov_component == "full":
                # full(cov)
                cityblock_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1._cov, \
                                                                     gauss_c2.mean, gauss_c2._cov, \
                                                                     cityblock, N=1000)
            logging.debug(f"[calculate_metrics] - cityblock(mean;cov): {cityblock_mean_cov}")
            word2pred[word]["cityblock_mean_cov"] = cityblock_mean_cov
        if "correlation_mean" in metrics:
            correlation_mean = correlation(gauss_c1.mean, gauss_c2.mean)
            logging.debug(f"[calculate_metrics] - correlation(mean): {correlation_mean}")
            word2pred[word]["correlation_mean"] = correlation_mean
        if "correlation_cov" in metrics:
            correlation_cov = correlation(np.diag(gauss_c1.cov), np.diag(gauss_c2.cov))
            logging.debug(f"[calculate_metrics] - correlation(cov): {correlation_cov}")
            word2pred[word]["correlation_cov"] = correlation_cov
        if "correlation_mean_cov" in metrics:
            if cov_component == "diag":
                # diag(cov)
                correlation_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1.cov, \
                                                                       gauss_c2.mean, gauss_c2.cov, \
                                                                       correlation, N=1000)
            if cov_component == "full":
                # full(cov)
                correlation_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1._cov, \
                                                                       gauss_c2.mean, gauss_c2._cov, \
                                                                       correlation, N=1000)
            logging.debug(f"[calculate_metrics] - correlation(mean;cov): {correlation_mean_cov}")
            word2pred[word]["correlation_mean_cov"] = correlation_mean_cov
        if "cosine_mean" in metrics:
            cosine_mean = cosine(gauss_c1.mean, gauss_c2.mean)
            logging.debug(f"[calculate_metrics] - cosine(mean): {cosine_mean}")
            word2pred[word]["cosine_mean"] = cosine_mean
        if "cosine_cov" in metrics:
            cosine_cov = cosine(np.diag(gauss_c1.cov), np.diag(gauss_c2.cov))
            logging.debug(f"[calculate_metrics] - cosine(cov): {cosine_cov}")
            word2pred[word]["cosine_cov"] = cosine_cov
        if "cosine_mean_cov" in metrics:
            if cov_component == "diag":
                # diag(cov)
                cosine_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1.cov, \
                                                                  gauss_c2.mean, gauss_c2.cov, \
                                                                  cosine, N=1000)
            if cov_component == "full":
                # full(cov)
                cosine_mean_cov = calculate_distance_from_samples(gauss_c1.mean, gauss_c1._cov, \
                                                                  gauss_c2.mean, gauss_c2._cov, \
                                                                  cosine, N=1000)
            logging.debug(f"[calculate_metrics] - cosine(mean;cov): {cosine_mean_cov}")
            word2pred[word]["cosine_mean_cov"] = cosine_mean_cov

    return word2pred


def write_results(word2gold, word2pred, metrics, output_name):
    """
    save results

    :param word2gold: dict[word]: changedvalue (this function does not use values)
    :param word2pred: dict[word]: {"METRIC": value, ...} 
    :param metrics: list of metric name
    :param output_name: name of model / experiment
    """
    with open(f"../results/grade_gold_preds_{output_name}.txt", "w") as fp:
        fp.write("word\tgold")
        for metric in metrics:
            fp.write(f"\t{metric}")
        fp.write("\n")
        for word in word2gold.keys():
            gold = word2gold[word]
            fp.write(f"{word}\t{gold}")
            for metric in metrics:
                value = word2pred[word][metric]
                fp.write(f"\t{value}")
            fp.write("\n")

