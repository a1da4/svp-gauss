import os
import argparse
import pickle
import logging
import random
import torch
import numpy as np
from scipy.stats import spearmanr

from utils_gauss import *


def main_gauss(args):
    os.makedirs("../results", exist_ok=True)
    logging.basicConfig(filename="../results/main_gauss.log", format="%(asctime)s %(message)s", level=logging.INFO)
    logging.info(f"[main_gauss] args: {args}")

    logging.info("[main_gauss] load word2grade (gold) ...")
    word2gold = {}
    with open(args.graded_words_list) as fp:
        for line in fp:
            word, grade = line.strip().split("\t")
            word2gold[word] = float(grade)

    # analyse randomly sampled 1000 words
    if args.words_list is not None:
        target_words = []
        with open(args.words_list) as fp:
            for line in fp:
                word = line.strip()
                target_words.append(word)

        random.seed(12345)
        sample_target_words = random.sample(target_words, k=1000)
    else:
        sample_target_words = None

    logging.info("[main_gauss] load word2vecs / obtain word2gauss...")
    word2vecs_c1 = load_word2vecs(args.wordvec_pathes[0], sample_target_words)
    word2gauss_c1 = obtain_word2gauss(word2vecs_c1)
    del word2vecs_c1
    word2vecs_c2 = load_word2vecs(args.wordvec_pathes[1], sample_target_words)
    word2gauss_c2 = obtain_word2gauss(word2vecs_c2)
    del word2vecs_c2
    logging.info("[main_gauss] word2gauss obtained successfully")

    logging.info("[main_gauss] save rank-freq ...")
    write_rank_freq(sample_target_words, word2gauss_c1, word2gauss_c2, args.output_name)
    logging.info("[main_gauss] rank-freq saved succeccfully")
    # analyse randomly sampled 1000 words, not make predictions
    if sample_target_words is not None:
        exit()

    logging.info("[main_gauss] calculate divergence N(mean, diag(cov)) ...")
    metrics = ["kl_c1_c2", "kl_c2_c1", "jeff",
               "l2_mean", "l2_cov", "l2_mean_cov",
               "braycurtis_mean", "braycurtis_cov", "braycurtis_mean_cov",
               "canberra_mean", "canberra_cov", "canberra_mean_cov",
               "chebyshev_mean", "chebyshev_cov", "chebyshev_mean_cov",
               "cityblock_mean", "cityblock_cov", "cityblock_mean_cov",
               "correlation_mean", "correlation_cov", "correlation_mean_cov",
               "cosine_mean", "cosine_cov", "cosine_mean_cov"]
    word2pred_diag = calculate_metrics(word2gold.keys(), word2gauss_c1, word2gauss_c2, metrics, cov_component="diag")
    logging.info("[main_gauss] divergence calculated successfully")

    logging.info("[main_gauss] save results ...")
    write_results(word2gold, word2pred_diag, metrics, f"{args.output_name}_diag")
    logging.info("[main_gauss] saved successfully")


    logging.info("[main_gauss] calculate divergence N(mean, full(cov)) ...")
    metrics = ["l2_mean_cov",
               "braycurtis_mean_cov",
               "canberra_mean_cov",
               "chebyshev_mean_cov",
               "cityblock_mean_cov",
               "correlation_mean_cov",
               "cosine_mean_cov"]
    word2pred_full = calculate_metrics(word2gold.keys(), word2gauss_c1, word2gauss_c2, metrics, cov_component="full")
    logging.info("[main_gauss] divergence calculated successfully")

    logging.info("[main_gauss] save results ...")
    write_results(word2gold, word2pred_full, metrics, f"{args.output_name}_full")
    logging.info("[main_gauss] saved successfully")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--wordvec_pathes", nargs=2, help="path of word2vecs (.pkl, obtained from bert")
    parser.add_argument("-w", "--words_list", help="target word list for analyze (freq-rank). default=None")
    parser.add_argument("-l", "--graded_words_list", help="annotated target word list")
    parser.add_argument("-o", "--output_name")
    args = parser.parse_args()
    main_gauss(args)


if __name__ == "__main__":
    cli_main()
