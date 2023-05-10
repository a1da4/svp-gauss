import logging
import numpy as np
import scipy

def concat_mean_cov(mean, cov):
    """
    concat mean and diag(cov)

    :param mean: mean vector (D dim, D = BERT dim)
    :param cov: covariance (D*D dim)
    
    :return: vec_mean_cov (2D dim)
    """
    diag_cov = np.diag(cov)
    vec_mean_cov = np.concatenate([mean, diag_cov])

    return vec_mean_cov


def calculate_distance_from_samples(mean_1, cov_1, mean_2, cov_2, dist, N=1000):
    """
    calculate average distance of N samples

    :param mean_1, mean_2: mean of vecs_1, vecs_2
    :param cov_1, cov_2: covariance of vecs_1, vecs_2
    :param dist: distance function d(vec_1, vec_2)
    :return: average_distance
    """
    vecs_1 = np.random.multivariate_normal(mean_1, cov_1, N)
    vecs_2 = np.random.multivariate_normal(mean_2, cov_2, N)
    average_distance = np.average([dist(vec_1, vec_2) for vec_1, vec_2 in zip(vecs_1, vecs_2)])

    return average_distance


def l2_norm(array_1, array_2):
    try:
        norm = np.linalg.norm(array_1 - array_2)
        return norm
    except:
        return None


def kl_div(mean_1, cov_1, mean_2, cov_2):
    """
    calculate kl divergence for multi-variate gaussian distributions

    N_1 = N(mean_1, cov_1), N_2 = N(mean_2, cov_2)
    KL(N_1 || N_2)
        = 0.5 * (log(|cov_2|/|cov_1|) + tr(cov_2^{-1} cov_1)
            + (mean_2 - mean_1).T cov_2^{-1} (mean_2 - mean_1) - bert_dim)
    
    :param mean_1, mean_2: mean of vecs_1, vecs_2
    :param cov_1, cov_2: covariance of vecs_1, vecs_2
    :return: kl
    """
    cov_2_inv = np.linalg.inv(cov_2)

    cov_1_det_log = np.linalg.slogdet(cov_1)[1]
    logging.debug(f"[kl_div] cov_1_det_log: {cov_1_det_log}")
    cov_2_det_log = np.linalg.slogdet(cov_2)[1]
    logging.debug(f"[kl_div] cov_2_det_log: {cov_2_det_log}")
    log_term = cov_2_det_log - cov_1_det_log

    tr_term = np.trace(cov_2_inv @ cov_1)
    logging.debug(f"[kl_div] tr_term: {tr_term}")
    diff_term = (mean_2 - mean_1).T @ cov_2_inv @ (mean_2 - mean_1)
    logging.debug(f"[kl_div] diff_term: {diff_term}")
     
    kl = 0.5 * (log_term + tr_term + diff_term - len(mean_1))

    return kl

def jeff_div(mean_1, cov_1, mean_2, cov_2):
    """
    calculate jeffreys' divergence for multi-variate gaussian distributions

    N_1 = N(mean_1, cov_1), N_2 = (mean_2, cov_2))
    Jeffreys(N_1 || N_2)
        = 0.5 * KL(N_1 || N_2) + 0.5 * KL(N_2 || N_1)
        = 0.25 * (tr(cov_2^{-1} cov_1) + (mean_2 - mean_1).T cov_2^{-1} (mean_2 - mean_1) 
            + tr(cov_1^{-1} cov_2) + (mean_1 - mean_2).T cov_1^{-1} (mean_1 - mean_2) - 2 * bert_dim)
        = 0.25 * (tr(cov_2^{-1} cov_1) + tr(cov_1^{-1} cov_2) - 2 * bert_dim
            + (mean_2 - mean_1).T (cov_2^{-1} + cov_1^{-1}) (mean_2 - mean_1))
    
    :param mean_1, mean_2: mean of vecs_1, vecs_2
    :param cov_1, cov_2: covariance of vecs_1, vecs_2
    :return: jeff
    """
    cov_1_inv = np.linalg.inv(cov_1)
    cov_2_inv = np.linalg.inv(cov_2)

    tr_term = np.trace(cov_2_inv @ cov_1) + np.trace(cov_1_inv @ cov_2)
    diff_term = (mean_2 - mean_1).T @ (cov_2_inv + cov_1_inv) @ (mean_2 - mean_1)

    jeff = 0.25 * (tr_term + diff_term - 2 * len(mean_1))

    return jeff

def kl_with_mcmc(mean_1, cov_1, mean_2, cov_2, N=1000):
    samples = np.random.multivariate_normal(mean_1, cov_1, N)
    probs_1 = scipy.stats.multivariate_normal.pdf(samples, mean=mean_1, cov=cov_1)
    probs_2 = scipy.stats.multivariate_normal.pdf(samples, mean=mean_2, cov=cov_2)
              
    kl = np.sum(np.log(probs_1) - np.log(probs_2)) / N

    return kl

def js_div(mean_1, cov_1, mean_2, cov_2, N=1000):
    mean_m = 0.5 * (mean_1 + mean_2)
    cov_m = 0.5 * (cov_1 + cov_2)

    kl_1_m = kl_with_mcmc(mean_1, cov_1, mean_m, cov_m, N)
    kl_2_m = kl_with_mcmc(mean_2, cov_2, mean_m, cov_m, N)
    jsd = 0.5 * (kl_1_m + kl_2_m)

    return jsd

