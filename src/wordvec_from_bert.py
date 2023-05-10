import os
import argparse
import json
import pickle
import torch
import logging
from tqdm import tqdm
from transformers import BertConfig, BertModel, BertTokenizer, TFBertModel
import tensorflow as tf

from typing import Dict, List


def collect_usage(target_word: str, tokenizer, file_path: str) -> List[str]:
    usages = []
    with open(file_path) as fp:
        for line in fp:
            sentence = line.strip()
            words = sentence.split()
            if target_word in set(words):
                usages.append(sentence)
    return usages


def make_batch(corpus: List[str], batch_size: int) -> List[List[str]]:
    """
    :param corpus: list of sentences
    :param batch_size: batch_size
    :return: samples
    """
    sorted_sentence = sorted(corpus, key=lambda x: len(x.split()))
    samples = [
        sorted_sentence[i : i + batch_size]
        if i + batch_size < len(sorted_sentence)
        else sorted_sentence[i:]
        for i in range(0, len(sorted_sentence), batch_size)
    ]
    return samples


def main(args):
    os.makedirs("../results", exist_ok=True)
    logging.basicConfig(filename="../results/wordvec_from_bert.log", format="%(asctime)s %(message)s", level=logging.INFO)
    logging.info(f"[INFO] args: {args}")

    logging.info("1. load models... ")
    if args.is_finetuned:
        logging.debug(" - load tokenizer")
        tokenizer = BertTokenizer(tokenizerfile=args.finetuned_tokenizer, 
                                  vocab_file=args.finetuned_vocab)
        logging.debug(" - load vocab")
        config = BertConfig.from_json_file(args.finetuned_config)
        logging.debug(" - load model")
        model = TFBertModel.from_pretrained(args.finetuned_model, from_pt=True, config=config)
        if args.finetuned_added_tokens is not None:
            logging.debug(" - add tokens")
            with open(args.finetuned_added_tokens, "r") as fp:
                added_tokens_dict: Dict[str, int] = json.load(fp)
            sorted_ids = sorted(added_tokens_dict.values())
            id2tokens = {token_id: token for token, token_id in added_tokens_dict.items()}
            added_tokens_list = [id2tokens[sorted_id] for sorted_id in sorted_ids]
            logging.debug(f" - add tokens (before): {len(tokenizer)} words")
            logging.debug(f" - add tokens : {len(added_tokens_list)} words")
            tokenizer.add_tokens(added_tokens_list)
            logging.debug(f" - add tokens (after): {len(tokenizer)} words")
            model.resize_token_embeddings(len(tokenizer))
            logging.debug(f" - added token: {tokenizer.convert_ids_to_tokens([len(tokenizer)-1])}")

    else:
        model = BertModel.from_pretrained("bert-base-uncased", return_dict=True)
        model.to("cpu")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    logging.info("2. encode target words...")
    target_words = []
    with open(args.target_words_list) as fp:
        for line in fp:
            target_word = line.strip()
            target_words.append(target_word)
    targetword2bertid = {}
    for target_word in target_words:
        # [0]: 2(SOS), [-1]: 3(EOS)
        bertids = tokenizer.encode(target_word)[1:-1]
        targetword2bertid[target_word] = bertids
        logging.debug(f" - {target_word}: {bertids}")

    logging.info("3. obtain target vectors...")
    word2vec = {}
    with torch.no_grad():
        for target_word in tqdm(target_words):
            word2vec[target_word] = []
            target_bertids: List[int] = targetword2bertid[target_word]
            target_sentences: List[str] = collect_usage(target_word, tokenizer, args.file_path)
            logging.debug(f" - {target_word}: {len(target_sentences)} sents")

            samples: List[List[str]] = make_batch(target_sentences, batch_size=16)
            for sample in tqdm(samples):
                if args.is_finetuned:
                    #inputs = tokenizer(sample, return_tensors="tf", padding=True, truncation=True)
                    # Tempobert has no max_length, so we need to define
                    inputs = tokenizer(sample, return_tensors="tf", padding=True, truncation=True, max_length=512)
                    vectors = model(**inputs, output_hidden_states=True)
                    # average pooling
                    vectors = vectors.hidden_states[-args.hidden_layers:]
                    vectors = tf.reduce_sum(tf.stack(vectors), 0) / args.hidden_layers
                    # tf -> np -> torch
                    vectors = torch.from_numpy(vectors.numpy())
                else:
                    inputs = tokenizer(sample, return_tensors="pt", padding=True, truncation=True)#.to("cpu")
                    vectors = model(**inputs, output_hidden_states=True)#.to("cpu")
                    # average pooling
                    vectors = vectors.hidden_states[-args.hidden_layers:]
                    vectors = torch.sum(torch.stack(vectors), 0) / args.hidden_layers
                
                del inputs

                for sentence_id in range(len(sample)):
                    tokens = tokenizer.encode(sample[sentence_id], padding=True, truncation=True)
                    if len(tokens) < len(target_bertids):
                        continue
                    for position_id in range(len(tokens)-len(target_bertids)+1):
                        seq_tokens = tokens[position_id:position_id+len(target_bertids)]
                        if seq_tokens == target_bertids:
                            # (ONLY SemEval2020 Task1) exclude "_nn", "_vb" from target word vectors
                            if "_" in target_word:
                                target_vector = torch.sum(vectors[sentence_id][position_id:position_id+len(target_bertids)-2], axis=0)
                            else:
                                target_vector = torch.sum(vectors[sentence_id][position_id:position_id+len(target_bertids)], axis=0)
                            word2vec[target_word].append(target_vector)

            logging.debug(f"   - {len(word2vec[target_word])} items")

    pickle.dump(word2vec, open(f"../results/bert_{args.output_name}_word2vec.pkl", "wb"))

    logging.info("[INFO] finished")


def cli_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path", help="path of corpus")
    parser.add_argument("-l", "--target_words_list", help="target word list")
    parser.add_argument("-o", "--output_name")
    parser.add_argument("--hidden_layers", type=int, default=1, help="number of layers. default(1) uses last hidden states only.")
    parser.add_argument("--is_finetuned", action="store_true", help="use finetuned model or not")
    parser.add_argument("--finetuned_tokenizer", help="(finetuned tempobert) path of tokenizer.json")
    parser.add_argument("--finetuned_added_tokens", help="(finetuned tempobert) path of added_tokens.json")
    parser.add_argument("--finetuned_vocab", help="(finetuned tempobert) path of vocab.txt")
    parser.add_argument("--finetuned_config", help="(finetuned tempobert) path of config.json")
    parser.add_argument("--finetuned_model", help="(finetuned tempobert) path of pytorch_model.bin")
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
