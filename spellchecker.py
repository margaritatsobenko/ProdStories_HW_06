import argparse
from typing import List

import numpy as np
from pyphonetics import Soundex
from spylls.hunspell import Dictionary
from textdistance import damerau_levenshtein, hamming


def get_candidates(word: str) -> List[str]:
    return list(vocabulary.suggester.ngram_suggestions(word, set()))


def calculate_features(target: str, candidates: List[str]) -> List[List[float]]:
    soundex = Soundex()

    result = []
    for candidate in candidates:
        result.append([damerau_levenshtein.normalized_distance(target, candidate),
                       hamming.normalized_distance(target, candidate),
                       damerau_levenshtein.normalized_distance(soundex.phonetics(target),
                                                               soundex.phonetics(candidate))])
    return result


def suggest(word: str, num_candidates: int = 3) -> List[str]:
    candidates = get_candidates(word)
    feature_values = calculate_features(word, candidates)

    ranks = np.array([np.mean(features) for features in feature_values])
    return np.array(candidates)[ranks.argsort()][:num_candidates]


if __name__ == '__main__':
    vocabulary = Dictionary.from_files("en_US")

    parser = argparse.ArgumentParser()
    parser.add_argument("--word", type=str, default="sunkine",
                        help="word")
    parser.add_argument("--num_candidates", type=int, default=3,
                        help="number of candidates")

    args = parser.parse_args()
    word = args.word
    num_candidates = args.num_candidates

    if vocabulary.lookup(word):
        print(f"{word} is already correctly spelt")
    else:
        print(f"Candidates: {suggest(word, num_candidates)}")
