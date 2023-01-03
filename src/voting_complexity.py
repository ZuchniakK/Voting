import time

import numpy as np
from src.voting_methods import VotingPreferences


def test_voting_complexity(candidates_list=None, voters_n=5, trials_n=5, methods=None):
    if candidates_list is None:
        candidates_list = [i for i in range(2, 15, 2)]
    if methods is None:
        methods = [
            "plurality",
            "copeland",
            "borda",
            "symetric_borda",
            "simpson",
            "dowdall",
            "baldwin",
            "nanson",
            "bucklin",
            "stv",
            "softmax",
        ]

    results_dict = {}
    for method in methods:
        results_dict[method] = {
            "name": method,
            "candidates": candidates_list,
            "time": [0 for i in range(len(candidates_list))],
        }

    for _ in range(trials_n):
        for idx, candidate_number in enumerate(candidates_list):
            vote_matrix = np.random.uniform(0, 1, [voters_n, candidate_number])
            vot_pref = VotingPreferences(vote_matrix, ground_truth=5)

            if "plurality" in methods:
                start = time.time()
                vot_pref.plurality_winner
                results_dict["plurality"]["time"][idx] += time.time() - start

            if "copeland" in methods:
                start = time.time()
                vot_pref.copeland_winner
                results_dict["copeland"]["time"][idx] += time.time() - start

            if "borda" in methods:
                start = time.time()
                vot_pref.borda_winner
                results_dict["borda"]["time"][idx] += time.time() - start

            if "symetric_borda" in methods:
                start = time.time()
                vot_pref.symetric_borda_winner
                results_dict["symetric_borda"]["time"][idx] += time.time() - start

            if "simpson" in methods:
                start = time.time()
                vot_pref.simpson_winner
                results_dict["simpson"]["time"][idx] += time.time() - start

            if "dowdall" in methods:
                start = time.time()
                vot_pref.dowdall_winner
                results_dict["dowdall"]["time"][idx] += time.time() - start

            if "baldwin" in methods:
                start = time.time()
                vot_pref.baldwin_winner
                results_dict["baldwin"]["time"][idx] += time.time() - start

            if "nanson" in methods:
                start = time.time()
                vot_pref.nanson_winner
                results_dict["nanson"]["time"][idx] += time.time() - start

            if "bucklin" in methods:
                start = time.time()
                vot_pref.bucklin_winner
                results_dict["bucklin"]["time"][idx] += time.time() - start

            if "stv" in methods:
                start = time.time()
                vot_pref.stv_winner
                results_dict["stv"]["time"][idx] += time.time() - start

            if "softmax" in methods:
                start = time.time()
                vot_pref.softmax_winner
                results_dict["softmax"]["time"][idx] += time.time() - start

    return results_dict


if __name__ == "__main__":
    test_voting_complexity()