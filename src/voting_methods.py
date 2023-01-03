import pickle
from functools import wraps
from itertools import combinations, permutations

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import scipy.stats


def save_obj(obj, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)


def similarity_matrix(df):
    prediction_collumns = []
    samples, predictions = 0, 0
    for collumn_name in list(df):
        if "true_label" in collumn_name:
            prediction_collumns.append(collumn_name)
            samples += 1
        elif "prediction" in collumn_name:
            prediction_collumns.append(collumn_name)
            predictions += 1

    if predictions % samples:
        raise ValueError("number of predictions per sample must be the same")

    class_number = predictions // samples
    print(f"sample number: {samples} class_number: {class_number}")
    probabilities = df[df.columns.intersection(prediction_collumns)]

    predictions = []

    for index, row in probabilities.iterrows():
        prediction_list = list(row.values)
        predictions.append(
            [
                prediction_list[i * class_number : (i + 1) * class_number].index(
                    max(prediction_list[i * class_number : (i + 1) * class_number])
                )
                for i in range(samples)
            ]
        )

    similarity = []
    for i in range(probabilities.shape[0]):
        print(i)
        similarity.append([])
        for j in range(probabilities.shape[0]):
            similarity_value = (
                sum(list(map(lambda a, b: a == b, predictions[i], predictions[j])))
                / samples
            )
            similarity[i].append(similarity_value)
    labels = [str(i + 1) for i in range(probabilities.shape[0])]
    print(similarity)

    return pd.DataFrame(similarity, index=labels, columns=labels)


class VotingPreferences:
    def __init__(self, vote_matrix, ground_truth=None, use_ranks=False):
        self.vote_matrix = vote_matrix
        self.ground_truth = ground_truth

        if use_ranks:
            self.ranks = vote_matrix
        else:
            reverse_order = np.argsort(vote_matrix)
            self.ranks = np.fliplr(reverse_order)

        self.n_voters = vote_matrix.shape[0]
        self.n_candidates = vote_matrix.shape[1]

        self._kendalltau_matrix = None
        self._kemanny_rank = None
        self._candidate_softmax_sum = None
        self._softmax_dict = None
        self._preference_matrix = None
        self._path_preference_graph = None

        self._borda_score = None
        self._symetric_borda_score = None
        self._plurality_score = None
        self._dowdall_score = None
        self._copeland_score = None
        self._simpson_score = None
        self._schulze_score = None

        self._softmax_winner = None

        self._borda_winner = None
        self._symetric_borda_winner = None
        self._plurality_winner = None
        self._dowdall_winner = None
        self._copeland_winner = None
        self._simpson_winner = None
        self._schulze_winner = None

        self._bucklin_winner = None
        self._baldwin_winner = None
        self._nanson_winner = None

        self._stv_winner = None

        self._condorcet_winner = None

    @staticmethod
    def count_kendalltau_dist(rank_a, rank_b):
        """
        number of flips you need
        to perform on a ranking to turn it into the other,
        it is sometimes called bubble-sort distance
        """
        tau = 0
        n_candidates = len(rank_a)
        for i, j in combinations(range(n_candidates), 2):
            tau += np.sign(rank_a[i] - rank_a[j]) == -np.sign(rank_b[i] - rank_b[j])
        return tau

    @property
    def kendalltau_matrix(self):
        if self._kendalltau_matrix is None:
            self.count_kendalltau_matrix()
        return self._kendalltau_matrix

    def count_kendalltau_matrix(self):
        """
        cound distance matrix between all voters
        """
        distance_matrix = np.zeros((self.n_voters, self.n_voters))
        for voter_a in range(self.n_voters):
            for voter_b in range(voter_a, self.n_voters):
                dist = self.count_kendalltau_dist(
                    self.ranks[voter_a], self.ranks[voter_b]
                )
                distance_matrix[voter_a, voter_b] = dist
                distance_matrix[voter_b, voter_a] = dist
        self._kendalltau_matrix = distance_matrix

    @property
    def kemanny_rank(self):
        if self._kemanny_rank is None:
            self.kemanny_rankaggr_brute()
        return self._kemanny_rank

    def kemanny_rankaggr_brute(self):
        # TODO implement np from http://vene.ro/blog/kemeny-young-optimal-rank-aggregation-in-python.html
        print("Kemanny rank not supported yet")
        return 0
        min_dist = np.inf
        best_rank = None
        for candidate_rank in permutations(range(self.n_candidates)):
            dist = np.sum(
                self.count_kendalltau_dist(candidate_rank, rank) for rank in self.ranks
            )
            if dist < min_dist:
                min_dist = dist
                best_rank = candidate_rank
        self._kemanny_rank = (best_rank, min_dist)

    def _build_graph(self):
        # for kemany aggregation
        edge_weights = np.zeros((self.n_candidates, self.n_candidates))
        for i, j in combinations(range(self.n_candidates), 2):
            preference = self.ranks[:, i] - self.ranks[:, j]
            h_ij = np.sum(preference < 0)  # prefers i to j
            h_ji = np.sum(preference > 0)  # prefers j to i
            if h_ij > h_ji:
                edge_weights[i, j] = h_ij - h_ji
            elif h_ij < h_ji:
                edge_weights[j, i] = h_ji - h_ij
        return edge_weights

    def does_pareto_dominate(self, candidate1, candidate2):
        """
        Returns True when candidate1 is preferred in all ballots.
        False, otherwise.
        """
        # A boolean list as candidate1 preferred
        preferred = [
            np.where(rank == candidate1)[0] < np.where(rank == candidate2)[0]
            for rank in self.ranks
        ]
        return all(preferred)

    def condorcet_winners(self):
        """Calculate the Condorcet Winners and returns a list of winner candidates"""
        winners = []  # list of winners
        for candidate in range(self.n_candidates):
            # If mayor beats everyone, condorcet_condition is True
            condorcet_condition = all(
                self.preference_matrix[candidate, opponent] >= 0
                for opponent in range(self.n_candidates)
            )
            # If True, mayor is a winner
            if condorcet_condition:
                winners.append(candidate)
        return winners

    @property
    def condorcet_winner(self):
        if self._condorcet_winner is None:
            self.find_condorcet_winner()
        return self._condorcet_winner

    def find_condorcet_winner(self):
        possible_winners = self.condorcet_winners()
        if len(possible_winners) == 0:
            self._condorcet_winner = max(
                self.candidate_softmax_sum, key=lambda x: x[1]
            )[0]
        elif len(possible_winners) > 0:
            filtered_softmax = [
                i for i in self.candidate_softmax_sum if i[0] in possible_winners
            ]
            self._condorcet_winner = max(filtered_softmax, key=lambda x: x[1])[0]
        else:
            self._condorcet_winner = possible_winners[0]

    @property
    def preference_matrix(self):
        if self._preference_matrix is None:
            self.count_preference_matrix()
        return self._preference_matrix

    def count_preference_matrix(self):
        preference_matrix = np.zeros((self.n_candidates, self.n_candidates))
        for i in range(self.n_candidates):
            for j in range(i, self.n_candidates):
                preferences = 0
                for rank in self.ranks:
                    preferences += np.sign(
                        np.where(rank == j)[0] - np.where(rank == i)[0]
                    )

                preference_matrix[i, j] = preferences
                preference_matrix[j, i] = -preferences

        self._preference_matrix = preference_matrix

    def _make_ranking(f):
        @wraps(f)
        def wrapped(inst, *args, **kwargs):
            result = [(i, value) for i, value in enumerate(f(inst, *args, **kwargs))]
            result.sort(key=lambda x: x[1], reverse=True)
            return result

        return wrapped

    @property
    @_make_ranking
    def candidate_softmax_sum(self):
        if self._candidate_softmax_sum is None:
            self.count_candidate_softmax_sum()
        return self._candidate_softmax_sum

    @property
    def softmax_winner(self):
        if self._softmax_winner is None:
            self._softmax_winner = max(self.candidate_softmax_sum, key=lambda x: x[1])[
                0
            ]
        return self._softmax_winner

    def count_candidate_softmax_sum(self):
        candidate_points = [0 for i in range(self.n_candidates)]
        for softmax_output in self.vote_matrix:
            for i in range(self.n_candidates):
                candidate_points[i] += softmax_output[i]
        self._candidate_softmax_sum = candidate_points

    @property
    @_make_ranking
    def borda_score(self):
        if self._borda_score is None:
            self.count_borda_score()
        return self._borda_score

    @property
    @_make_ranking
    def plurality_score(self):
        if self._plurality_score is None:
            self.count_plurality_score()
        return self._plurality_score

    @property
    @_make_ranking
    def symetric_borda_score(self):
        if self._symetric_borda_score is None:
            self.count_symetric_borda_score()
        return self._symetric_borda_score

    @property
    @_make_ranking
    def dowdall_score(self):
        if self._dowdall_score is None:
            self.count_dowdall_score()
        return self._dowdall_score

    @property
    @_make_ranking
    def copeland_score(self):
        if self._copeland_score is None:
            self.count_copeland_score()
        return self._copeland_score

    @property
    @_make_ranking
    def simpson_score(self):
        if self._simpson_score is None:
            self.count_simpson_score()
        return self._simpson_score

    @property
    def borda_winner(self):
        if self._borda_winner is None:
            self._borda_winner = self.strongest_candidate(self.borda_score)
        return self._borda_winner

    @property
    def symetric_borda_winner(self):
        if self._symetric_borda_winner is None:
            self._symetric_borda_winner = self.strongest_candidate(
                self.symetric_borda_score
            )
        return self._symetric_borda_winner

    @property
    def plurality_winner(self):
        if self._plurality_winner is None:
            self._plurality_winner = self.strongest_candidate(self.plurality_score)
        return self._plurality_winner

    @property
    def dowdall_winner(self):
        if self._dowdall_winner is None:
            self._dowdall_winner = self.strongest_candidate(self.dowdall_score)
        return self._dowdall_winner

    @property
    def copeland_winner(self):
        if self._copeland_winner is None:
            self._copeland_winner = self.strongest_candidate(self.copeland_score)
        return self._copeland_winner

    @property
    def simpson_winner(self):
        if self._simpson_winner is None:
            self._simpson_winner = self.strongest_candidate(self.simpson_score)
        return self._simpson_winner

    @property
    def schulze_winner(self):
        if self._schulze_winner is None:
            self._schulze_winner = self.strongest_candidate(self.schulze_score)
        return self._schulze_winner

    def strongest_candidate(self, ranking):
        strong_indicies = []
        strongest = max(ranking, key=lambda x: x[1])[0]
        for index, value in ranking:
            if strongest == index:
                strong_indicies.append(index)
        if len(strong_indicies) == 1:
            return strong_indicies[0]
        else:
            filtered_softmax = [
                i for i in self.candidate_softmax_sum if i[0] in strong_indicies
            ]
            return max(filtered_softmax, key=lambda x: x[1])[0]

    def count_borda_score(self):
        scoring_vector = [self.n_candidates - i - 1 for i in range(self.n_candidates)]
        ranking = self.vector_scoring_system(scoring_vector)
        self._borda_score = ranking

    def count_plurality_score(self):
        scoring_vector = [0 for i in range(self.n_candidates)]
        scoring_vector[0] = 1
        ranking = self.vector_scoring_system(scoring_vector)
        self._plurality_score = ranking

    def count_dowdall_score(self):
        scoring_vector = [1 / (i + 1) for i in range(self.n_candidates)]
        ranking = self.vector_scoring_system(scoring_vector)
        self._dowdall_score = ranking

    def vector_scoring_system(self, scoring_vector):
        candidate_points = [0 for i in range(self.n_candidates)]
        for rank in self.ranks:
            for pos, score in zip(rank, scoring_vector):
                candidate_points[pos] += score
        return candidate_points

    def count_symetric_borda_score(self):
        self._symetric_borda_score = [
            sum(
                [
                    self.preference_matrix[candidate, opponent]
                    for opponent in range(self.n_candidates)
                ]
            )
            for candidate in range(self.n_candidates)
        ]

    def count_copeland_score(self):

        self._copeland_score = [
            sum(
                [
                    np.sign(self.preference_matrix[candidate, opponent])
                    for opponent in range(self.n_candidates)
                ]
            )
            for candidate in range(self.n_candidates)
        ]

    def count_simpson_score(self):

        """
        Calculate the Simpson score also known as minimax method
        """
        self._simpson_score = [
            min(
                [
                    self.preference_matrix[candidate, opponent]
                    for opponent in range(self.n_candidates)
                    if not candidate == opponent
                ]
            )
            for candidate in range(self.n_candidates)
        ]

    @property
    def stv_winner(self):
        if self._stv_winner is None:
            self.single_transferable_vote()
        return self._stv_winner

    @property
    def bucklin_winner(self):
        if self._bucklin_winner is None:
            self.find_bucklin_winner()
        return self._bucklin_winner

    def find_bucklin_winner(self, mode="linear"):
        if mode == "linear":
            vector_score = [1 for i in range(self.n_candidates)]
        if mode == "borda":
            vector_score = [
                (self.n_candidates - i) / self.n_candidates
                for i in range(self.n_candidates)
            ]

        majority_treshold = self.n_voters / 2
        candidate_points = [0 for i in range(self.n_candidates)]
        candidates_with_majority = []

        for position in range(self.n_candidates):
            for vote in range(self.n_voters):
                candidate = self.ranks[vote][position]
                candidate_points[candidate] += vector_score[position]

            candidates_with_majority = [
                i for i, j in enumerate(candidate_points) if j > majority_treshold
            ]
            if len(candidates_with_majority) > 0:
                break
        if len(candidates_with_majority) > 1:
            filtered_softmax = [
                i
                for i in self.candidate_softmax_sum
                if i[0] in candidates_with_majority
            ]
            self._bucklin_winner = max(filtered_softmax, key=lambda x: x[1])[0]
        self._bucklin_winner = candidates_with_majority[0]

    @property
    def baldwin_winner(self):
        if self._baldwin_winner is None:
            self.find_baldwin_winner()
        return self._baldwin_winner

    def find_baldwin_winner(self):
        ranking = self.borda_score
        winners = [i for i in range(self.n_candidates)]

        while len(winners) > 1:
            weakest = self.weakest_candidate(ranking)
            winners.remove(weakest)
            new_ranks = self.new_order(winners)
            ranking = self.custom_vector_scoring_system(
                self.get_borda_score(len(winners)), new_ranks, winners
            )
        self._baldwin_winner = winners[0]

    @property
    def nanson_winner(self):
        if self._nanson_winner is None:
            self.find_nanson_winner()
        return self._nanson_winner

    def find_nanson_winner(self):
        ranking = self.borda_score
        winners = [i for i in range(self.n_candidates)]
        while len(winners) > 1:
            candidates, scores = zip(*ranking)
            borda_avg = sum(scores) / len(ranking)
            remove_counter = 0
            for i in range(len(ranking)):
                candidate, score = ranking[i]
                if score < borda_avg:
                    winners.remove(candidate)
                    remove_counter += 1
            if not remove_counter:
                break

            new_ranks = self.new_order(winners)
            ranking = self.custom_vector_scoring_system(
                self.get_borda_score(len(winners)), new_ranks, winners
            )
        if len(winners) > 1:
            filtered_softmax = [
                i for i in self.candidate_softmax_sum if i[0] in winners
            ]
            self._nanson_winner = max(filtered_softmax, key=lambda x: x[1])[0]
        self._nanson_winner = winners[0]

    def custom_vector_scoring_system(self, scoring_vector, custom_ranks, candidates):
        candidate_points = [0 for i in range(len(custom_ranks[0]))]
        for rank in custom_ranks:
            for pos, score in zip(rank, scoring_vector):
                candidate_points[candidates.index(pos)] += score
        result = [(i, j) for i, j in zip(candidates, candidate_points)]
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def get_borda_score(self, n):
        return [n - i - 1 for i in range(n)]

    def new_order(self, candidate_list):
        new = []
        for rank in self.ranks:
            new_rank = [i for i in rank if i in candidate_list]
            new.append(new_rank)
        return new

    def weakest_candidate(self, ranking):
        weak_indicies = []
        weakest = min(ranking, key=lambda x: x[1])[0]
        for index, value in ranking:
            if weakest == index:
                weak_indicies.append(index)
        if len(weak_indicies) == 1:
            return weak_indicies[0]
        else:
            filtered_softmax = [
                i for i in self.candidate_softmax_sum if i[0] in weak_indicies
            ]
            return min(filtered_softmax, key=lambda x: x[1])[0]

    def locate_min(a):
        min_indicies = []
        smallest = min(a)
        for index, element in enumerate(a):
            if smallest == element:  # check if this element is the minimum_value
                min_indicies.append(index)  # add the index to the list if it is

        return smallest, min_indicies

    @property
    def softmax_dict(self):
        if self._softmax_dict is None:
            self.get_dict_softmax_sum()
        return self._softmax_dict

    def get_dict_softmax_sum(self):
        new_dict = {}
        for i, j in self.candidate_softmax_sum:
            new_dict[i] = j
        self._softmax_dict = new_dict

    def single_transferable_vote(self, n=1):
        """Calculate the (n) winners using Single Tranferable Voting System"""
        candidate_points = [0 for i in range(self.n_candidates)]
        options = {i: 0 for i in range(self.n_candidates)}

        while len(options) > n:
            # count votes
            for rank in self.ranks:
                for choice in rank:
                    if choice in options:
                        options[choice] += 1
                        break
            fewest_votes = min(options.values())
            candidate_sums_selected = []
            for op in options:
                if options[op] == fewest_votes:
                    candidate_sums_selected.append(self.softmax_dict[op])
            options_copy = dict(options)  # shallow copy
            minimum_sum = min(candidate_sums_selected)
            for i in options:
                if options[i] == fewest_votes and self.softmax_dict[i] == minimum_sum:
                    del options_copy[i]
                    break

            options = options_copy

            # Reset votes to 0
            for i in options:
                options[i] = 0
        self._stv_winner = list(options.keys())[0]

    @property
    def schulze_score(self):
        if self._schulze_score is None:
            self.count_schulze_score()
        return self._schulze_score

    def count_schulze_score(self):
        #
        # TODO finish it
        score = []
        for candidate in range(self.n_candidates):
            wins = []

            for opponent in [i for i in range(self.n_candidates) if not i == candidate]:
                candidate_strength = self.path_preference_graph[candidate, opponent]
                opponent_strength = self.path_preference_graph[opponent, candidate]

                win_or_defeat = int(candidate_strength > opponent_strength)
                wins.append(win_or_defeat)
            score.append(sum(wins))
        self._schulze_score = score

    @property
    def path_preference_graph(self):
        if self._path_preference_graph is None:
            self._calc_path_preference()
        return self._path_preference_graph

    def _calc_path_preference(self):
        """Calculate paths' strengths for Schulze method."""

        path_preference_graph = np.zeros((self.n_candidates, self.n_candidates))
        for i in range(self.n_candidates):

            for j in range(i + 1, self.n_candidates):
                # Get strengths
                strength1 = self.__calc_strength(i, j)  # candidate1 VS candidate2
                strength2 = self.__calc_strength(j, i)  # candidate2 VS candidate1

                # Save strengths
                self.path_preference_graph[i][j] = strength1
                self.path_preference_graph[j][i] = strength2

        self._path_preference_graph = path_preference_graph

    def __calc_strength(self, candidate1, candidat2):
        """Return the weakest link of the strongest path."""
        # Find possible paths between mayor1 and mayor2
        paths = self.__calc_paths(candidate1, candidat2)

        # Get strength for each path (weakest link)
        strength = list(map(lambda x: min(x), paths))

        # Return the strongest strength
        return max(strength)

    def __calc_paths(self, candidate1, candidat2, candidates=None):
        """Find the possible paths between candidate1 and candidat2"""
        if candidates is None:
            candidates = [i for i in range(self.n_candidates) if not i == candidate1]

        n_candidates = len(candidates)  # number of mayors
        paths = []  # list of possible paths
        path = []  # list of weights

        for candidate in candidates:

            # Get preference of mayor1 over mayor
            preference = self.preference_matrix[candidate1, candidate]
            path.append(preference)  # save current weigth

            if candidate == candidat2:
                paths.append(path)  # add to possible paths
                path = []  # start a new path
            else:  # path isn't over
                new_candidates = [i for i in candidates if not i == candidate]
                subpath = self.__calc_paths(candidate, candidat2, new_candidates)

                # For each subpath (list of weights),
                # concatenate with current path and save it
                for weights in subpath:
                    paths.append(path + weights)

        # Return a list of possible paths between mayor1 and mayor2
        return paths


class DataAnalyser:
    def __init__(self, file_path):
        self.prepare_data(file_path)

    def change_source(self, file_path):
        self.prepare_data(file_path)

    def prepare_data(self, file_path):
        """
        convert file_path csv file with full dataframe from training to dataframe with only true_label_n and prediction_n_0-k collumns,
        return converted_dataframe, dict with sample_number and class_number
        """
        self.df_dict = {}
        if isinstance(file_path, pd.DataFrame):
            df = file_path
        else:
            df = pd.read_csv(file_path, delimiter=",")
            if df.shape[1] == 1:
                df = pd.read_csv(file_path, delimiter=";")
            if df.shape[1] == 1:
                raise NotImplementedError(f"delimiter could be only , or ;")

        is_last_line_bad = "Unnamed" in df.iloc[:, -1].name
        print(f"is_last_line_bad {is_last_line_bad}")
        print(f"df.shape {df.shape}")
        if is_last_line_bad:
            df = df.drop(labels=df.iloc[:, -1].name, axis=1)
        is_last_line_bad = "Unnamed" in df.iloc[:, -1].name
        print(f"is_last_line_bad {is_last_line_bad}")
        print(f"df.shape {df.shape}")

        try:
            self.df_dict["accuracy"] = df.loc[:, "accuracy"].values
        except:
            print("not found accuracy collumn")

        # iterate over collumn names
        prediction_collumns = []
        samples, predictions = 0, 0
        for collumn_name in list(df):
            if "true_label" in collumn_name:
                prediction_collumns.append(collumn_name)
                samples += 1
            elif "prediction" in collumn_name:
                prediction_collumns.append(collumn_name)
                predictions += 1

        if predictions % samples:
            raise ValueError("number of predictions per sample must be the same")

        class_number = predictions // samples
        print(f"sample number: {samples} class_number: {class_number}")

        self.df = df[df.columns.intersection(prediction_collumns)]
        self.df_dict["samples"] = samples
        self.df_dict["class_number"] = class_number

    def applay_analyse_schedule(self, schedule, plot_path, dict_path, plot_title):
        self.analyze_voting(
            size_list=schedule["voting"]["size_list"],
            number_of_test=schedule["voting"]["number_of_test"],
            max_sample_per_test=schedule["voting"]["max_sample_per_test"],
            methods=schedule["voting"]["methods"],
        )

        self.analyze_vote_distribution(
            size_list=schedule["distribution"]["size_list"],
            number_of_test=schedule["distribution"]["number_of_test"],
            max_sample_per_test=schedule["distribution"]["max_sample_per_test"],
            confidence=schedule["distribution"]["confidence"],
        )

        self.analyze_count_condorcet_winers(
            size_list=schedule["condorcet"]["size_list"],
            number_of_test=schedule["condorcet"]["number_of_test"],
            max_sample_per_test=schedule["condorcet"]["max_sample_per_test"],
        )

        self.save_datadict(dict_path)
        self.plot_voting_analyze(plot_path, plot_title)

    def vote_matrix_generator(self, voting_group_size=10):
        """
        count vote distribution, return list of size class_number, each value represents avg number of diffrient candidates
        per single voting, normalize to number of voters.
        """
        counts_list = []
        while True:
            df_sample = self.df.sample(n=voting_group_size, replace=False)
            for i in range(self.df_dict["samples"]):
                ground_true = df_sample.iloc[0, i * (self.df_dict["class_number"] + 1)]
                vote_matrix = df_sample.iloc[
                    :,
                    i * (self.df_dict["class_number"] + 1)
                    + 1 : i * (self.df_dict["class_number"] + 1)
                    + 1
                    + self.df_dict["class_number"],
                ]
                vote_matrix = vote_matrix.values

                yield (vote_matrix, ground_true)

    def analyze_voting(
        self, size_list, number_of_test, methods=None, max_sample_per_test=100
    ):
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
        self.df_dict["voting"] = {}
        self.df_dict["voting"]["size_list"] = size_list
        self.df_dict["voting"]["methods"] = methods
        self.df_dict["voting"]["number_of_test"] = number_of_test
        self.df_dict["voting"]["max_sample_per_test"] = max_sample_per_test

        voting_results = {i: [] for i in methods}
        for size in size_list:
            result = self.aplay_voting(
                voting_group_size=size,
                number_of_test=number_of_test,
                max_sample_per_test=max_sample_per_test,
                methods=methods,
            )

            for key, value in result.items():
                voting_results[key].append(value)

        self.df_dict["voting"]["method_results"] = voting_results

    def aplay_voting(
        self,
        voting_group_size=10,
        number_of_test=1,
        max_sample_per_test=None,
        methods=None,
        selection_epoch=None,
    ):
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

        voting_results = {i: [] for i in methods}
        test_range = (
            min(self.df_dict["samples"], max_sample_per_test)
            if max_sample_per_test
            else self.df_dict["samples"]
        )
        for voters_set in range(number_of_test):
            if selection_epoch is None:
                df_sample = self.df.sample(n=voting_group_size, replace=False)
            else:
                if selection_epoch == 100:
                    selection_epoch = 0
                df_sample = self.df[
                    self.df.index.map(lambda x: (x + 1) % 100 == selection_epoch)
                ].sample(n=voting_group_size, replace=False)
            voting_dict = {i: [] for i in methods}
            for i in range(test_range):

                ground_true = df_sample.iloc[0, i * (self.df_dict["class_number"] + 1)]
                vote_matrix = df_sample.iloc[
                    :,
                    i * (self.df_dict["class_number"] + 1)
                    + 1 : i * (self.df_dict["class_number"] + 1)
                    + 1
                    + self.df_dict["class_number"],
                ]
                vote_matrix = vote_matrix.values

                vot_pref = VotingPreferences(vote_matrix, ground_truth=ground_true)

                if "plurality" in methods:
                    voting_dict["plurality"].append(
                        vot_pref.plurality_winner == ground_true
                    )

                if "copeland" in methods:
                    voting_dict["copeland"].append(
                        vot_pref.copeland_winner == ground_true
                    )

                if "borda" in methods:
                    voting_dict["borda"].append(vot_pref.borda_winner == ground_true)

                if "symetric_borda" in methods:
                    voting_dict["symetric_borda"].append(
                        vot_pref.symetric_borda_winner == ground_true
                    )

                if "simpson" in methods:
                    voting_dict["simpson"].append(
                        vot_pref.simpson_winner == ground_true
                    )

                if "dowdall" in methods:
                    voting_dict["dowdall"].append(
                        vot_pref.dowdall_winner == ground_true
                    )

                if "baldwin" in methods:
                    voting_dict["baldwin"].append(
                        vot_pref.baldwin_winner == ground_true
                    )

                if "nanson" in methods:
                    voting_dict["nanson"].append(vot_pref.nanson_winner == ground_true)

                if "bucklin" in methods:
                    voting_dict["bucklin"].append(
                        vot_pref.bucklin_winner == ground_true
                    )

                if "stv" in methods:
                    voting_dict["stv"].append(vot_pref.stv_winner == ground_true)

                if "softmax" in methods:
                    voting_dict["softmax"].append(
                        vot_pref.softmax_winner == ground_true
                    )

            for key, value in voting_results.items():
                voting_results[key].append(
                    sum(voting_dict[key]) / len(voting_dict[key])
                )

        voting_accuracy = {}
        for key, value in voting_results.items():
            voting_accuracy[key] = sum(voting_results[key]) / len(voting_results[key])

        return voting_accuracy

    def analyze_vote_distribution(
        self, size_list, number_of_test, confidence=0.95, max_sample_per_test=100
    ):
        self.df_dict["vote_distribution"] = {}
        self.df_dict["vote_distribution"]["size_list"] = size_list
        self.df_dict["vote_distribution"]["confidence_level"] = confidence
        distribution_list = []
        std_errors, confidence_ranges = [], []
        for size in size_list:
            distribution, se, h = self.check_vote_distribution(
                voting_group_size=size,
                number_of_test=number_of_test,
                confidence=confidence,
                max_sample_per_test=max_sample_per_test,
            )
            distribution_list.append(distribution)
            std_errors.append(se)
            confidence_ranges.append(h)

        self.df_dict["vote_distribution"]["values"] = distribution_list
        self.df_dict["vote_distribution"]["std_errors"] = std_errors
        self.df_dict["vote_distribution"]["confidence_ranges"] = confidence_ranges

    def check_vote_distribution(
        self,
        voting_group_size=10,
        number_of_test=1,
        confidence=0.95,
        max_sample_per_test=100,
    ):
        """
        count vote distribution, return list of size class_number, each value represents avg number of diffrient candidates
        per single voting, normalize to number of voters.
        """
        counts_list = []
        for voters_set in range(number_of_test):
            df_sample = self.df.sample(n=voting_group_size, replace=False)
            for i in range(min(self.df_dict["samples"], max_sample_per_test)):
                ground_true = df_sample.iloc[0, i * (self.df_dict["class_number"] + 1)]
                vote_matrix = df_sample.iloc[
                    :,
                    i * (self.df_dict["class_number"] + 1)
                    + 1 : i * (self.df_dict["class_number"] + 1)
                    + 1
                    + self.df_dict["class_number"],
                ]
                vote_matrix = vote_matrix.values
                reverse_order = np.argsort(vote_matrix)
                order = np.fliplr(reverse_order)
                order = order.T
                counts = [len(set(o)) for o in order]
                counts_list.append(counts)

        counts_array = np.array(counts_list)
        vote_distribution, se, h = self.mean_confidence_interval(
            counts_array, confidence=confidence
        )
        return vote_distribution, se, h

    def analyze_count_condorcet_winers(
        self, size_list, number_of_test, max_sample_per_test=100
    ):
        self.df_dict["condorcet_winner"] = {}
        self.df_dict["condorcet_winner"]["size_list"] = size_list
        condorcet_list, accuracy_list = [], []
        for size in size_list:
            counter, condorcet, hits = self.count_condorcet_winers(
                voting_group_size=size,
                number_of_test=number_of_test,
                max_sample_per_test=max_sample_per_test,
            )
            condorcet_list.append(condorcet / counter)
            if condorcet == 0:
                accuracy_list.append(0)
            else:
                accuracy_list.append(hits / condorcet)
        self.df_dict["condorcet_winner"]["occurrences"] = condorcet_list
        self.df_dict["condorcet_winner"]["accuracy"] = accuracy_list

    def count_condorcet_winers(
        self, voting_group_size=9, number_of_test=1, max_sample_per_test=100
    ):
        """
        count vote distribution, return list of size class_number, each value represents avg number of diffrient candidates
        per single voting, normalize to number of voters.
        """
        sample_counter, is_condorcet_winner, properly_clasification = 0, 0, 0
        for voters_set in range(number_of_test):
            df_sample = self.df.sample(n=voting_group_size, replace=False)
            for i in range(min(self.df_dict["samples"], max_sample_per_test)):
                ground_true = df_sample.iloc[0, i * (self.df_dict["class_number"] + 1)]
                vote_matrix = df_sample.iloc[
                    :,
                    i * (self.df_dict["class_number"] + 1)
                    + 1 : i * (self.df_dict["class_number"] + 1)
                    + 1
                    + self.df_dict["class_number"],
                ]
                vote_matrix = vote_matrix.values
                reverse_order = np.argsort(vote_matrix)
                order = np.fliplr(reverse_order)
                if voting_group_size % 2:
                    sample_counter += 1
                    indices = self.detect_condorcet(order)
                    if len(indices) == 1:
                        is_condorcet_winner += 1
                        if indices[0] == ground_true:
                            properly_clasification += 1

        return (sample_counter, is_condorcet_winner, properly_clasification)

    def detect_condorcet(self, order):
        candidate_points = [0 for i in range(self.df_dict["class_number"])]
        for candidate in range(self.df_dict["class_number"]):
            for oponent in range(candidate, self.df_dict["class_number"]):

                dominance = 0
                for vote in order:
                    vote = list(vote)
                    if vote.index(candidate) < vote.index(oponent):
                        dominance += 1
                    else:
                        dominance -= 1
                if dominance > 0:
                    candidate_points[candidate] += 1
                else:
                    candidate_points[oponent] += 1
        return [i for i, x in enumerate(candidate_points) if x == max(candidate_points)]

    def mean_confidence_interval(self, data, confidence=0.95):
        n = data.shape[0]
        m, std = np.mean(data, axis=0), np.std(data, axis=0)
        se = std / (n) ** 0.5
        h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
        return m, se, h

    def plot_voting_analyze(self, save_filepath, plot_title):

        fig = plt.figure(constrained_layout=True)
        fig.suptitle(plot_title)

        gs = gridspec.GridSpec(6, 4)
        gs.update(wspace=0.75, hspace=1)
        ax_dist = plt.subplot(gs[:2, :2])
        ax_dist_norm = plt.subplot(gs[2:4, :2])
        ax_hist = plt.subplot(gs[2:4, 2:])
        ax_condorcet = plt.subplot(gs[0, 2:])
        ax_cond_acc = plt.subplot(gs[1, 2:])
        ax_voting = plt.subplot(gs[4:, :])

        for key, value in self.df_dict["voting"]["method_results"].items():
            ax_voting.plot(
                self.df_dict["voting"]["size_list"],
                value,
                label=key,
                linestyle="-",
                linewidth=1,
                marker=".",
                markersize=2,
            )
        ax_voting.legend(loc=2, prop={"size": 5})
        ax_voting.set_title("accuracy vs ensemble size", fontsize=7)
        ax_voting.tick_params(axis="both", which="major", labelsize=7)

        for dist, size, errors in zip(
            self.df_dict["vote_distribution"]["values"],
            self.df_dict["vote_distribution"]["size_list"],
            self.df_dict["vote_distribution"]["confidence_ranges"],
        ):
            x = [i for i in range(len(dist))]

            ax_dist.plot(
                dist,
                "-.",
                label=f"voters: {size}",
                linestyle="-",
                linewidth=1,
                marker=".",
                markersize=2,
            )
            ax_dist.fill_between(x, dist - errors, dist + errors, alpha=0.2)
            ax_dist_norm.plot(
                dist / dist[0],
                "-.",
                label=f"voters: {size}",
                linestyle="-",
                linewidth=1,
                marker=".",
                markersize=2,
            )
            ax_dist_norm.fill_between(
                x, (dist - errors) / dist[0], (dist + errors) / dist[0], alpha=0.2
            )

        ax_dist.legend(loc=2, prop={"size": 5})
        ax_dist.set_title("voting distribution", fontsize=7)
        ax_dist_norm.legend(loc=2, prop={"size": 5})
        ax_dist_norm.set_title("normalized voting distribution", fontsize=7)

        ax_dist.tick_params(axis="both", which="major", labelsize=6)
        ax_dist_norm.tick_params(axis="both", which="major", labelsize=6)

        ax_hist.hist(self.df_dict["accuracy"])
        ax_hist.set_title("accuracy histogram", fontsize=7)
        ax_hist.tick_params(axis="both", which="major", labelsize=6)

        ax_condorcet.plot(
            self.df_dict["condorcet_winner"]["size_list"],
            self.df_dict["condorcet_winner"]["occurrences"],
            linestyle="-",
            linewidth=1,
            marker=".",
            markersize=5,
        )
        ax_condorcet.set_title("condorcet winners %", fontsize=7)
        ax_condorcet.tick_params(axis="both", which="major", labelsize=6)

        ax_cond_acc.plot(
            self.df_dict["condorcet_winner"]["size_list"],
            self.df_dict["condorcet_winner"]["accuracy"],
            linestyle="-",
            linewidth=1,
            marker=".",
            markersize=5,
        )
        ax_cond_acc.set_title("condorcet accuracy", fontsize=7)
        ax_cond_acc.tick_params(axis="both", which="major", labelsize=6)

        plt.savefig(save_filepath, figsize=[4 * 4, 6 * 4], dpi=300)

    def save_datadict(self, filepath):
        save_obj(self.df_dict, filepath)