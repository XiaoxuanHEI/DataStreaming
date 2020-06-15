#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 20:11:32 2020

@author: cherilyn
"""

import copy
import textwrap
from abc import ABCMeta
from operator import attrgetter, itemgetter

import numpy as np

from skmultiflow.utils.utils import get_dimensions, normalize_values_in_dict, calculate_object_size
from skmultiflow.core import BaseSKMObject, ClassifierMixin
from skmultiflow.trees.numeric_attribute_class_observer_gaussian import NumericAttributeClassObserverGaussian
from skmultiflow.trees.nominal_attribute_class_observer import NominalAttributeClassObserver
from skmultiflow.trees.attribute_class_observer_null import AttributeClassObserverNull
from skmultiflow.trees.attribute_split_suggestion import AttributeSplitSuggestion
from skmultiflow.trees.info_gain_split_criterion import InfoGainSplitCriterion
from skmultiflow.trees.numeric_attribute_binary_test import NumericAttributeBinaryTest
from skmultiflow.bayes import do_naive_bayes_prediction


MAJORITY_CLASS = 'mc'
NAIVE_BAYES = 'nb'
NAIVE_BAYES_ADAPTIVE = 'nba'


class HoeffdingOptTree(BaseSKMObject, ClassifierMixin):
    class FoundNode(object):
        def __init__(self, node=None, parent=None, parent_branch=None):
            self.node = node
            self.parent = parent
            self.parent_branch = parent_branch

    class Node(metaclass=ABCMeta):
        def __init__(self, class_observations=None):
            """ Node class constructor. """
            if class_observations is None:
                class_observations = {}  # Dictionary (class_value, weight)
            self._observed_class_distribution = class_observations

        def is_leaf():
            return True

        def filter_instance_to_leaves(self, X, parent, parent_branch, update_splitter_counts):
            nodes = []
            self._filter_instance_to_leaves(X, parent, parent_branch, nodes, update_splitter_counts)
            return nodes

        def _filter_instance_to_leaves(self, X, parent, parent_branch, nodes, update_splitter_counts):
            nodes.append(HoeffdingOptTree.FoundNode(self, parent, parent_branch))

        def get_observed_class_distribution(self):
            return self._observed_class_distribution

        def set_observed_class_distribution(self, observed_class_distribution):
            self._observed_class_distribution = observed_class_distribution

        def get_class_votes(self, X, hot):
            return self._observed_class_distribution

        def observed_class_distribution_is_pure(self):
            count = 0
            for _, weight in self._observed_class_distribution.items():
                if weight != 0:
                    count += 1
                    if count == 2:  # No need to count beyond this point
                        break
            return count < 2
            
        def subtree_depth(self):
            return 0  # 0 if leaf

        def calculate_promise(self):
            total_seen = sum(self._observed_class_distribution.values())
            if total_seen > 0:
                return total_seen - max(self._observed_class_distribution.values())
            else:
                return 0

    class SplitNode(Node):
        def __init__(self, split_test, class_observations, next_option=None):
            """ SplitNode class constructor."""
            super().__init__(class_observations)
            self._split_test = split_test
            # Dict of tuples (branch, child)
            self._children = {}
            self.next_option = next_option
            self.option_count = 1

        def num_children(self):
            return len(self._children)

        def get_split_test(self):
            return self._split_test

        def set_child(self, index, node):
            if (self._split_test.max_branches() >= 0) and (index >= self._split_test.max_branches()):
                raise IndexError
            self._children[index] = node

        def get_child(self, index):
            if index in self._children:
                return self._children[index]
            else:
                return None

        def instance_child_index(self, X):
            return self._split_test.branch_for_instance(X)

        @staticmethod
        def is_leaf():
            return False

        def __filter_instance_to_leaves(self, X, y, weight, parent, parent_branch, nodes, update_splitter_counts):
            if (update_splitter_counts):
                try:
                    self._observed_class_distribution[y] += weight
                except KeyError:
                    self._observed_class_distribution[y] = weight
                    self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))
            child_index = self.instance_child_index(X)
            if child_index >= 0:
                child = self.get_child(child_index)
                if child is not None:
                    return child.__filter_instance_to_leaves(X, y, weight, self, child_index, nodes, update_splitter_counts)
                    return nodes.append(HoeffdingOptTree.FoundNode(None, self, child_index))
            if self.next_option is not None:
                self.next_option.__filter_instance_to_leaves(X, y, weight, self, -999, nodes, update_splitter_counts)


        def subtree_depth(self):
            max_child_depth = 0
            for child in self._children:
                if child is not None:
                    depth = child.subtree_depth()
                    if depth > max_child_depth:
                        max_child_depth = depth
            return max_child_depth + 1

        def compute_merit_of_existing_split(self, split_criterion, pre_split_dist):
            post_split_dist = []
            for i in range(len(self._children)):
                post_split_dist[i] = self._children[i].get_observed_class_distribution()
            return split_criterion.get_merit_of_split(pre_split_dist, post_split_dist)

        def update_option_count(self, source, hot):
            if self.option_count == -999:
                self.parent.update_option_count(source, hot)
            else:
                max_child_count = -999
                curr = self
                while curr is not None:
                    for child in curr._children:
                        if isinstance(child, self.SplitNode):
                            split_child = self.SplitNode(child)
                            if (split_child.option_count > max_child_count):
                                max_child_count = split_child.option_count
                    if curr.next_option is not None and isinstance(curr.next_option, self.SplitNode):
                        curr = self.SplitNode(curr.next_option)
                    else:
                        curr = None
                if max_child_count > self.option_count:
                    delta = max_child_count - self.option_count
                    self.option_count = max_child_count
                    if self.option_count >= self.max_option_path:
                        self.kill_option_leaf(hot)
                    curr = self
                    while curr is not None:
                        for child in curr._children:
                            if isinstance(child, self.SplitNode):
                                split_child = self.SplitNode(child)
                                if split_child is not source:
                                    split_child.update_option_count_below(delta, hot)
                        if curr.next_option is not None and isinstance(curr.next_option, self.SplitNode):
                            curr = self.SplitNode(curr.next_option)
                        else:
                            curr = None
                    if self.parent is not None:
                        self.parent.update_option_count(self, hot)

        def update_option_count_below(self, delta, hot):
            if self.option_count != -999:
                self.option_count += delta
                if self.option_count >= self.max_option_path:
                    self.kill_option_leaf(hot)
            for child in self._children:
                split_child = self.SplitNode(child)
                split_child.update_option_count_below(delta, hot)
            if isinstance(self.next_option, self.SplitNode):
                self.splitNode(self.next_option).update_option_count_below(delta, hot)

        def kill_option_leaf(self, hot):
            if isinstance(self.next_option, self.SplitNode):
                self.splitNode(self.next_option).kill_option_leaf(hot)
            elif isinstance(self.next_option, self.ActiveLearningNode):
                self.next_option = None
                hot.active_leaf_node_count -= 1
            elif isinstance(self.next_option, self.InactiveLearningNode):
                self.next_option = None
                hot.inactive_leaf_node_count -= 1

    class LearningNode(Node):
        def __init__(self, initial_class_observations=None):
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, hot):
            pass

    class InactiveLearningNode(LearningNode):
        def __init__(self, initial_class_observations=None):
            super().__init__(initial_class_observations)

        def learn_from_instance(self, X, y, weight, hot):
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                self._observed_class_distribution[y] = weight
                self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))

    class ActiveLearningNode(LearningNode):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self._weight_seen_at_last_split_evaluation = self.get_weight_seen()
            self._attribute_observers = {}

        def learn_from_instance(self, X, y, weight, hot):
            try:
                self._observed_class_distribution[y] += weight
            except KeyError:
                self._observed_class_distribution[y] = weight
                self._observed_class_distribution = dict(sorted(self._observed_class_distribution.items()))

            for i in range(len(X)):
                try:
                    obs = self._attribute_observers[i]
                except KeyError:
                    if hot.nominal_attributes is not None and i in hot.nominal_attributes:
                        obs = NominalAttributeClassObserver()
                    else:
                        obs = NumericAttributeClassObserverGaussian()
                    self._attribute_observers[i] = obs
                obs.observe_attribute_class(X[i], int(y), weight)

        def get_weight_seen(self):
            return sum(self._observed_class_distribution.values())

        def get_weight_seen_at_last_split_evaluation(self):
            return self._weight_seen_at_last_split_evaluation

        def set_weight_seen_at_last_split_evaluation(self, weight):
            self._weight_seen_at_last_split_evaluation = weight

        def get_best_split_suggestions(self, criterion, hot):
            best_suggestions = []
            pre_split_dist = self._observed_class_distribution
            null_split = AttributeSplitSuggestion(None, [{}],
                                                  criterion.get_merit_of_split(pre_split_dist, [pre_split_dist]))
            best_suggestions.append(null_split)
            for i, obs in self._attribute_observers.items():
                best_suggestion = obs.get_best_evaluated_split_suggestion(criterion, pre_split_dist,
                                                                          i, hot.binary_split)
                if best_suggestion is not None:
                    best_suggestions.append(best_suggestion)
            return best_suggestions

        def disable_attribute(self, att_idx):
            if att_idx in self._attribute_observers:
                self._attribute_observers[att_idx] = AttributeClassObserverNull()

    class LearningNodeNB(ActiveLearningNode):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)

        def get_class_votes(self, X, hot):
            if self.get_weight_seen() >= 0:
                return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            else:
                return super().get_class_votes(X, hot)

        def disable_attribute(self, att_index):
            pass

    class LearningNodeNBAdaptive(LearningNodeNB):
        def __init__(self, initial_class_observations):
            super().__init__(initial_class_observations)
            self._mc_correct_weight = 0.0
            self._nb_correct_weight = 0.0

        def learn_from_instance(self, X, y, weight, hot):
            if self._observed_class_distribution == {}:
                # All classes equal, default to class 0
                if 0 == y:
                    self._mc_correct_weight += weight
            elif max(self._observed_class_distribution, key=self._observed_class_distribution.get) == y:
                self._mc_correct_weight += weight
            nb_prediction = do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)
            if max(nb_prediction, key=nb_prediction.get) == y:
                self._nb_correct_weight += weight
            super().learn_from_instance(X, y, weight, hot)

        def get_class_votes(self, X, hot):
            if self._mc_correct_weight > self._nb_correct_weight:
                return self._observed_class_distribution
            return do_naive_bayes_prediction(X, self._observed_class_distribution, self._attribute_observers)

    def __init__(self,
                 max_option_path = 5,
                 grace_period=200,
                 split_confidence=0.00000001,
                 secondary_split_confidence=0.955,
                 tie_threshold=0.05,
                 binary_split=False,
                 leaf_prediction='nba',
                 nominal_attributes=None,
                 ):
        """ HoeffdingOptionTree class constructor."""
        super().__init__()
        self.max_option_path = max_option_path 
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.secondary_split_confidence = secondary_split_confidence
        self.tie_threshold = tie_threshold
        self.binary_split = binary_split
        self.leaf_prediction = leaf_prediction
        self.nominal_attributes = nominal_attributes

        self._tree_root = None
        self._decision_node_cnt = 0
        self._active_leaf_node_cnt = 0
        self._inactive_leaf_node_cnt = 0
        self._max_prediction_path = 0
        self._train_weight_seen_by_model = 0.0
        self.classes = None

    @property
    def max_option_path(self):
        return self._max_option_path

    @max_option_path.setter
    def max_option_path(self, max_option_path):
        self._max_option_path = max_option_path
        
    @property
    def grace_period(self):
        return self._grace_period

    @grace_period.setter
    def grace_period(self, grace_period):
        self._grace_period = grace_period


    @property
    def split_confidence(self):
        return self._split_confidence

    @split_confidence.setter
    def split_confidence(self, split_confidence):
        self._split_confidence = split_confidence

    @property
    def secondary_split_confidence(self):
        return self._secondary_split_confidence

    @secondary_split_confidence.setter
    def secondary_split_confidence(self, secondary_split_confidence):
        self._secondary_split_confidence = secondary_split_confidence

    @property
    def binary_split(self):
        return self._binary_split

    @binary_split.setter
    def binary_split(self, binary_split):
        self._binary_split = binary_split


    @property
    def leaf_prediction(self):
        return self._leaf_prediction

    @leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction != MAJORITY_CLASS and leaf_prediction != NAIVE_BAYES \
                and leaf_prediction != NAIVE_BAYES_ADAPTIVE:
            self._leaf_prediction = NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    @property
    def nominal_attributes(self):
        return self._nominal_attributes

    @nominal_attributes.setter
    def nominal_attributes(self, nominal_attributes):
        self._nominal_attributes = nominal_attributes

    @property
    def classes(self):
        return self._classes

    @classes.setter
    def classes(self, value):
        self._classes = value

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        if self.classes is None and classes is not None:
            self.classes = classes
        if y is not None:
            row_cnt, _ = get_dimensions(X)
            if sample_weight is None:
                sample_weight = np.ones(row_cnt)
            if row_cnt != len(sample_weight):
                raise ValueError('Inconsistent number of instances ({}) and weights ({}).'.format(row_cnt,
                                                                                                  len(sample_weight)))
            for i in range(row_cnt):
                if sample_weight[i] != 0.0:
                    self._train_weight_seen_by_model += sample_weight[i]
                    self._partial_fit(X[i], y[i], sample_weight[i])

        return self

    def _partial_fit(self, X, y, sample_weight):
        if self._tree_root is None:
            self._tree_root = self._new_learning_node()
            self._active_leaf_node_cnt = 1
        found_node = self._tree_root.filter_instance_to_leaves(X, None, -1, True)
        for fn in found_node:
            leaf_node = fn.node
            if leaf_node is None:
                leaf_node = self._new_learning_node()
                fn.parent.set_child(fn.parent_branch, leaf_node)
                self._active_leaf_node_cnt += 1
            if isinstance(leaf_node, self.LearningNode):
                learning_node = leaf_node
                learning_node.learn_from_instance(X, y, sample_weight, self)
                if isinstance(learning_node, self.ActiveLearningNode):
                    active_learning_node = learning_node
                    weight_seen = active_learning_node.get_weight_seen()
                    weight_diff = weight_seen - active_learning_node.get_weight_seen_at_last_split_evaluation()
                    if weight_diff >= self.grace_period:
                        self._attempt_to_split(active_learning_node, fn.parent, fn.parent_branch)
                        active_learning_node.set_weight_seen_at_last_split_evaluation(weight_seen)

    def get_votes_for_instance(self, X):
        if self._tree_root is not None:
            found_nodes = self._tree_root.filter_instance_to_leaves(X, None, -1, 0)
            result = []
            for fn in found_nodes:
                if fn.parent_branch != -999:
                    leaf_node = fn.node;
                    if leaf_node is None:
                        leaf_node = fn.parent
                    dist = leaf_node.get_class_votes(X, self)
                    result.append(dist)
            return result
        else:
            return {}

    def predict(self, X):
        r, _ = get_dimensions(X)
        predictions = []
        y_proba = self.predict_proba(X)
        for i in range(r):
            index = np.argmax(y_proba[i])
            predictions.append(index)
        return np.array(predictions)

    def predict_proba(self, X):
        r, _ = get_dimensions(X)
        predictions = []
        for i in range(r):
            votes = self.get_votes_for_instance(X[i]).copy()
            if votes == {}:
                # Tree is empty, all classes equal, default to zero
                predictions.append([0])
            else:
                new_votes = dict((key,d[key]) for d in votes for key in d)                    
                if sum(new_votes.values()) != 0:
                    normalize_values_in_dict(new_votes)
                if self.classes is not None:
                    y_proba = np.zeros(int(max(self.classes)) + 1)
                else:
                    y_proba = np.zeros(int(max(new_votes.keys())) + 1)
                for key, value in new_votes.items():
                    y_proba[int(key)] = value
                predictions.append(y_proba)
        return np.array(predictions)


    def _new_learning_node(self, initial_class_observations=None):
        """ Create a new learning node. The type of learning node depends on the tree configuration."""
        if initial_class_observations is None:
            initial_class_observations = {}
        if self._leaf_prediction == MAJORITY_CLASS:
            return self.ActiveLearningNode(initial_class_observations)
        elif self._leaf_prediction == NAIVE_BAYES:
            return self.LearningNodeNB(initial_class_observations)
        else:
            return self.LearningNodeNBAdaptive(initial_class_observations)


    def compute_hoeffding_bound(self, range_val, confidence, n):
        return np.sqrt((range_val * range_val * np.log(1.0 / confidence)) / (2.0 * n))

    def new_split_node(self, split_test, class_observations):
        """ Create a new split node."""
        return self.SplitNode(split_test, class_observations)

    def _attempt_to_split(self, node: ActiveLearningNode, parent: SplitNode, parent_idx: int):
        split_criterion = InfoGainSplitCriterion()
        best_split_suggestions = node.get_best_split_suggestions(split_criterion, self)
        best_split_suggestions.sort(key=attrgetter('merit'))
        should_split = False
        if parent_idx != -999:
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                    node.get_observed_class_distribution()), self.split_confidence, node.get_weight_seen())
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (best_suggestion.merit - second_best_suggestion.merit > hoeffding_bound
                        or hoeffding_bound < self.tie_threshold):  # best_suggestion.merit > 1e-10 and \
                    should_split = True


        elif len(best_split_suggestions) > 0:
            hoeffding_bound = self.compute_hoeffding_bound(split_criterion.get_range_of_merit(
                node.get_observed_class_distribution()), self.secondary_split_confidence, node.get_weight_seen())
            best_suggestion = best_split_suggestions[-1]
            current = parent
            best_previous_merit = float("-inf")
            pre_dist = node.get_observed_class_distribution()
            while (True):
                merit = current.compute_merit_of_existing_split(split_criterion, pre_dist);
                if merit > best_previous_merit:
                    best_previous_merit = merit
                if current.option_count != -999:
                    break
                current = current.parent
            if best_suggestion.merit - best_previous_merit > hoeffding_bound:
                should_split = True

        if should_split:
            split_decision = best_split_suggestions[-1]
            if split_decision.split_test is None:
                if parent_idx != -999:
                    self._deactivate_learning_node(node, parent, parent_idx)
            else:
                new_split = self.new_split_node(split_decision.split_test,
                                                node.get_observed_class_distribution())
                new_split.parent = parent

                # Add option procedure
                option_head = parent
                if parent is not None:
                    while option_head.option_count == -999:
                        option_head = option_head.parent
                if parent_idx == -999 and parent is not None:
                    # adding a new option
                    new_split.option_count = -999
                    option_head.update_option_count_below(1, self)
                    if option_head.parent is not None:
                        option_head.parent.update_option_count(option_head, self)
                        self.add_to_option_table(split_decision, option_head.parent)
                else:
                    # adding a regular leaf
                    if option_head is None:
                        new_split.option_count = 1
                    else:
                        new_split.option_count = option_head.option_count
                num_option = 1
                if option_head is not None:
                    num_option = option_head.option_count
                if num_option < self.max_option_path:
                    new_split.next_option = node
                    split_atts = split_decision.split_test.get_atts_test_depends_on()
                    for i in split_atts:
                        node.disable_attribute(i)
                else:
                    self._active_leaf_node_cnt -= 1
                for i in range(split_decision.num_splits()):
                    new_child = self._new_learning_node(split_decision.resulting_class_distribution_from_split(i))
                    new_split.set_child(i, new_child)
                self._decision_node_cnt += 1

                self._active_leaf_node_cnt += split_decision.num_splits()
                if parent is None:
                    self._tree_root = new_split
                else:
                    if parent_idx != -999:
                        parent.set_child(parent_idx, new_split)
                    else:
                        parent.next_option = new_split

    def add_to_option_table(self, best_suggestion: AttributeSplitSuggestion, parent: SplitNode):
        split_atts = best_suggestion.split_test.get_atts_test_depends_on()[0]
        split_val = -1.0
        if isinstance(best_suggestion.split_test, NumericAttributeBinaryTest):
            test = NumericAttributeBinaryTest(best_suggestion.split_test)
            split_val = test.get_split_value()
        tree_depth = 0
        while parent is not None:
            parent = parent.parent
            tree_depth += 1
        print(self._train_weight_seen_by_model + ","
              + tree_depth + "," + split_atts + "," + split_val)

    def deactivate_all_leaves(self):
        """ Deactivate all leaves. """
        learning_nodes = self._find_learning_nodes()
        for i in range(len(learning_nodes)):
            if isinstance(learning_nodes[i], self.ActiveLearningNode):
                self._deactivate_learning_node(learning_nodes[i].node,
                                               learning_nodes[i].parent,
                                               learning_nodes[i].parent_branch)

    def _deactivate_learning_node(self, to_deactivate: ActiveLearningNode, parent: SplitNode, parent_branch: int):
        new_leaf = self.InactiveLearningNode(to_deactivate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            if parent_branch != -999:
                parent.set_child(parent_branch, new_leaf)
            else:
                parent.next_option = new_leaf
        self._active_leaf_node_cnt -= 1
        self._inactive_leaf_node_cnt += 1


    def _activate_learning_node(self, to_activate: InactiveLearningNode, parent: SplitNode, parent_branch: int):
        new_leaf = self._new_learning_node(to_activate.get_observed_class_distribution())
        if parent is None:
            self._tree_root = new_leaf
        else:
            if parent_branch != -999:         
                parent.set_child(parent_branch, new_leaf)
            else:
                parent.next_option = new_leaf
        self._active_leaf_node_cnt += 1
        self._inactive_leaf_node_cnt -= 1
        
        
    def _find_learning_nodes(self):
        found_list = []
        self.__find_learning_nodes(self._tree_root, None, -1, found_list)
        return found_list  
    
    def __find_learning_nodes(self, node, parent, parent_branch, found):
        if node is not None:
            if isinstance(node, self.LearningNode):
                found.append(self.FoundNode(node, parent, parent_branch))
            if isinstance(node, self.SplitNode):
                split_node = node
                for i in range(split_node.num_children()):
                    self.__find_learning_nodes(split_node.get_child(i), split_node, i, found)
                self.__find_learning_nodes(split_node.next_option, split_node, -999, found)
