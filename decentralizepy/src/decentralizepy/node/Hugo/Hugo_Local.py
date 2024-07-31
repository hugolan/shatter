import copy
import importlib
import json
import logging
import math
import os
import random
from collections import deque
from random import Random

import numpy as np
import torch
from matplotlib import pyplot as plt

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.Node import Node
from scipy.stats import entropy


class Hugo_Local(Node):
    """
    This class defines the node on overlay graph

    """

    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def calculate_kl_distance(self, P, Q):
        epsilon = 1e-8
        min_positive = 1e-10

        shift_value = np.abs(min(np.min(Q),np.min(P))) + epsilon
        P = P + shift_value
        Q = Q + shift_value

        #clip
        P = np.clip(P, min_positive, None)
        Q = np.clip(Q, min_positive, None)


        # Normalize to ensure they sum to 1
        P = P / np.sum(P)
        Q = Q / np.sum(Q)

        # Calculate KL divergence using scipy.stats.entropy
        distance = entropy(P, Q)
        return distance

    def save_plot(self, l, label, title, xlabel, filename):
        """
        Save Matplotlib plot. Clears previous plots.

        Parameters
        ----------
        l : dict
            dict of x -> y. `x` must be castable to int.
        label : str
            label of the plot. Used for legend.
        title : str
            Header
        xlabel : str
            x-axis label
        filename : str
            Name of file to save the plot as.

        """
        plt.clf()
        y_axis = [l[key] for key in l.keys()]
        x_axis = list(map(int, l.keys()))
        plt.plot(x_axis, y_axis, label=label)
        plt.xlabel(xlabel)
        plt.title(title)
        plt.savefig(filename)

    def get_neighbors(self, node=None):
        #new_neigh = set()
        #while len(new_neigh) < self.degree:
        #    index = random.choices(self.list_neighbours, weights=self.probability_matrix, k=1)[0]
        #    if index != self.rank:
        #        new_neigh.add(index)
        #return list(new_neigh)
        l = list(np.random.choice(self.list_neighbours, size=self.degree, replace=False, p=self.probability_matrix))
        l = [int(i) for i in l]
        return l
    def get_neighbors_skewscount(self):
        return self.my_neighbors


    def update_stored_models(self, averaging_deque):
        #if self.rank == 0:
        #    print(averaging_deque)
        for rank, d in averaging_deque.items():
            self.stored_models[rank] = averaging_deque[int(rank)][0]['params']
        return
    def update_probability_matrix(self, to_send):
        sum_dist = 0
        count  = 0

        for i in range(len(self.stored_models)):
            if len(self.stored_models[i]) != 0 and i != self.rank:
                count += 1
                if self.distance_similarity == "closer" and self.distance_nodes == 2:
                    d = 1/np.linalg.norm(self.stored_models[i] - to_send['params'], ord=self.distance_nodes)
                    self.probability_matrix[i] = d
                    sum_dist += d
                elif self.distance_similarity == "further" and self.distance_nodes == 2:
                    d = np.linalg.norm(self.stored_models[i] - to_send['params'],ord=self.distance_nodes)
                    self.probability_matrix[i] = d
                    sum_dist += d
                elif self.distance_similarity == "furtherexp" and self.distance_nodes == 2:
                    d = np.linalg.norm(self.stored_models[i] - to_send['params'],ord=self.distance_nodes)
                    if self.iteration > 50:
                        d = np.exp(-self.weighting_factor * d)
                    self.probability_matrix[i] = d
                    sum_dist += d
                elif self.distance_similarity == "closer" and self.distance_nodes == "kl_distance":
                    d = 1/self.calculate_kl_distance(self.stored_models[i], to_send['params'])
                    if self.iteration > 40:
                        d = np.exp(-self.weighting_factor * d)
                    self.probability_matrix[i] = d
                    sum_dist += d
                elif self.distance_similarity == "further" and self.distance_nodes == "kl_distance":
                    d = self.calculate_kl_distance(self.stored_models[i], to_send['params'])
                    self.probability_matrix[i] = d
                    sum_dist += d
                else:
                    print("wrong_distance")


        for i in range(len(self.stored_models)):
            if len(self.stored_models[i]) == 0 and i != self.rank:
                self.probability_matrix[i] = sum_dist/count

        #if self.rank == 0:
        #    print(self.probability_matrix)

        #if self.iteration < 100:
        sum_weights = np.sum(self.probability_matrix)
        for i in range(len(self.probability_matrix)):
            self.probability_matrix[i] = self.probability_matrix[i]/sum_weights
        #else:
        #    self.probability_matrix = list(self.softmax(self.probability_matrix))
        #if self.rank == 0:
        #    print(self.probability_matrix)
        return


    def receive_DPSGD(self):

        return self.receive_channel("DPSGD", block=True)

    def received_from_all(self):
        """
        Check if all neighbors have sent the current iteration

        Returns
        -------
        bool
            True if required data has been received, False otherwise

        """
        for k in self.my_neighbors:
            if (
                (k not in self.peer_deques)
                or len(self.peer_deques[k]) == 0
                or self.peer_deques[k][0]["iteration"] != self.iteration
            ):
                return False
        return True

    def run(self):
        """
        Start the decentralized learning

        """

        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1
        self.rng = Random()
        self.rng.seed(self.dataset.random_seed + self.uid)
        self.connect_neighbors()
        logging.info("Connected to all neighbors")

        logging.info("Total number of neighbor: {}".format(len(self.my_neighbors)))
        try:
            os.mkdir(self.log_dir + "/models")
        except:
            None

        for iteration in range(self.iterations):
            # Local Phase
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1

            self.iteration = iteration
            self.trainer.train(self.dataset)

            neighbors_this_round = self.get_neighbors()
            to_send = self.sharing.get_data_to_send()
            to_send["CHANNEL"] = "DPSGD"

            # Communication Phase
            for neighbor in neighbors_this_round:
                logging.debug("Sending to neighbor: %d", neighbor)
                self.communication.send(neighbor, to_send)

            for x in self.my_neighbors:
                if x not in neighbors_this_round:
                    self.communication.send(
                        x,
                        {
                            "CHANNEL": "DPSGD",
                            "iteration": iteration,
                            "NotWorking": True,
                        },
                    )

            while not self.received_from_all():
                response = self.receive_DPSGD()
                if response:
                    sender, data = response
                    logging.debug(
                        "Received Model from {} of iteration {}: {}".format(
                            sender,
                            data["iteration"],
                            "NotWorking" if "NotWorking" in data else "",
                        )
                    )
                    if sender not in self.peer_deques:
                        self.peer_deques[sender] = deque()

                    if data["iteration"] == self.iteration:
                        self.peer_deques[sender].appendleft(data)
                    else:
                        self.peer_deques[sender].append(data)

            averaging_deque = dict()
            atleast_one = False
            for x in self.my_neighbors:
                if x in self.peer_deques and len(self.peer_deques[x]) > 0:
                    this_message = self.peer_deques[x][0]
                    if (
                        this_message["iteration"] == self.iteration
                        and "NotWorking" not in this_message
                    ):
                        averaging_deque[x] = self.peer_deques[x]
                        atleast_one = True
                    elif this_message["iteration"] == self.iteration:
                        self.peer_deques[x].popleft()
                        logging.debug(
                            "Discarding message from {} of iteration {}".format(
                                x, this_message["iteration"]
                            )
                        )
            #print("neigh=" + str(neighbors_this_round) + " rank " +str(self.rank) + " deque " + str(averaging_deque))

            if atleast_one:
                self.update_stored_models(averaging_deque)

            if atleast_one:
                self.update_probability_matrix(to_send)


            #if (iteration % self.alternate_rounds) == 0:
            if atleast_one:
                self.sharing._averaging(averaging_deque)
            else:
                self.sharing.communication_round += 1
            #else:
            #    self.sharing.communication_round += 1


            if self.reset_optimizer:
                self.optimizer = self.optimizer_class(
                    self.model.parameters(), **self.optimizer_params
                )  # Reset optimizer state
                self.trainer.reset_optimizer(self.optimizer)

            if iteration:
                with open(
                    os.path.join(self.log_dir, "{}_results.json".format(self.rank)),
                    "r",
                ) as inf:
                    results_dict = json.load(inf)
            else:
                results_dict = {
                    "train_loss": {},
                    "test_loss": {},
                    "test_acc": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                    "received_this_round": {},
                    "probability_matrix": {},
                    "neighbors": {},
                    "model_weights": {},
                }

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes
            results_dict["neighbors"][iteration + 1] = neighbors_this_round

            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][
                    iteration + 1
                ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.communication.total_data
            if hasattr(self.communication, "received_this_round"):
                results_dict["received_this_round"][
                    iteration + 1
                ] = self.communication.received_this_round

            if rounds_to_train_evaluate == 0:
                logging.info("Evaluating on train set.")
                rounds_to_train_evaluate = self.train_evaluate_after * change
                loss_after_sharing = self.trainer.eval_loss(self.dataset)
                results_dict["train_loss"][iteration + 1] = loss_after_sharing
                self.save_plot(
                    results_dict["train_loss"],
                    "train_loss",
                    "Training Loss",
                    "Communication Rounds",
                    os.path.join(self.log_dir, "{}_train_loss.png".format(self.rank)),
                )

                results_dict["probability_matrix"][iteration+1] = self.probability_matrix

            #if (iteration%100) == 0 and iteration != 0:
            #    torch.save(self.model.state_dict(), self.log_dir + "/models/" + str(self.rank)+ "_" + str(iteration) + "_weights.pt")

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)

        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")

    def cache_fields(
        self,
        rank,
        machine_id,
        mapping,
        graph,
        iterations,
        log_dir,
        weights_store_dir,
        test_after,
        train_evaluate_after,
        reset_optimizer,
    ):
        """
        Instantiate object field with arguments.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        """
        self.rank = rank
        self.machine_id = machine_id
        self.graph = graph
        self.mapping = mapping
        self.uid = self.mapping.get_uid(rank, machine_id)
        self.log_dir = log_dir
        self.weights_store_dir = weights_store_dir
        self.iterations = iterations
        self.test_after = test_after
        self.train_evaluate_after = train_evaluate_after
        self.reset_optimizer = reset_optimizer
        self.sent_disconnections = False


        logging.debug("Rank: %d", self.rank)
        logging.debug("type(graph): %s", str(type(self.rank)))
        logging.debug("type(mapping): %s", str(type(self.mapping)))

    def init_comm(self, comm_configs):
        """
        Instantiate communication module from config.

        Parameters
        ----------
        comm_configs : dict
            Python dict containing communication config params

        """
        comm_module = importlib.import_module(comm_configs["comm_package"])
        comm_class = getattr(comm_module, comm_configs["comm_class"])
        comm_params = utils.remove_keys(comm_configs, ["comm_package", "comm_class"])
        self.addresses_filepath = comm_params.get("addresses_filepath", None)
        self.communication = comm_class(
            self.rank, self.machine_id, self.mapping, self.graph.n_procs, **comm_params
        )

    def instantiate(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        *args
    ):
        """
        Construct objects.

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations.
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """
        logging.info("Started process.")

        self.init_log(log_dir, rank, log_level)

        self.cache_fields(
            rank,
            machine_id,
            mapping,
            graph,
            iterations,
            log_dir,
            weights_store_dir,
            test_after,
            train_evaluate_after,
            reset_optimizer,
        )

        self.init_dataset_model(config["DATASET"])
        self.init_optimizer(config["OPTIMIZER_PARAMS"])
        self.init_trainer(config["TRAIN_PARAMS"])
        self.init_comm(config["COMMUNICATION"])

        self.message_queue = dict()

        self.barrier = set()
        self.my_neighbors = self.graph.neighbors(self.uid)

        self.init_sharing(config["SHARING"])
        self.peer_deques = dict()
        self.connect_neighbors()
        self.probability_matrix = [1/(graph.n_procs - 1) if self.rank != i else 0 for i in range(graph.n_procs)]
        self.stored_models = [[] for i in range(graph.n_procs)]
        self.list_neighbours = [i for i in range(graph.n_procs)]

        self.distance_nodes = config["PARAMS"]["distance_nodes"]
        self.distance_similarity = config["PARAMS"]["distance_similarity"]
        self.alternate_rounds = config["PARAMS"]["alternate_rounds"]
        self.weighting_factor = config["PARAMS"]["weighting_factor"]

    def __init__(
        self,
        rank: int,
        machine_id: int,
        mapping: Mapping,
        graph: Graph,
        config,
        iterations=1,
        log_dir=".",
        weights_store_dir=".",
        log_level=logging.INFO,
        test_after=5,
        train_evaluate_after=1,
        reset_optimizer=1,
        *args
    ):
        """
        Constructor

        Parameters
        ----------
        rank : int
            Rank of process local to the machine
        machine_id : int
            Machine ID on which the process in running
        mapping : decentralizepy.mappings
            The object containing the mapping rank <--> uid
        graph : decentralizepy.graphs
            The object containing the global graph
        config : dict
            A dictionary of configurations. Must contain the following:
            [DATASET]
                dataset_package
                dataset_class
                model_class
            [OPTIMIZER_PARAMS]
                optimizer_package
                optimizer_class
            [TRAIN_PARAMS]
                training_package = decentralizepy.training.Training
                training_class = Training
                epochs_per_round = 25
                batch_size = 64
        iterations : int
            Number of iterations (communication steps) for which the model should be trained
        log_dir : str
            Logging directory
        weights_store_dir : str
            Directory in which to store model weights
        log_level : logging.Level
            One of DEBUG, INFO, WARNING, ERROR, CRITICAL
        test_after : int
            Number of iterations after which the test loss and accuracy arecalculated
        train_evaluate_after : int
            Number of iterations after which the train loss is calculated
        reset_optimizer : int
            1 if optimizer should be reset every communication round, else 0
        args : optional
            Other arguments

        """

        total_threads = os.cpu_count()
        self.threads_per_proc = max(
            math.floor(total_threads / mapping.procs_per_machine), 1
        )
        torch.set_num_threads(self.threads_per_proc)
        torch.set_num_interop_threads(1)
        self.instantiate(
            rank,
            machine_id,
            mapping,
            graph,
            config,
            iterations,
            log_dir,
            weights_store_dir,
            log_level,
            test_after,
            train_evaluate_after,
            reset_optimizer,
            *args
        )

        nodeConfigs = config["NODE"]
        self.degree = (
            nodeConfigs["graph_degree"] if "graph_degree" in nodeConfigs else 2
        )

        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )
        self.run()
