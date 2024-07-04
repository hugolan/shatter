import copy
import json
import logging
import math
import os
from collections import deque

import torch

from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Mapping import Mapping
from decentralizepy.node.DPSGDWithPeerSampler import DPSGDWithPeerSampler


class Hugo(DPSGDWithPeerSampler):
    """
    This class defines the node for DPSGD

    """

    def receive_neighbors(self):
        return self.receive_channel("PEERS")[1]["NEIGHBORS"]

    def get_neighbors(self, node=None):
        logging.debug("Requesting neighbors from the peer sampler.")
        self.communication.send(
            self.peer_sampler_uid,
            {
                "REQUEST_NEIGHBORS": self.uid,
                "iteration": self.iteration,
                "CHANNEL": "SERVER_REQUEST",
            },
        )
        my_neighbors = self.receive_neighbors()
        logging.debug("Neighbors this round: {}".format(my_neighbors))
        return my_neighbors

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
        peer_sampler_uid=-1,
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
            math.floor(total_threads / mapping.get_local_procs_count()), 1
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
        logging.info(
            "Each proc uses %d threads out of %d.", self.threads_per_proc, total_threads
        )

        self.message_queue["PEERS"] = deque()

        self.peer_sampler_uid = peer_sampler_uid
        self.connect_neighbor(self.peer_sampler_uid)
        self.wait_for_hello(self.peer_sampler_uid)

        self.run()

    def disconnect_neighbors(self):
        """
        Disconnects all neighbors.

        Raises
        ------
        RuntimeError
            If received another message while waiting for BYEs

        """
        if not self.sent_disconnections:
            logging.info("Disconnecting neighbors")

            if self.peer_sampler_uid in self.barrier:
                self.communication.send(
                    self.peer_sampler_uid,
                    {"BYE": self.uid, "CHANNEL": "SERVER_REQUEST"},
                )
                self.barrier.remove(self.peer_sampler_uid)

            for uid in self.barrier:
                self.communication.send(uid, {"BYE": self.uid, "CHANNEL": "DISCONNECT"})
            self.sent_disconnections = True

            while len(self.barrier):
                sender, _ = self.receive_disconnect()
                self.barrier.remove(sender)


    def run(self):
        """
        Start the decentralized learning

        """
        print("WeightsStore: ", self.weights_store_dir)
        self.testset = self.dataset.get_testset()
        rounds_to_test = self.test_after
        rounds_to_train_evaluate = self.train_evaluate_after
        global_epoch = 1
        change = 1

        for iteration in range(self.iterations):
            logging.info("Starting training iteration: %d", iteration)
            rounds_to_train_evaluate -= 1
            rounds_to_test -= 1

            self.iteration = iteration
            self.trainer.train(self.dataset)

            intermediate_weights = copy.deepcopy(self.model.state_dict())

            new_neighbors = self.get_neighbors()

            self.my_neighbors = new_neighbors
            self.connect_neighbors()
            logging.debug("Connected to all neighbors")

            to_send = self.sharing.get_data_to_send(degree=len(self.my_neighbors))
            to_send["CHANNEL"] = "DPSGD"

            for neighbor in self.my_neighbors:
                self.communication.send(neighbor, to_send)

            while not self.received_from_all():
                sender, data = self.receive_DPSGD()
                logging.debug(
                    "Received Model from {} of iteration {}".format(
                        sender, data["iteration"]
                    )
                )
                if sender not in self.peer_deques:
                    self.peer_deques[sender] = deque()

                if data["iteration"] == self.iteration:
                    self.peer_deques[sender].appendleft(data)
                else:
                    self.peer_deques[sender].append(data)

            averaging_deque = dict()
            for neighbor in self.my_neighbors:
                averaging_deque[neighbor] = self.peer_deques[neighbor]

            self.sharing._averaging(averaging_deque)

            post_weights = copy.deepcopy(self.model.state_dict())

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
                    "validation_loss": {},
                    "validation_acc": {},
                    "total_bytes": {},
                    "total_meta": {},
                    "total_data_per_n": {},
                }

            results_dict["total_bytes"][iteration + 1] = self.communication.total_bytes

            if hasattr(self.communication, "total_meta"):
                results_dict["total_meta"][
                    iteration + 1
                ] = self.communication.total_meta
            if hasattr(self.communication, "total_data"):
                results_dict["total_data_per_n"][
                    iteration + 1
                ] = self.communication.total_data

            if iteration == 0 or rounds_to_train_evaluate == 0:
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
                torch.save(
                    intermediate_weights,
                    os.path.join(
                        self.weights_store_dir,
                        "{}_{}_inter.pt".format(self.uid, iteration),
                    ),
                )
                torch.save(
                    post_weights,
                    os.path.join(
                        self.weights_store_dir,
                        "{}_{}_post.pt".format(self.uid, iteration),
                    ),
                )

            if self.dataset.__testing__ and rounds_to_test == 0:
                rounds_to_test = self.test_after * change
                logging.info("Evaluating on test set.")
                ta, tl = self.dataset.test(self.model, self.loss)
                results_dict["test_acc"][iteration + 1] = ta
                results_dict["test_loss"][iteration + 1] = tl
                if self.dataset.__validating__:
                    logging.info("Evaluating on the validation set")
                    va, vl = self.dataset.validate(self.model, self.loss)
                    results_dict["validation_acc"][iteration + 1] = va
                    results_dict["validation_loss"][iteration + 1] = vl

                if global_epoch == 49:
                    change *= 2

                global_epoch += change

            with open(
                os.path.join(self.log_dir, "{}_results.json".format(self.rank)), "w"
            ) as of:
                json.dump(results_dict, of)
        if self.model.shared_parameters_counter is not None:
            logging.info("Saving the shared parameter counts")
            with open(
                os.path.join(
                    self.log_dir, "{}_shared_parameters.json".format(self.rank)
                ),
                "w",
            ) as of:
                json.dump(self.model.shared_parameters_counter.numpy().tolist(), of)
        self.disconnect_neighbors()
        logging.info("Storing final weight")
        self.model.dump_weights(self.weights_store_dir, self.uid, iteration)
        logging.info("All neighbors disconnected. Process complete!")

