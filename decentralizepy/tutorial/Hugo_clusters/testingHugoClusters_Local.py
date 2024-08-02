import json
import logging
import os
from pathlib import Path
from shutil import copy

import networkx as nx
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from localconfig import LocalConfig
from torch import multiprocessing as mp

from decentralizepy import utils
from decentralizepy.graphs.Graph import Graph
from decentralizepy.mappings.Linear import Linear
from decentralizepy.node.Hugo_cluster.HugoClusters_Local import HugoClusters_Local


def read_ini(file_path):
    config = LocalConfig(file_path)
    for section in config:
        #print("Section: ", section)
        for key, value in config.items(section):
            None
            #print((key, value))
    #print(dict(config.items("DATASET")))
    return config


if __name__ == "__main__":
    args = utils.get_args()

    log_level = {
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    config = read_ini(args.config_file)
    my_config = dict()
    for section in config:
        my_config[section] = dict(config.items(section))


    try:
        learning_rates = my_config['OPTIMIZER_PARAMS']['lr'].split(',')
    except:
        learning_rates = [my_config['OPTIMIZER_PARAMS']['lr']]
    #el_starting_points = my_config['PARAMS']['el_start'].split(',')
    #distance_nodes_array = my_config['PARAMS']['distance_nodes'].split(',')
    try:
        distance_similarity_array = my_config['PARAMS']['distance_similarity'].split(',')
    except:
        distance_similarity_array = [my_config['PARAMS']['distance_similarity']]

    try:
        alternate_rounds_array = my_config['PARAMS']['alternate_rounds'].split(',')
    except:
        alternate_rounds_array = [my_config['PARAMS']['alternate_rounds']]

    try:
        seeds = my_config['DATASET']['random_seed'].split(',')
    except:
        seeds = [my_config['DATASET']['random_seed']]

    try:
        alphas_1 = my_config['DATASET']['alpha_1'].split(',')
    except:
        alphas_1 = [my_config['DATASET']['alpha_1']]

    try:
        alphas_2 = my_config['DATASET']['alpha_2'].split(',')
    except:
        alphas_2 = [my_config['DATASET']['alpha_2']]


    print(alphas_1)
    print(alphas_2)
    print(learning_rates)
    print(distance_similarity_array)
    print(alternate_rounds_array)
    print(seeds)
    el_start = 0

    for distance_similarity in distance_similarity_array:
        for alpha_1 in alphas_1:
            alpha_1 = float(alpha_1)
            for alpha_2 in alphas_2:
                alpha_2 = float(alpha_2)
                for distance_nodes in [2]:
                    #distance_nodes = int(distance_nodes)
                    for lr in learning_rates:
                        lr = float(lr)
                        for seed in seeds:
                            seed = int(seed)
                            #alternate log_dir = '/home/hugo/shatter/decentralizepy/eval/data/alternate_'+ str(alternate_rounds) + '_V2_' + str(lr) + "_" + str(distance_nodes) + "_" + str(distance_similarity) + "_4_1024_" + str(my_config['TRAIN_PARAMS']['rounds'])
                            log_dir = '/home/hugo/shatter/decentralizepy/eval/data/clusters_V3_' + str(lr) + "_" + str(distance_similarity) + "_" + str(my_config['TRAIN_PARAMS']['rounds']) + "_" + str(seed) + "_dirichlet1=" + str(alpha_1) + "_dirichlet2=" + str(alpha_2) + "_nodes=" + str(my_config['NODE']['graph_degree']) + "_degree=" + str( my_config['NODE']['graph_degree'])
                            if int(my_config['NODE']['graph_degree']) == 0:
                                log_dir += '_nocom'
                            args.log_dir = log_dir
                            Path(args.log_dir).mkdir(parents=True, exist_ok=True)

                            #copy(args.config_file, args.log_dir)
                            #copy(args.graph_file, args.log_dir)
                            utils.write_args(args, args.log_dir)

                            g = Graph()
                            g.read_graph_from_file(args.graph_file, args.graph_type)
                            n_machines = args.machines
                            procs_per_machine = args.procs_per_machine[0]

                            l = Linear(n_machines, procs_per_machine)
                            m_id = args.machine_id
                            processes = []

                            current_my_config = my_config.copy()
                            current_my_config['DATASET']['random_seed'] = seed
                            current_my_config['OPTIMIZER_PARAMS']['lr'] = lr
                            current_my_config['PARAMS']['el_start'] = el_start
                            current_my_config['PARAMS']['distance_nodes'] = distance_nodes
                            current_my_config['PARAMS']['distance_similarity'] = distance_similarity
                            #current_my_config['PARAMS']['alternate_rounds'] = alternate_rounds
                            current_my_config['DATASET']['alpha_1'] = alpha_1
                            current_my_config['DATASET']['alpha_2'] = alpha_2
                            current_my_config['NODE']['graph_degree'] = my_config['NODE']['graph_degree']

                            for r in range(procs_per_machine):
                                processes.append(
                                    mp.Process(
                                        target=HugoClusters_Local,
                                        args=[
                                            r,
                                            m_id,
                                            l,
                                            g,
                                            current_my_config,
                                            args.iterations,
                                            args.log_dir,
                                            args.weights_store_dir,
                                            log_level[args.log_level],
                                            args.test_after,
                                            args.train_evaluate_after,
                                            args.reset_optimizer,
                                        ],
                                    )
                                )



                            for p in processes:
                                p.start()

                            for p in processes:
                                p.join()


    print("end")

