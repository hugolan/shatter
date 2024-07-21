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
from decentralizepy.node.Hugo.Hugo_Local import Hugo_Local


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
        alphas = my_config['DATASET']['alpha'].split(',')
    except:
        alphas = [my_config['DATASET']['alpha']]

    print(alphas)
    print(learning_rates)
    print(distance_similarity_array)
    print(alternate_rounds_array)
    print(seeds)
    el_start = 0

    for distance_similarity in distance_similarity_array:
        for alpha in alphas:
            alpha = float(alpha)
            for distance_nodes in [2]:
                #distance_nodes = int(distance_nodes)
                for lr in learning_rates:
                    lr = float(lr)
                    for seed in seeds:
                        seed = int(seed)
                        #alternate log_dir = '/home/hugo/shatter/decentralizepy/eval/data/alternate_'+ str(alternate_rounds) + '_V2_' + str(lr) + "_" + str(distance_nodes) + "_" + str(distance_similarity) + "_4_1024_" + str(my_config['TRAIN_PARAMS']['rounds'])
                        log_dir = '/home/hugo/shatter/decentralizepy/eval/data/test_V3_' + str(lr) + "_" + str(distance_nodes) + "_" + str(distance_similarity) + "_" + str(my_config['TRAIN_PARAMS']['rounds']) + "_" + str(seed) + "_dirichlet" + str(alpha)
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
                        current_my_config['DATASET']['alpha'] = alpha
                        current_my_config['NODE']['graph_degree'] = my_config['NODE']['graph_degree']

                        for r in range(procs_per_machine):
                            processes.append(
                                mp.Process(
                                    target=Hugo_Local,
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
    '''
    path = "/home/hugo/shatter/decentralizepy/eval/data/Hugo_2_1024_WEL_Distance/machine0"
    #plots

    list_of_prob = []
    for n in range((args.procs_per_machine[0])):
        #path = os.path.join(args.log_dir, "{}_results.json".format(n))
        path_n = path + "/{}_results.json".format(n)
        with open(path_n, 'r') as data:
            probs = json.load(data)
        list_of_prob.append(probs["probability_matrix"]["1000"])
    list_of_prob = pd.DataFrame(list_of_prob)



    df = list_of_prob

    # Create a directed graph
    G = nx.DiGraph()

    # Add nodes
    nodes = df.columns.tolist()
    for node in nodes:
        G.add_node(node)

    # Add weighted edges based on probabilities
    for i in range(len(nodes)):
        # Sort the probabilities for each node
        sorted_probs = df.iloc[i].sort_values(ascending=False)
        # Get the top 3 outgoing edges
        top_edges = sorted_probs.iloc[:1]
        for node, weight in top_edges.items():
            if weight > 0:
                G.add_edge(nodes[i], node, weight=weight)

    # Draw the graph with edge labels
    pos = nx.spring_layout(G)  # positions for all nodes
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, font_size=15, font_weight='bold', width=2,
            arrows=True)
    #edge_labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Add legend for number of incoming edges per node
    legend_handles = []
    for node in G.nodes():
        num_incoming_edges = len(G.in_edges(node))
        legend_handles.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=10,
                                         label=f'{node}: {num_incoming_edges} incoming edges'))

    plt.legend(handles=legend_handles, loc='upper right', title='Incoming Edges per Node', fontsize=5, title_fontsize=5)

    plt.title("Gossip Network Directed Graph with Top 1 Probabilities for 2 outdegree")

    # Show the plot
    plt.savefig(path + "/gossipez.png")

    plt.show()
    '''
