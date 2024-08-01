import json
import math
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.cm as cm

import scipy
import torch
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

from decentralizepy.datasets.CIFAR10 import LeNet
from decentralizepy.models.Model import Model

log_dir = "/home/hugo/shatter/decentralizepy/tutorial/Hugo_clusters/plots"


def average_attribute(attribute, path, number_of_nodes=16):
    values = []
    for node in range(number_of_nodes):
        with open(path + str(node) + '_results.json', 'r') as file:
            data = json.load(file)
            values.append(data[attribute])

    sums = defaultdict(int)
    counts = defaultdict(int)

    # Iterate through the list of dictionaries
    for d in values:
        for key, value in d.items():
            sums[key] += value
            counts[key] += 1

    # Calculate the average for each key
    averages = {key: sums[key] / counts[key] for key in sums}

    return averages

def get_node_accuracies(attribute, path, number_of_nodes=16):
    node_accuracies = {}
    for node in range(number_of_nodes):
        with open(path + str(node) + '_results.json', 'r') as file:
            data = json.load(file)
            node_accuracies[node] = data[attribute]
    return node_accuracies

def build_accuracies_neighbors(path, attribute, current_node, number_of_nodes=16):
    nodes_dict = group_incoming_neighbors(path, attribute, current_node, number_of_nodes=16)


    node_accuracies = {}
    for node in range(number_of_nodes):
        with open(path + str(node) + '_results.json', 'r') as file:
            data = json.load(file)
            node_accuracies[node] = data["test_acc"]

    accuracy_dict = node_accuracies[current_node]

    # Get the unique nodes and map them to indices
    all_nodes = set([i for i in range(number_of_nodes)])
    node_list = sorted(all_nodes)
    node_index = {node: idx for idx, node in enumerate(node_list)}
    total_nodes = len(node_list)

    # Sort keys
    sorted_time_steps = sorted(int(k) for k in accuracy_dict.keys())

    # Initialize result dictionary
    result_dict = {}

    # Process each 20 time step interval
    for t in sorted_time_steps:
        # Initialize count array for this interval
        node_count = [0] * total_nodes

        # Collect nodes from the last 20 time steps
        for step in range(t - 19, t + 1):
            step_str = str(step)
            if step_str in nodes_dict:
                for node in nodes_dict[step_str]:
                    node_count[node_index[node]] += 1

        # Convert counts to percentages
        node_percentage = [(count / 20) * 100 for count in node_count]

        # Get accuracy for the current time step
        accuracy = accuracy_dict[str(t)]

        # Store in the result dictionary
        result_dict[str(t)] = (accuracy, node_percentage)

    return result_dict

def standard_deviation_attribute(attribute, path, number_of_nodes=16):
    values = []
    for node in range(number_of_nodes):
        with open(path + str(node) + '_results.json', 'r') as file:
            data = json.load(file)
            values.append(data[attribute])

    # Initialize dictionaries to store sums and counts for mean calculation
    sums = defaultdict(int)
    counts = defaultdict(int)

    # Calculate the sum and count for each key
    for d in values:
        for key, value in d.items():
            sums[key] += value
            counts[key] += 1

    # Calculate the mean for each key
    means = {key: sums[key] / counts[key] for key in sums}

    # Initialize a dictionary to store the sum of squared differences from the mean
    sum_squared_diff = defaultdict(int)

    # Calculate the sum of squared differences from the mean for each key
    for d in values:
        for key, value in d.items():
            sum_squared_diff[key] += (value - means[key]) ** 2

    # Calculate the variance for each key
    variances = {key: sum_squared_diff[key] / counts[key] for key in sum_squared_diff}

    # Calculate the standard deviation for each key
    standard_deviations = {key: math.sqrt(variances[key]) for key in variances}

    return standard_deviations

def group_neighbors(path, attribute, filter = "global", number_of_nodes=16):
    values = []
    for node in range(number_of_nodes):
        if filter == "global":
            with open(path + str(node) + '_results.json', 'r') as file:
                data = json.load(file)
                values.append(data[attribute])
        else:
            if node == filter:
                with open(path + str(node) + '_results.json', 'r') as file:
                    data = json.load(file)
                    values.append(data[attribute])


    all_neighbors = defaultdict(list)
    for d in values:
        for key, value in d.items():
            if attribute == "neighbors":
                all_neighbors[key].extend(value)
            if attribute == "probability_matrix":
                all_neighbors[key].append(value)

    return all_neighbors

def group_incoming_neighbors(path, attribute,current_node, number_of_nodes=16):
    values = dict()
    for node in range(number_of_nodes):
        if node != current_node:
            with open(path + str(node) + '_results.json', 'r') as file:
                data = json.load(file)
                values[node] = data[attribute]

    incoming_neighbors = defaultdict(list)
    for node in range(number_of_nodes):
        if node != current_node:
            for key, value in values[node].items():
                if current_node in value:
                    incoming_neighbors[key].append(node)
    return incoming_neighbors

def improve_label(path):
    offset = 0
    if 'el' in path:
        return "Epidemic Learning"
    else:
        splitted_path = path.split("/")[-2].split("_")
        splitted_path[3+offset] = "distance" if splitted_path[3+offset] == "further" else "1/distance"
        ret =  "Learning rate:" + splitted_path[1+offset] + " Distance:L" + splitted_path[2+offset] + " Based on:" + splitted_path[3+offset]
        if "alternate" in path:
            ret += " Alternate: " + str(splitted_path[1])
        return ret


def build_probability_matrix(path, iteration, number_of_nodes=16):
    matrix = []
    for node in range(number_of_nodes):
        with open(path + str(node) + '_results.json', 'r') as file:
            data = json.load(file)
            matrix.append(data["probability_matrix"][iteration])

    return matrix

def aggregagte_attribute_per_lr(path, lrs, attribute, seeds, number_of_nodes=16):
    lr_accuracies = defaultdict(list)


    for folder in os.listdir(path):

        if "clusters" not in folder or "dirichlet0.7"  in folder:
            continue
        print(folder)
        #if str(lr) in folder and seed in folder and "V3" in folder and "el" not in folder and "nocom" not in folder:
        lr_accuracies[folder.split('_')[-1]].append(average_attribute(attribute, path + folder + '/'))

    print(lr_accuracies.keys())
    aggregated = {}
    for main_key, list_of_dicts in lr_accuracies.items():
        # Initialize an empty dictionary for the current main key
        aggregated[main_key] = {}
        for sub_dict in list_of_dicts:
            for sub_key, value in sub_dict.items():
                # Initialize the sub_key with an empty list if it doesn't exist
                if sub_key not in aggregated[main_key]:
                    aggregated[main_key][sub_key] = []
                # Append the value to the list
                aggregated[main_key][sub_key].append(value)

    return aggregated

'''
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
'''


def plot_accuracy_rounds(paths):
    plt.figure(figsize=(13, 8))
    for path in paths:
        data = average_attribute("test_acc", path)
        plt.plot(data.keys(), data.values(), label=improve_label(path))


    # Customize plot (optional)
    plt.title('Accuracy over time')
    plt.xlabel('rounds')

    # Set x-ticks to display every 100 iterations
    keys = list(data.keys())
    xticks = [key for key in keys if int(key) % 100 == 0]
    plt.xticks(ticks=xticks, labels=xticks, rotation=45, fontsize=10)

    plt.ylabel('accuracy')
    plt.legend()
    plt.grid()
    plt.savefig(log_dir + '/accuracy.jpg')

    return

def plot_accuracy_and_deviation(path, lrs, attribute, seeds, number_of_nodes):
    data = aggregagte_attribute_per_lr(path, lrs, attribute, seeds, number_of_nodes=number_of_nodes)
    # Plotting the mean accuracies with standard deviation shaded area for each learning rate
    plt.figure(figsize=(12, 8))

    for lr, iterations_data in data.items():
        iterations = sorted(iterations_data.keys(), key=int)
        mean_accuracies = []
        min_accuracies = []
        max_accuracies = []

        for i in iterations:
            accuracies = iterations_data[i]
            mean_accuracies.append(np.mean(accuracies))
            min_accuracies.append(np.min(accuracies))
            max_accuracies.append(np.max(accuracies))

        # Convert lists to numpy arrays for easy plotting
        iterations = np.array(iterations)
        mean_accuracies = np.array(mean_accuracies)
        min_accuracies = np.array(min_accuracies)
        max_accuracies = np.array(max_accuracies)

        # Plotting for each learning rate
        plt.plot(iterations, mean_accuracies, label=f'{lr}', linestyle='-', marker='o')
        plt.fill_between(iterations, min_accuracies, max_accuracies, alpha=0.2)


    xticks = [key for key in iterations if int(key) % 100 == 0]
    plt.xticks(ticks=xticks, labels=xticks, rotation=45, fontsize=10)
    plt.xlabel('rounds')
    plt.ylabel('Test Accuracy')
    plt.title('Test accuracy over rounds for different learning rates, shaded area covering max and min accuracy for 10 runs on each LR')
    plt.grid()
    plt.legend()
    plt.show()
    plt.savefig(log_dir + '/deviation_accuracy.jpg')
    return

def plot_boxplot_accuracy(path, lrs, attribute, seeds, number_of_nodes=16):
    learning_rates  = aggregagte_attribute_per_lr(path, lrs, attribute, seeds, number_of_nodes=16)

    time_steps  = ['20', '100','260', '500', '1000', '1600', '2000']

    # Plotting boxplots for each learning rate on separate subplots
    plt.figure(figsize=(14, 10))  # Increase figure size

    # Iterate over each learning rate
    for i, (lr, time_dict) in enumerate(learning_rates.items()):
        # Prepare data for boxplot
        lr_data = []
        for timestep in time_steps:
            accuracies = time_dict.get(timestep, [])
            lr_data.append(accuracies)

        # Create subplot
        plt.subplot(len(learning_rates), 1, i + 1)

        # Plot boxplot
        plt.boxplot(lr_data, positions=range(len(time_steps)), labels=time_steps)

        # Set title and labels
        plt.title(f"Learning Rate: {lr}")
        plt.ylabel('Test accuracy')
        plt.xlabel('Rounds')

        # Adjust font size for better readability
        plt.xticks(fontsize=10)
        plt.yticks(fontsize=10)
        plt.title('Boxplots for '+ str(lr) + ' for test accuracies for 10 run')


    plt.tight_layout()
    plt.show()

    plt.savefig(log_dir + '/boxplot_accuracy.jpg')
    return

def plot_accuracy_per_node_rounds(paths, number_of_nodes=16):
    nrows, ncols = 8, 2
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(18,10), sharex=True)
    axes = axes.flatten()

    for path in paths:
        node_accuracies = get_node_accuracies("test_acc", path, number_of_nodes)
        for node in range(number_of_nodes):
            accuracies = node_accuracies[node]
            ax = axes[node]
            ax.plot(list(accuracies.keys()), list(accuracies.values()))
            ax.set_ylabel(f'Node {node}')
            ax.legend(loc='upper right')
            # Incline x-axis ticks
            for label in ax.get_xticklabels():
                label.set_rotation(45)

    for ax in axes:
        ax.set_xlabel('Rounds')

    fig.suptitle(improve_label(path))
    plt.tight_layout()
    plt.savefig(log_dir + '/accuracy_nodes.jpg')

    return

def plot_proportion_of_models_sent_global(path,direction,current_number_of_neighbors,filter="global"):
    if direction == "outgoing":
        data = group_neighbors(path, "neighbors", filter, current_number_of_neighbors)
    elif direction == "incoming":
        data = group_incoming_neighbors(path, "neighbors", filter, current_number_of_neighbors)

    # Initialize variables
    num_neighbors = current_number_of_neighbors
    buckets = []
    bucket_size = 100
    counts = {i: 0 for i in range(num_neighbors)}

    # Iterate through data in sorted order
    for iteration in sorted(data.keys(), key=int):
        iteration = int(iteration)
        neighbors = data[str(iteration)]
        for neighbor in neighbors:
            counts[neighbor] += 1

        # Check if we've reached a new bucket boundary
        if (iteration + 1) % bucket_size == 0:
            # Calculate percentages for the current bucket
            total_counts = sum(counts.values())
            percentages = {key: (count / total_counts) * 100 for key, count in counts.items()}
            buckets.append(percentages)

            # Reset counts for the next bucket
            counts = {i: 0 for i in range(num_neighbors)}

    # If there are remaining iterations not yet added to a bucket
    if counts != {i: 0 for i in range(num_neighbors)}:
        total_counts = sum(counts.values())
        percentages = {key: (count / total_counts) * 100 for key, count in counts.items()}
        buckets.append(percentages)

    # Prepare data for plotting
    iterations = np.arange(len(buckets)) + 1
    labels = [f"Node {i}" for i in range(num_neighbors)]
    percent_data = np.array([[bucket[i] for bucket in buckets] for i in range(num_neighbors)])

    # Plotting
    plt.figure(figsize=(16, 7))

    # Use a color map to get unique colors
    colors = cm.get_cmap('tab20', num_neighbors)
    hatches = ['/', '\\', '|'] * (num_neighbors // 3 + 1)

    # Stacked bar chart
    bottom = np.zeros(len(buckets))
    bars = []
    for i in range(num_neighbors):
        bar = plt.bar(iterations, percent_data[i], label=labels[i], bottom=bottom, color=colors(i), hatch=hatches[i])
        bars.append(bar)
        bottom += percent_data[i]

    # Add text annotations
    for bar_group, percent_group in zip(bars, percent_data):
        for bar, percentage in zip(bar_group, percent_group):
            if percentage > 0:  # Only add text for non-zero percentages
                plt.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() / 2 + bar.get_y(),
                    f'{percentage:.1f}%',
                    ha='center', va='center', fontsize=8, color='white'
                )

    plt.title('Percentage of Each Neighbor Over 100 Iteration Buckets for ' + improve_label(path))
    plt.xlabel('Bucket of 100 Iterations')
    plt.ylabel('Percentage (%)')
    plt.xticks(iterations, labels=[f'{i * bucket_size + 1}-{(i + 1) * bucket_size}' for i in range(len(buckets))],
               rotation=45, ha='right')
    plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1))
    plt.tight_layout()
    plt.show()
    plt.savefig(log_dir + '/' + direction + 'neighbors_percent_' + str(filter) + '.png')
    return


def plot_neighbors_KL(path,filter="global"):
    data_outgoing = group_neighbors(path, "neighbors", filter=filter)
    data_incoming = group_incoming_neighbors(path, "neighbors", current_node=filter)

    # Plotting
    plt.figure(figsize=(16, 10))
    # Initialize variables
    num_neighbors = 16
    buckets_outgoing = []
    bucket_size = 100
    counts = {i: 0 for i in range(num_neighbors)}

    # Iterate through data in sorted order
    for iteration in sorted(data_outgoing.keys(), key=int):
        iteration = int(iteration)
        neighbors = data_outgoing[str(iteration)]
        for neighbor in neighbors:
            counts[neighbor] += 1

        # Check if we've reached a new bucket boundary
        if (iteration + 1) % bucket_size == 0:
            # Calculate percentages for the current bucket
            total_counts = sum(counts.values())
            percentages = {key: (count / total_counts) * 100 for key, count in counts.items()}
            buckets_outgoing.append(percentages)

            # Reset counts for the next bucket
            counts = {i: 0 for i in range(num_neighbors)}

    # If there are remaining iterations not yet added to a bucket
    if counts != {i: 0 for i in range(num_neighbors)}:
        total_counts = sum(counts.values())
        percentages = {key: (count / total_counts) * 100 for key, count in counts.items()}
        buckets_outgoing.append(percentages)

    buckets_incoming = []

    # Iterate through data in sorted order
    for iteration in sorted(data_incoming.keys(), key=int):
        iteration = int(iteration)
        neighbors = data_incoming[str(iteration)]
        for neighbor in neighbors:
            counts[neighbor] += 1

        # Check if we've reached a new bucket boundary
        if (iteration + 1) % bucket_size == 0:
            # Calculate percentages for the current bucket
            total_counts = sum(counts.values())
            percentages = {key: (count / total_counts) * 100 for key, count in counts.items()}
            buckets_incoming.append(percentages)

            # Reset counts for the next bucket
            counts = {i: 0 for i in range(num_neighbors)}

    # If there are remaining iterations not yet added to a bucket
    if counts != {i: 0 for i in range(num_neighbors)}:
        total_counts = sum(counts.values())
        percentages = {key: (count / total_counts) * 100 for key, count in counts.items()}
        buckets_incoming.append(percentages)

    percent_data_outgoing = np.array([list(bucket.values()) for bucket in buckets_outgoing])
    percent_data_incoming = np.array([list(bucket.values()) for bucket in buckets_incoming])



    # Calculate KL divergence for each timestamp
    kl_divergences = []
    for timestamp in range(len(percent_data_outgoing)):
        dist_1 = percent_data_outgoing[timestamp]
        dist_2 = percent_data_incoming[timestamp]
        kl_div = scipy.stats.entropy(dist_1, dist_2)
        kl_divergences.append(kl_div)


    # Plot the KL divergence over time
    plt.plot(kl_divergences[:len(kl_divergences)], marker='o')
    plt.xlabel('Timestamp')
    plt.xticks([i for i in range(len(kl_divergences))] , labels=[i*100 for i in range(len(kl_divergences))],rotation=45)
    plt.ylabel('KL Divergence')
    plt.title('KL Divergence Over Time')
    plt.show()
    plt.savefig(log_dir + '/neighbors_KL_' + str(filter) + '.png')
    return




def plot_sd_accros_nodes(paths):
    plt.figure(figsize=(15, 10))
    for path in paths:
        data = standard_deviation_attribute("test_acc",path)
        plt.plot(data.keys(), data.values(), label=improve_label(path))

    # Customize plot (optional)
    plt.title('Sd over time')
    plt.xlabel('rounds')
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('sd')
    plt.legend()
    plt.savefig(log_dir + '/sd.jpg')
    return

def plot_boxplot_of_accuracy_over_time():

    return

def plot_pairwise_divergence_of_models(path, number_of_nodes=16):
    plt.figure(figsize=(20, 15))
    data = group_neighbors(path, "probability_matrix")
    div = dict()
    for key in data.keys():
        sum = 0
        count = 0
        for i in range(0, number_of_nodes):
            for j in range(i+1, number_of_nodes):
                print(data[key][i])
                sum += scipy.stats.entropy(data[key][i],data[key][j], base=None)
                count +=1
        div[key] = sum

    plt.plot(div.keys(), div.values(), label=path)
    plt.title('Pairwise KL over time')
    plt.xlabel('rounds')
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(log_dir + '/pairwise_kl.png')

    return

def plot_pairwise_divergence_of_models(path, number_of_nodes=16):
    plt.figure(figsize=(20, 15))
    data = group_neighbors(path, "model_weights")
    div = dict()
    for key in data.keys():
        sum = 0
        count = 0
        for i in range(0, number_of_nodes):
            for j in range(i+1, number_of_nodes):
                print(data[key][i])
                sum += scipy.stats.entropy(data[key][i],data[key][j], base=None)
                count +=1
        div[key] = sum

    plt.plot(div.keys(), div.values(), label=path)
    plt.title('Pairwise KL over time')
    plt.xlabel('rounds')
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('accuracy')
    plt.legend()
    plt.savefig(log_dir + '/pairwise_kl.png')

    return

def plot_kl_divergence_of_probability_distributions(paths, number_of_nodes=16):
    plt.figure(figsize=(15, 10))
    for path in paths:
        data = group_neighbors(path, "probability_matrix")
        steps = (list(data.keys()))
        steps = [int(x) for x in steps]
        steps = sorted(steps)
        values_div_kl = dict()
        for key in range(len(steps) - 1):
            sum_div = 0
            for node in range(number_of_nodes):
                sum_div += scipy.stats.entropy(data[str(steps[key])][node],data[str(steps[key+1])][node], base=None)
            values_div_kl[steps[key]] = sum_div

        plt.plot(values_div_kl.keys(), values_div_kl.values(),  label=improve_label(path))

    plt.title('Sum of KL divergence between two time steps over time')
    plt.xlabel('rounds')
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('Sum of KL divergence')
    plt.legend()
    plt.savefig(log_dir + '/over_time_kl.png')

    return
def plot_heatmap_of_models_dvergence():

    return

def plot_most_models_sent_to():

    return

def plot_of_node_received_most_with_diff():

    return

def plot_average_number_of_neighbors():

    return

def plot_distance_same_models(path):
    plt.figure(figsize=(15, 10))
    sum_weight_diff = defaultdict(list)
    for node in range(16):
        for iter in range(100,1999,100):
            model_1 = LeNet()
            model_1.load_state_dict(torch.load(path + 'models/' + str(node) + "_" + str(iter)  + "_weights.pt"))
            model_2 = LeNet()
            model_2.load_state_dict(torch.load(path + 'models/' + str(node) + "_" + str(iter + 100)  + "_weights.pt"))
            sum_weight_diff[node].append(torch.dist(model_1.get_weights(),model_2.get_weights()))

    for key, value in zip(sum_weight_diff.keys(), sum_weight_diff.values()):
        plt.plot(value, label=key)
    plt.title('distance same model over time')
    plt.xlabel('100 rounds')
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('L2 dist')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(log_dir + '/distance_same_models.png')
    return

def plot_distance_inter_models(path):
    plt.figure(figsize=(15, 10))
    sum_weight_diff = defaultdict(list)
    for iter in range(100, 1999, 100):
        sum_dist = 0.0
        for node in range(16):
            for node2 in range(node+1,16):
                model_1 = LeNet()
                model_1.load_state_dict(torch.load(path + 'models/' + str(node) + "_" + str(iter)  + "_weights.pt"))
                model_2 = LeNet()
                model_2.load_state_dict(torch.load(path + 'models/' + str(node2) + "_" + str(iter)  + "_weights.pt"))
                sum_dist +=  torch.dist(model_1.get_weights(),model_2.get_weights())
        sum_weight_diff[iter] = sum_dist

    plt.plot(sum_weight_diff.keys(), sum_weight_diff.values())
    plt.title('distance inter model over time')
    plt.xlabel('rounds')
    plt.xticks(rotation=45, fontsize=10)
    plt.ylabel('L2 dist')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(log_dir + '/distance_inter_models.png')
    return


def plot_heatmap_probability_matrix(path):
    iterations = ["20", "40", "100", "260", "500", "1000"]#, "1200", "1520", "2000"]
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 18))
    axes = axes.flatten()

    for ax, iteration in zip(axes, iterations):
        matrix = build_probability_matrix(path, iteration=iteration)
        matrix_df = pd.DataFrame(matrix)
        sns.heatmap(matrix_df, ax=ax, cbar=False)
        ax.set_title(f'Iteration {iteration}')

    plt.tight_layout()
    plt.suptitle('Gossip Matrix for different iterations', y=1.02)
    plt.savefig(log_dir + "/gossip_matrix_iterations.png")
    return

def plot_accuracy_corr(path, current_node, total_nodes=16):
    result_dict = build_accuracies_neighbors(path, "neighbors", current_node)
    differences_dict = {}
    node_list = sorted([i for i in range(total_nodes)])

    # Process each consecutive interval
    previous_key = None
    for key in sorted(result_dict.keys(), key=int):
        if previous_key is not None:
            # Calculate accuracy difference
            accuracy_diff = result_dict[key][0] - result_dict[previous_key][0]

            # Calculate percentage differences for each node
            percentage_diff = [
                result_dict[key][1][i] - result_dict[previous_key][1][i]
                for i in range(total_nodes)
            ]

            # Store in differences dictionary
            differences_dict[accuracy_diff] = percentage_diff

        previous_key = key

    # Prepare data for correlation analysis
    accuracy_diffs = list(differences_dict.keys())
    percentage_diffs = np.array(list(differences_dict.values()))

    # Initialize a dictionary to store correlation results
    correlation_results = {}

    # Calculate correlation for each node's percentage difference with accuracy difference
    for i in range(total_nodes):
        node_percentage_diffs = percentage_diffs[:, i]

        # Calculate Pearson and Spearman correlations
        pearson_corr, _ = pearsonr(accuracy_diffs, node_percentage_diffs)
        spearman_corr, _ = spearmanr(accuracy_diffs, node_percentage_diffs)

        # Store the results
        correlation_results[node_list[i]] = {
            'pearson_correlation': pearson_corr,
            'spearman_correlation': spearman_corr
        }

    # Plot the results
    pearson_corrs = [correlation_results[node]['pearson_correlation'] for node in node_list]
    spearman_corrs = [correlation_results[node]['spearman_correlation'] for node in node_list]

    x = np.arange(total_nodes)
    width = 0.35  # width of the bars

    fig, ax = plt.subplots(figsize=(15, 6))
    bars1 = ax.bar(x - width / 2, pearson_corrs, width, label='Pearson')
    bars2 = ax.bar(x + width / 2, spearman_corrs, width, label='Spearman')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_xlabel('Node')
    ax.set_ylabel('Correlation Coefficient')
    ax.set_title('Correlation between Percentage Changes and Accuracy Changes')
    ax.set_xticks(x)
    ax.set_xticklabels(node_list)
    ax.legend()

    # Attach a text label above each bar in *bars*, displaying its height.
    def autolabel(bars, offset):
        """Attach a text label above each bar in *bars*, displaying its height."""
        for bar in bars:
            height = bar.get_height()
            ax.annotate('%.2f' % height,
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(offset, 3),  # offset points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')

    autolabel(bars1, -10)
    autolabel(bars2, 10)

    fig.tight_layout()

    plt.show()
    plt.savefig(log_dir + '/correlation.png')
    return

if __name__ == "__main__":
    seeds = [90,42,123,789,1001,5555,8675309,314159]
    seeds = [str(i) for i in seeds]
    lrs = [0.0125,0.025,0.05,0.1]

    print("start building plots")
    data_path = "/home/hugo/shatter/decentralizepy/eval/data"
    #softmax
    #data_sources = ["/home/hugo/shatter/decentralizepy/eval/data/softmax_90_V2_0.025_2_closer_4_1024_10/","/home/hugo/shatter/decentralizepy/eval/data/softmax_90_V2_0.025_2_further_4_1024_10/", "/home/hugo/shatter/decentralizepy/eval/data/el_4degree_2048_10rounds/machine0/"]
    #32 nodes
    #data_sources = ["/home/hugo/shatter/decentralizepy/eval/data/32_90_V2_0.025_2_closer_4_1024_10/","/home/hugo/shatter/decentralizepy/eval/data/32_90_V2_0.025_2_further_4_1024_10/","/home/hugo/shatter/decentralizepy/eval/data/el_5degree_2048_10rounds/machine0/"]
    #benchmark
    data_sources = ["/home/hugo/shatter/decentralizepy/eval/data/V3_0.1_2_further_10_90/"]
    #10K
    #data_sources = ["/home/hugo/shatter/decentralizepy/eval/data/90_V2_0.025_2_closer_4_10020_10/","/home/hugo/shatter/decentralizepy/eval/data/90_V2_0.025_2_further_4_10020_10/",]#"/home/hugo/shatter/decentralizepy/eval/data/el_4degree_10020_10rounds/machine0/"]
    #KL distance
    #data_sources = ["/home/hugo/shatter/decentralizepy/eval/data/90_V2_0.025_kl_distance_closer_4_1024_10/","/home/hugo/shatter/decentralizepy/eval/data/90_V2_0.025_kl_distance_further_4_1024_10/"]#, "/home/hugo/shatter/decentralizepy/eval/data/el_4degree_2048_10rounds/machine0/"]
    #alternate
    #data_sources = ["/home/hugo/shatter/decentralizepy/eval/data/alternate_2_V2_0.025_2_further_4_1024_10/","/home/hugo/shatter/decentralizepy/eval/data/alternate_4_V2_0.025_2_further_4_1024_10/","/home/hugo/shatter/decentralizepy/eval/data/alternate_3_V2_0.025_2_further_4_1024_10/","/home/hugo/shatter/decentralizepy/eval/data/el_4degree_2048_10rounds/machine0/"]
    #data_sources = ["/home/hugo/shatter/decentralizepy/eval/data/alternate_2_V2_0.025_2_further_4_1024_10/","/home/hugo/shatter/decentralizepy/eval/data/alternate_2_V2_0.025_2_closer_4_1024_10/"]
    #weight decay
    #data_sources = ["/home/hugo/shatter/decentralizepy/eval/data/weight_decay_V2_0.025_2_further_4_1024_10/","/home/hugo/shatter/decentralizepy/eval/data/weight_decay_el/machine0/"]
    #plot_accuracy_rounds(data_sources)
    plot_accuracy_and_deviation("/home/hugo/shatter/decentralizepy/eval/data/", lrs, "test_acc", seeds, number_of_nodes=32)

    data_look_into = "/home/hugo/shatter/decentralizepy/eval/data/V3_0.05_2_further_10_42/"
    neighbors = 32
    node = 1
    #plot_heatmap_probability_matrix(data_look_into)
    #plot_proportion_of_models_sent_global(data_look_into, "outgoing", neighbors, node)
    #plot_proportion_of_models_sent_global(data_look_into, "incoming", neighbors, node)
    #plot_neighbors_KL(data_look_into, neighbors, node)
    #plot_accuracy_corr(data_look_into, node)
    #plot_accuracy_and_deviation("/home/hugo/shatter/decentralizepy/eval/data/", lrs, "test_acc", seeds, number_of_nodes=16)
    #plot_boxplot_accuracy("/home/hugo/shatter/decentralizepy/eval/data/", lrs, "test_acc", seeds, number_of_nodes=16)
    #plot_sd_accros_nodes(data_sources)
    #plot_accuracy_per_node_rounds(data_sources)
    #plot_kl_divergence_of_probability_distributions(data_sources)
    #plot_distance_same_models("/home/hugo/shatter/decentralizepy/eval/data/V2_0.025_2_further_4_1024_10/")
    #plot_distance_inter_models("/home/hugo/shatter/decentralizepy/eval/data/V2_0.025_2_further_4_1024_10/")