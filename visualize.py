import numpy as np
import matplotlib.pyplot as plt


def plot_XGB_feature_importance(feature_importance, folder, xaxis='Hz', error_bar=False):
    """
    Plot average feature importance for each frequency bin for each region
    """

    regions = ['F', 'LT', 'C', 'RT', 'P', 'O']
    regions_subplots = {'F': (0, 1), 'LT': (1, 0), 'C': (1, 1), 'RT': (1, 2), 'P': (2, 1), 'O': (3, 1)}

    # Features of regions
    fig, axs = plt.subplots(4, 3, figsize=(10, 14))

    # Remove axis for all subplots
    for row in range(4):
        for col in range(3):
            axs[row, col].axis('off')

    freq = feature_importance['freq']
    max_values = []
    for i_region, region in enumerate(regions):
        position = regions_subplots[region]
        feature_importance_mean = np.mean(feature_importance[region], axis=0)
        feature_importance_std = np.mean(feature_importance[region], axis=0)

        if error_bar:
            axs[position].bar(freq, feature_importance_mean, yerr=feature_importance_std)
        else:
            axs[position].bar(freq, feature_importance_mean)
        axs[position].axis('on')
        axs[position].set_title(region)
        axs[position].set_ylim(0, 15)
        if xaxis == 'Hz':
            axs[position].set_xticks(np.arange(min(freq), max(freq) + 1, 5.0))

        max_values.append(np.max(feature_importance_mean+feature_importance_std))

    # Unify limit of y-axis
    max_y = max(max_values)
    for region in regions:
        position = regions_subplots[region]
        axs[position].set_ylim(0, max_y)

    fig.suptitle(
        'Average XGB feature importance across {num_folds} folds'.format(num_folds=len(feature_importance['F'])))
    fig.text(0.5, 0.05, 'Frequency ({})', ha='center'.format(xaxis))
    fig.text(0.05, 0.5, 'Feature Importance', va='center', rotation='vertical')
    fig.savefig('./results/{folder}/XGB_feature_importance_barplot_{xaxis}.png'.format(folder=folder, xaxis=xaxis))
