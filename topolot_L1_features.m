clear variables; clc; close all;

% Read counts from mat file
% {'theta': [F, LT, C, RT, P, O], ...}
folder = './results/L1_BP/';
counts = load([folder 'L1_features.mat']);

% Load sample data to get locations
EEG = pop_loadset('/Users/hundredball/Documents/eeglab2019_1/sample_data/eeglab_data.set');
chanlocs = EEG.chanlocs;

% Plot topoplots for each band
for band = {'theta', 'alpha', 'beta', 'gamma'}
    counts_band = counts.(band{1});
    
    figure('Name', band{1});
    topoplot(counts_band, chanlocs([4,11,14,15,22,31]));
    saveas(gcf, [folder band{1} '.png']);
end