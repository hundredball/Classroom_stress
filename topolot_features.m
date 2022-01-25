clear variables; clc; close all;

% Read counts from mat file
% {'theta': [F, LT, C, RT, P, O], ...}
folder = './results/XGB_resting_ASR_psd/';
feature = 'XGB_feature_importances';

counts = load([folder feature '.mat']);

% Load sample data to get locations
EEG = pop_loadset('/Users/hundredball/Documents/eeglab2019_1/sample_data/eeglab_data.set');
chanlocs = EEG.chanlocs;

figure();
figure_count = 1;

% Search maximum
maximum = 0;
for band = {'theta', 'alpha', 'beta', 'gamma'}
    counts_band = counts.(band{1});
    
    if max(counts_band) > maximum
        maximum = max(counts_band);
    end
end

% Plot topoplots for each band
for band = {'theta', 'alpha', 'beta', 'gamma'}
    counts_band = counts.(band{1});
    
    subplot(2, 2, figure_count);
    topoplot( counts_band, chanlocs([4,11,14,15,22,31]) );
    title(band{1});
    
    % Set colorbar scale
    caxis([0, maximum]);
    
    figure_count = figure_count + 1;
end

% Build colorbar
colorbar('Position', [0.9, 0.3, 0.01, 0.4]);

% Build title
if contains(feature, 'XGB')
    title_type = 'feature_importances_';
else
    title_type = 'degree_of_use_';
end

axes( 'Position', [0, 0.95, 1, 0.05] ) ;
set( gca, 'Color', 'None', 'XColor', 'White', 'YColor', 'White' ) ;
text( 0.5, 0, title_type, 'FontSize', 14', 'FontWeight', 'Bold', ...
  'HorizontalAlignment', 'Center', 'VerticalAlignment', 'Bottom' ) ;

% Save figure
saveas(gcf, [folder 'feature_topoplot.png']);
