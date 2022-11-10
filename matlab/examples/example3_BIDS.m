%% BIDS Example Overview
% This tutorial builds on Example1 and Example2 where all the processing
% steps are explained in detail. We advise to run/read them first.
% The aim of this example is to introduce users how to use GLMsingle
% with BIDS formatted data. This data comes from an openneuro database 
% "study forest" https://openneuro.org/datasets/ds000113/versions/1.3.0
% from subj01 from an auditory perception task. To descrease
% the processing time we have selected 1 slice of data (17).

% Start fresh
clear
clc
close all


BIDS_folder = sprintf('%s/data/BIDS',this_dir);
subj        = '01';
ses         = 'auditoryperception';
task        = 'auditoryperception';
tr          = 2;


%% Add dependencies and download the example dataset
this_dir    = fileparts(which('example3_BIDS.m'));
% Add path to GLMsingle
run(fullfile(this_dir, '..', '..', 'setup.m'));
% Name of directory to which outputs will be saved
outputdir   = fullfile(this_dir, 'exampleBIDSoutput');

% Download files to data directory
input_dir = fullfile(this_dir, 'data');
if ~exist(input_dir, 'dir')
    mkdir('data')
end

URL = 'https://osf.io/deazx/download';
input_file = fullfile(input_dir, 'BIDS.zip');

download_data(URL, input_file);
unzip(sprintf('%s',input_file),input_dir);


%% find functional data inside BIDS directory
func_folder = sprintf('%s/subj-%s/ses-%s/func/',BIDS_folder,subj,ses);
runs        = dir(sprintf('%s/*%s*.*gz',func_folder,task)); 
% runs are ordered from 1-8
data        = cell(1,length(runs));

% find design files 
designPath = func_folder;
design        = cell(1,length(runs));

for r = 1 : length(runs)     
    data{r} = niftiread([func_folder filesep runs(r).name]);
end


runnum = 1 : length(runs);  
n = length(runnum);
%% load BIDS-formatted tsv files (experimantal design files).

TR = zeros(n,1);
numvol = zeros(n,1);
T = cell(n,1);
scan = 1;

    for jj = 1:length(runnum)

        TR(scan)     = tr;
        numvol(scan) = size(data{1},4);
        % TSV
        prefix = sprintf('sub-%s_ses-%s_task-%s_run-0%d', ...
            subj, ses, task, runnum(jj));
        tsvfile  = sprintf('%s_events.tsv', prefix);        
        assert(exist(fullfile(func_folder,tsvfile), 'file')>0)  
        T{scan}      = tdfread(fullfile(func_folder,tsvfile));
        scan = scan+1;
    end


all_trial_types = [];
for ii = 1:n    
    all_trial_types = cat(1, all_trial_types, T{ii}.trial_type);
end

unique_conditions = unique(all_trial_types,'rows');
num_conditions = size(unique_conditions,1);


% In this dataset stimulus is always presented for 6s
stimdur = unique(cell2mat(cellfun(@(x) x.duration,T,'UniformOutput',false)));

%% Loop over all runs and make each design matrix
for ii = 1:n   
    
    m = zeros(numvol(ii), num_conditions);
    these_conditions = T{ii}.trial_type;    
    [~,col_num] = ismember(these_conditions, unique_conditions,'rows');
    
    % time in seconds of start of each event
    row_nums = round(T{ii}.onset / TR(ii))+1;
    linearInd = sub2ind(size(m), row_nums, col_num);
    m(linearInd) = 1;
    design{ii} = m;
end

%% 
clc
% whos

% data -> consists of several runs of 4D volume files (x,y,z,t)  where
% (t)ime is the 4th dimention. In this example, data consists of only a
% single slice and has been prepared with a TR = 2s


fprintf('There are %d runs in total.\n',length(design));
fprintf('The dimensions of the data for the first run are %s.\n',mat2str(size(data{1})));
fprintf('The stimulus duration is %.3f seconds.\n',stimdur);
fprintf('The sampling rate (TR) is %.3f seconds.\n',tr);

%%
figure(1);clf

%Show example design matrix.
for d = 1
    imagesc(design{d}); colormap gray; drawnow
    xlabel('Conditions')
    ylabel('TRs')
    title(sprintf('Design matrix for run%i',d))
%     axis image
end

xticks(1:1:num_conditions);
xticklabels(unique_conditions);

%% run GLMsingle
opt = struct('wantmemoryoutputs',[1 1 1 1]);
[results] = GLMestimatesingletrial(design,data,stimdur,tr,[outputdir '/GLMsingle'],opt);


%% Plot a slice of brain with GLMsingle outputs

models.FIT_HRF_GLMdenoise_RR = results{4};
% We are going to plot several outputs from the FIT_HRF_GLMdenoise_RR GLM,
% which contains the full set of GLMsingle optimizations.

% we will plot betas, R2, optimal HRF indices, and the voxel frac values
val2plot = {'modelmd';'R2';'HRFindex';'FRACvalue'};
cmaps = {cmapsign2;hot;jet;copper};

% Mask out voxels that are outside the brain
brainmask = models.FIT_HRF_GLMdenoise_RR.meanvol > 50;

figure(3);clf

for v = 1 : length(val2plot)
    f=subplot(2,2,v);
    
    % Set non-brain voxels to nan to ease visualization
    plotdata = models.FIT_HRF_GLMdenoise_RR.(val2plot{v});
    
    if contains('modelmd',val2plot{v})
        toplot = nanmean(plotdata,4);
        toplot(~brainmask) = NaN;

        imagesc(nanmean(toplot,4),[-5 5]); axis off image;
        title(sprintf('Average GLM betas (%i stimuli)',sum(sum(cat(1,design{:})))))
    else
         toplot = plotdata;
         toplot(~brainmask) = NaN;

        imagesc(toplot); axis off image;
        title(val2plot{v})
    end
    colormap(f,cmaps{v})
    colorbar
    set(gca,'FontSize',15)
end

set(gcf,'Position',[418   412   782   605])
