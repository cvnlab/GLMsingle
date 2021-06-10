%% Add dependencies and download the data.


% Add path to GLMsingle
addpath(genpath('./../utilities'))

% You also need fracridge repository to run this code
% https://github.com/nrdg/fracridge.git
% addpath('fracridge')

clear
clc
close all

dataset = 'nsdfloc';

% Download files to data directory
if ~exist('./data','dir')
    mkdir('data')
end

if  ~exist('./data/nsdflocexampledataset.mat','file')
    % download data with curl
    system('curl -L --output ./data/nsdflocexampledataset.mat https://osf.io/zxqu3/download')
end
load('./data/nsdflocexampledataset.mat')
% Data comes from subject1, fLoc session from NSD dataset.
% https://www.biorxiv.org/content/10.1101/2021.02.22.432340v1.full.pdf
%%
clc
whos

% data -> consists of several runs of 4D volume files (x,y,z,t)  where
% (t)ime is the 4th dimention.

% ROI -> manually defined region in the occipital cortex. It is a binary
% matrix where (x,y,z) = 1 corresponds to the cortical area that responded 
% to visual stimuli used in the NSD project.

fprintf('There are %d runs in total.\n',length(design));
fprintf('The dimensions of the data for the first run are %s.\n',mat2str(size(data{1})));
fprintf('The stimulus duration is %.6f seconds.\n',stimdur);
fprintf('The sampling rate (TR) is %.6f seconds.\n',tr);
%%
figure(1);clf
%Show example design matrix.

for d = 1:length(design)
    subplot(2,2,d)
    imagesc(design{d}); colormap gray; drawnow
    xlabel('Conditions')
    ylabel('TRs')
    title(sprintf('Design matrix for run%i',d))
%     axis image
end

%%
% design -> Each run has a corresponding design matrix where each column
% describes a single condition (conditions are repeated across runs). Each
% design matrix is binary with 1 specfing the time (TR) when the stimulus 
% is presented on the screen.

% In this NSD fLOC session there were 10 distinct images shown and hence
% there are 10 predictor columns/conditions. Notice that white rectangles 
% are pseudo randomized and they indicate when the presentaion of each 
% image occurs. Details of the stimulus are described here
% https://github.com/VPNL/fLoc
%%
figure(2);clf

imagesc(makeimagestack(data{1}(:,:,:,1)));
colormap(gray);
axis equal tight;
colorbar;
title('fMRI data (first volume)');
%% Call GLMestimatesingletrial with default parameters.
opt = struct('wantmemoryoutputs',[1 1 1 1]);
[results] = GLMestimatesingletrial(design,data,stimdur,tr,dataset,opt);
models.FIT_HRF = results{2};
models.FIT_HRF_GLMdenoise = results{3};
models.FIT_HRF_GLMdenoise_RR = results{4};

%% Plot 1 slice of brain data
slice = 20; % adjust this number when using different datasets
val2plot = {'meanvol';'R2';'HRFindex';'FRACvalue'};
cmaps = {gray;hot;parula;copper};
figure(3);clf

for v = 1 : length(val2plot)
    
    f=subplot(2,2,v);
    imagesc(results{4}.(val2plot{v})(:,:,slice)); axis off image;
    colormap(f,cmaps{v}) % Error message is related to this line
    colorbar
    title(val2plot{v})
    set(gca,'FontSize',20)
    
end

set(gcf,'Position',[1224 840 758 408])

%%
%% Run a standard GLM.
opt.wantlibrary = 0; % switch off HRF fitting
opt.wantglmdenoise = 0; % switch off GLMdenoise
opt.wantfracridge = 0; % switch off ridge regression
opt.wantfileoutputs = [0 0 0 0];
opt.wantmemoryoutputs = [0 1 0 0];

[ASSUME_HRF] = GLMestimatesingletrial(design,data,stimdur,tr,NaN,opt);
models.ASSUME_HRF = ASSUME_HRF{2};
%%

% Now, "models" variable holds solutions for 4 GLM models

disp(fieldnames(models))