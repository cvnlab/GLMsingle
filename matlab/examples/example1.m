%% Add dependencies and download the data.


% Add path to GLMsingle
addpath(genpath('./../'))

% You also need fracridge repository to run this code
% https://github.com/nrdg/fracridge.git
addpath(genpath('/Users/jk7127/Documents/fracridge'))

clear
clc
close all

dataset = 'nsdcore';

% Download the data to data directory
if ~exist('./data','dir')
    mkdir('data')
end

if ~exist('./data/nsdcoreexampledataset.mat','file')
    !curl -L --output ./data/nsdcoreexampledataset.mat https://osf.io/k89b2/download
end
load('./data/nsdcoreexampledataset.mat')
% Data comes from subject1, NSD01 session from NSD dataset.
% https://www.biorxiv.org/content/10.1101/2021.02.22.432340v1.full.pdf
%% Data overview.
clc
whos

% data -> Consists of several runs of 4D volume files (x,y,z,t)  where
% (t)ime is the 4th dimention. In this example data consists of only a %
% single slice and and has been prepared with a 1 sec TR.


% ROI -> Manually defined region in the occipital cortex. It is a binary
% mask where 1 corresponds to the cortical area that responded to visual
% stimuli in the NSD project

fprintf('There are %d runs in total.\n',length(design));
fprintf('The dimensions of the data for the first run are %s.\n',mat2str(size(data{1})));
fprintf('The stimulus duration is %.6f seconds.\n',stimdur);
fprintf('The sampling rate (TR) is %.6f seconds.\n',tr);

%%

figure(1);clf
%Show example design matrix.

for d = 1
    imagesc(design{d}); colormap gray; drawnow
    xlabel('Conditions')
    ylabel('TRs')
    title(sprintf('Design matrix for run%i',d))
    axis image
end


% design -> Each run has a corresponding design matrix where each colum
% describes single condition (conditions are repeated across runs). Each
% design matrix is binary with 1 specfing the time (TR) when stimulus is
% presented on the screen.

% In this NSD scan session there were 571 distinct images shown and hence
% there are 571 predictor columns. Notice that white square are pseudo
% randomized they indicate when the presentaion of each image occurs.
set(gcf,'Position',[ 1000         786         861         552])
%%
% Show an example slice of the average fMRI volume
figure(2);clf
imagesc(data{1}(:,:,:,1));
colormap(gray);
axis equal tight;
c=colorbar;
title('fMRI data (first volume)');
set(gcf,'Position',[ 1000         786         861         552])
axis off
c.Label.String = 'T2*w intensity';
set(gca,'FontSize',15)
%% Call GLMestimatesingletrial with default parameters.
% Outputs and figures will be stored in folder in the current directory or
% saved to the results variable which is the only output of
% GLMestimatesingletrial

% Optional parameters below can be assigned to a variable i.e
% opt by creating fields (i.e opt.wantlibrary). Options are the 6th input
% to GLMestimatesingletrial.

% DEFAULT OPTIONS:

% wantlibrary = 1 -> Fit hRF to each voxel
% wantglmdenoise = 1 -> Use GLMdenoise
% wantfracridge = 1  -> Use ridge regression to improve beta estimates
% chunknum = 50000 -> is the number of voxels that we will process at the
%   same time. For setups with lower memory deacrease this number.
%

% wantmemoryoutputs is a logical vector [A B C D] indicating which of the
%     four model types to return in the output <results>. The user must be careful with this,
%     as large datasets can require a lot of RAM. If you do not request the various model types,
%     they will be cleared from memory (but still potentially saved to disk).
%     Default: [0 0 0 1] which means return only the final type-D model.

% wantfileoutputs is a logical vector [A B C D] indicating which of the
%     four model types to save to disk (assuming that they are computed).
%     A = 0/1 for saving the results of the ONOFF model
%     B = 0/1 for saving the results of the FITHRF model
%     C = 0/1 for saving the results of the FITHRF_GLMDENOISE model
%     D = 0/1 for saving the results of the FITHRF_GLMDENOISE_RR model
%     Default: [1 1 1 1] which means save all computed results to disk.


% For the purpose of this example we will keep all outputs in the memory
opt = struct('wantmemoryoutputs',[1 1 1 1]);
% [results] = GLMestimatesingletrial(design,data,stimdur,tr,dataset,opt);
load results
% We assign outputs from GLMestimatesingletrial to "models" structure
models.FIT_HRF = results{2};
models.FIT_HRF_GLMDENOISE = results{3};
models.FIT_HRF_GLMDENOISE_RR = results{4};


%% Important outputs.

% R2 -> is model accuracy expressed in terms of R^2 (percentage).
% modelmd -> is the full set of single-trial beta weights (X x Y x Z x
% TRIALS). Beta weights are arranged in a chronological order)
% HRFindex -> is the 1-index of the best fit HRF. HRFs can be recovered
% with getcanonicalhrflibrary(stimdur,tr)
% FRACvalue -> is the fractional ridge regression regularization level
% chosen for each voxel.

%% Plot a slice of brain with GLMSingle outputs for FIT_HRF_GLMDENOISE_RR model.

slice = 1;
val2plot = {'modelmd';'R2';'HRFindex';'FRACvalue'};
cmaps = {hot;hot;jet;copper};
figure(3);clf

for v = 1 : length(val2plot)
    f=subplot(2,2,v);
    
    if contains('modelmd',val2plot{v})
        
        imagesc(nanmean(results{4}.(val2plot{v})(:,:,slice),4),[0 10]); axis off image;
        title('BETA WEIGHT (averaged across conditions)')
        
    else
        
        imagesc(results{4}.(val2plot{v})(:,:,slice)); axis off image;
        title(val2plot{v})
        
    end
    
    colormap(f,cmaps{v})
    colorbar
    
    set(gca,'FontSize',15)
end

set(gcf,'Position',[ 1000         786         861         552])
%% Run standard GLM.
% Additionally, for comparison purposes we are going to run standard GLM
% without hrf fitting, GLMdenoise or Ridge regression regularization.
opt.wantlibrary= 0; % switch off hrf fitting
opt.assume = 1; % assume one hrf
opt.wantglmdenoise = 0; % switch off glmdenoise
opt.wantfracridge = 0; % switch off Ridge regression
opt.wantfileoutputs =[0 0 0 0];
opt.wantmemoryoutputs =[0 1 0 0];
[ASSUME_HRF] = GLMestimatesingletrial(design,data,stimdur,tr,NaN,opt);

% We assing outputs from GLMestimatesingletrial to "models" structure
models.ASSUME_HRF = ASSUME_HRF{2};

%% Now "models" variable hold solutions for 4 GLM models
disp(fieldnames(models))
%% Compare GLM results.
% To compare the results of different GLMs we are going to calculate the
% reliablity voxel-wise index for each model. Reliablity index represents a
% correlation between beta weights for repeated presentations of the same
% stimuli. In short, we are going to check how reliable/reproducible are
% single trial responses to repeated images estimated with each GLM type.

% This NSD scan session has a large number of images that are just shown once
% during the session, some images that are shown twice, and a few that are
% shown three times. In the code below, we are attempting to locate the
% indices in the beta weight GLMsingle outputs modelmd(x,y,z,trials) that
% correspond to repated images. Here we only consider stimuli that have
% been repeated once. For the purpose of the example we ignore the 3rd
% repetition of the stimulus.

% consolidate design matrices
designALL = cat(1,design{:});

% compute a vector containing 1-indexed condition numbers in chronological order.
corder = [];
for p=1:size(designALL,1)
    if any(designALL(p,:))
        corder = [corder find(designALL(p,:))];
    end
end

% let's take a look at the first few entries
corder(1:3)

% Note that [375 497 8] means that the first stimulus trial involved
% presentation of the 375th condition, the second stimulus trial involved
% presentation of the 497th condition, and so on.

% in order to compute split-half reliability, we have to do some indexing.
% we want to find images with least two repetitions and then prepare a useful
% matrix of indices that refer to when these occur.
repindices = [];  % 2 x images containing stimulus trial indices.
% the first row refers to the first presentation;
% the second row refers to the second presentation.
for p=1:size(designALL,2)  % loop over every condition
    temp = find(corder==p);
    if length(temp) >= 2
        repindices = cat(2,repindices,[temp(1); temp(2)]);  % note that for images with 3 presentations, we are simply ignoring the third trial
    end
end

% let's take a look at a few entries
repindices(:,1:3)


fprintf('There are %i repeated images in the experiment \n',length(repindices))

% Now, for each voxel we are going to correlate beta weights describing the
% response to images presented for the first time  with beta weights
% describing the response from the repetition of the same
% image. With 136 repeated images R value for each voxel will correspond
% to correlation between vectors with 136 beta weights.

%% Calculate reliability index.

model_names = fieldnames(models);
% We arrange models from least to most sophisticated
model_names = model_names([4 1 2 3]);
vox_reliabilities = cell(1,length(models));

for m = 1 : length(model_names)
    
    % Notice that the first condition is presented on the 217th stimulus trial
    % and the 486th stimulus trial, the second condition is presented on the
    % 218th and 621st stimulus trials, and so on.
    % Finally, let's compute some split-half reliability.
    betas = models.(model_names{m}).modelmd(:,:,:,repindices);  % use indexing to pull out the trials we want
    betas_reshaped = reshape(betas,size(betas,1),size(betas,2),size(betas,3),2,[]);  % reshape to X x Y x Z x 2 x CONDITIONS
    vox_reliabilities{m} = calccorrelation(betas_reshaped(:,:,:,1,:),betas_reshaped(:,:,:,2,:),5);
    
end
%% Plot split-half reliability as an overlay.
figure(4);clf
% figure; hold on;
for m = 1 : 4
    
    vox_reliability = vox_reliabilities{m};
    underlay = data{1}(:,:,:,1);
    ROI(ROI~=1) = NaN;
    overlay = vox_reliability;
    
    im1 = cmaplookup(underlay,min(underlay(:)),max(underlay(:)),[],gray(256));
    im2 = cmaplookup(overlay,-0.5,0.5,[],hot(256));
    
    mask = ROI==1;
        
    subplot(2,2,m);
    hold on
    imagesc(im1);
    imagesc(im2, 'AlphaData', mask);
    hold off

    colormap hot
    axis image  off
    c = colorbar;
    c.Label.String = 'Split-half reliability';
    c.Ticks = [0 0.5 1];
    c.TickLabels = {'-0.5';'0';'0.5'};
    set(gca,'FontSize',15)

end
set(gcf,'Position',[ 1000         786         861         552])
%% Plot median reliability for each GLM.
figure(5);clf


cmap = [0.2314    0.6039    0.6980
    0.8615    0.7890    0.2457
    0.8824    0.6863         0
    0.9490    0.1020         0];
% For each GLM type we calculate median reliability for voxels within the
% visual ROI.

for m = 1 : 4
    bar(m,nanmedian(vox_reliabilities{m}(ROI==1)),'FaceColor','None','Linewidth',3,'EdgeColor',cmap(m,:)); hold on
end
% xticks([1:4])
ylabel('Median reliability')
legend(model_names,'Interpreter','None','Location','NorthWest')
set(gca,'Fontsize',16)
set(gca,'TickLabelInterpreter','none')
xtickangle(0)
xticks([])
ylim([0.1 0.2])
set(gcf,'Position',[ 1000         585         650         753])
title('Median voxel split-half reliability of GLM models')