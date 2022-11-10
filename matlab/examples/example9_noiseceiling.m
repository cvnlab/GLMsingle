clc
clear all
close all

% To run this script we recommend running Example1 first to create all the
% necessary outpouts from GLMsingle that are going to be reused here.

% This script shows how to calculate noise-ceiling SNR (ncsnr) for the data
% included in the NSD dataset (example1). This will produce one estimate
% for each voxel. The higher the value the higher the test-retest
% repdocucbility of estimated beta weights. In this example we analyze
% responses to stimuli that were repeated 3 times during 1st NSD session.

% load design nsdcoreexampledataset.mat from downloaded data folder.
load('./data/nsdcoreexampledataset.mat')


% load data
workdir = './example1outputs/GLMsingle/';
models = dir(sprintf('%s/*.mat',workdir));

results = cell(1,length(models));
for m = 1 : length(models)
    
results{m} = load(sprintf('%s/%s',workdir,models(m).name));

end

%%
% Consolidate design matrices
designALL = cat(1,design{:});

% Construct a vector containing 1-indexed condition numbers in
% chronological order.

corder = [];
for p=1:size(designALL,1)
    if any(designALL(p,:))
        corder = [corder find(designALL(p,:))];
    end
end


% We will now find indices for each condition. First lets find all unique
% condition in the corder list

condition_list = unique(corder);

% We will now create a structure where each cell containst indices for
% different condition. We could use a matrix to store the indices however
% sometimes the conditions are not repated the same amount of time.

condition_ind = cell(length(condition_list),1);
for i = 1 : length(condition_list)
    
    condition_ind{i} =  find(corder == condition_list(i));
    
end
    
%%

% In order to compute ncsnr, we have to do some indexing.
% we want to find images with three repetitions and then prepare a
% useful matrix of indices that refer to when these occur.
repindices = [];  % 3 x images containing stimulus trial indices.

% The first row refers to the first presentation; the second row refers to
% the second presentation and the third row refers to the third
% presentation of the same sitmulus
for p=1:size(designALL,2)  % loop over every condition
    temp = find(corder==p);
    if length(temp) == 3 % note that we only consider images that were repated 3 times
        repindices = cat(2,repindices,[temp(1); temp(2);temp(3)]);  
    end
end

%%
figure(1);clf

    
betas = results{m}.modelmd;



%%%%

% Details on the theory behind the noise ceiling calculation can be
% found in the NSD data paper (Allen et al., Nature Neuroscience, 2022).

% Here, we compute noise ceiling signal-to-noise ratio (ncsnr).

% In the NSD data paper, we recommend z-scoring each voxel's single-trial betas
% within each session to compensate for session-to-session instabilities.
% In the example below, we will omit that step just to demonstrate a more
% general implementation.

% Reorder the trials to become X x Y x Z x 3 x images where the 4th dimension
% has the 3 trial repeats for each image.
betas2 = reshape(betas(:,:,:,flatten(repindices)),size(betas,1),size(betas,2),size(betas,3),3,[]);

% Calculate the standard deviation across the 3 trials, square the result,
% average across images, and then take the square root. The result is
% the estimate of the 'noise standard deviation'.
noisesd = sqrt(mean(std(betas2,[],4).^2,5));

% Calculate the total variance of the single-trial betas.
totalvar = std(reshape(betas2,size(betas2,1),size(betas2,2),size(betas2,3),[]),[],4).^2;

% Estimate the signal variance and positively rectify.
signalvar = totalvar - noisesd.^2;
signalvar(signalvar < 0) = 0;

% Compute ncsnr as the ratio between signal standard deviation and noise standard deviation.
ncsnr = sqrt(signalvar) ./ noisesd;

% Compute noise ceiling in units of percentage of explainable variance
% for the case of 3 trials.
noiseceiling = 100 * (ncsnr.^2 ./ (ncsnr.^2 + 1/3));

%%%%


underlay = data{1}(:,:,:,1);
ROI(ROI~=1) = NaN;
overlay = noiseceiling;


underlay_im = cmaplookup(underlay,min(underlay(:)),max(underlay(:)),[],gray(256));
overlay_im  = cmaplookup(overlay,0,100,[],jet);

mask = isfinite(ROI);


% subplot(1,2,2);
hold on
imagesc(underlay_im);
imagesc(overlay_im, 'AlphaData', mask);

hold off
axis image
axis off
colormap(jet)
c = colorbar;
c.Ticks = [0 0.5 1];
c.TickLabels = {'0';'50';'100'};
title('Noise ceiling (%) in the visual cortex')
set(gca,'FontSize',15)