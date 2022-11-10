clc
clear 
close all


% To run this script we recommend running example2 first to create all the
% necessary outpouts from GLMsingle that are going to be reused here.

% This script shows how to find single-trial beta weights in the output of
% the GLMsingle. We will show how to average them to create one response
% to each condition rather then a response to each trial. This will produce
% one beta weight for each condition. Additionaly we will show how to 
% calculatea t-statistic for each condition and a contrast between two 
% example conditions in the fLoc experiment (number vs. face) using single 
% trial betas.


% load design
load('./data/nsdflocexampledataset.mat')

figure(1);clf

%Show design matricies.
for d = 1:length(design)
    subplot(2,2,d)
    imagesc(design{d}); colormap gray; drawnow
    xlabel('Conditions')
    ylabel('TRs')
    title(sprintf('Design matrix for run%i',d))
end

% There are 10 differenet conditions in this dataset.

%%
% This NSD fLOC scan session has 6 repetitions of each condition per run.
% In the code below, we are attempting to locate the indices in the beta
% weight GLMsingle outputs modelmd(x,y,z,trials) that correspond to
% repeated conditions.

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
% different condition. 

condition_ind = cell(length(condition_list),1);
for i = 1 : length(condition_list)
    
    condition_ind{i} =  find(corder == condition_list(i));
    
end
    
% Knowing the indicies of each condition we can now load an example output
% of GLMsingle and create 1 beta weight for each condition. In the example
% 1 there are 10 unique conditions.

results = load('./example2outputs/GLMsingle/TYPED_FITHRF_GLMDENOISE_RR.mat');
betas = results.modelmd;
sz = size(betas);
betas_average = zeros([sz(1) sz(2) sz(3) length(condition_ind),1]);

for i = 1 : length(condition_list)
    
    betas_average(:,:,:,i) = nanmean(betas(:,:,:,condition_ind{i}),4);
    
end

% Initially there were 240 betas (each of 10 conditions was repeated 24
% times. 

fprintf('GLMsingle output had %i betas before averaging \n',size(betas,4))

% After averaging there is 1 beta weight for each condition

fprintf('GLMsingle output has %i betas after averaging \n',size(betas_average,4))


%% Calculate t-statistic for each condition
% We calculate t-stat as the mean over the standard deviation across all
% repetitions of the same condition. 

t_stat =  zeros([sz(1) sz(2) sz(3) length(condition_ind),1]);

for n = 1 : length(condition_list)

    t_stat(:,:,:,n) = nanmean(betas(:,:,:,condition_ind{i}),4) ./ (std(betas(:,:,:,condition_ind{i}),[],4)./sqrt(length(condition_ind{i})));
    
end

%% Calculate a contrast between condition 2 (number) and condition 5 (adult face)

[~,p,~,stats] = ttest2(betas(:,:,:,condition_ind{2}),betas(:,:,:,condition_ind{5}),'dim',4);

%% Plot estiamted contrasts 
slices = [20 10];


underlay = data{1}(:,:,:,1);

things2plot = {'t_stat';'stats.tstat'};

figure(1);clf
for f = 1 : length(things2plot)
    
    slice = slices(f);
    subplot(1,2,f)
    overlay  = eval(things2plot{f});
    overlay  = overlay(:,:,slice);
    
    brainmask = (overlay < -3 | overlay > 3);

    underlay_im = cmaplookup(underlay,min(underlay(:)),max(underlay(:)),[],gray(256));
    overlay_im  = cmaplookup(overlay,-4,4,[],cmapsign2);
    
    hold on
    imagesc(squeeze(underlay_im(:,:,slice,:)));
    imagesc(overlay_im,'AlphaData',brainmask);
    axis image
    axis off
    set(gca,'FontSize',14)
    title(sprintf('%s, slice = %i',things2plot{f},slice),'Interpreter','None')
    
    colormap(cmapsign2)
    c = colorbar;
    c.Ticks = [0 0.5 1];
    c.TickLabels = {'-3';'0';'3'};
            

end

% The t-statistic (t_stat) shows high values in the visual cortex while the
% t-test between two conditions shows positive voxels in more ventral and
% lateral portions of the brain.


