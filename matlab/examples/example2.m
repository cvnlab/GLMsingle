%% Add dependencies and download the data.


% Add path to GLMsingle
addpath('./../')
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
% Call GLMestimatesingletrial with default parameters.
opt = struct('wantmemoryoutputs',[1 1 1 1]);
[results] = GLMestimatesingletrial(design,data,stimdur,tr,dataset,opt);
models.FIT_HRF = results{2};
models.FIT_HRF_GLMdenoise = results{3};
models.FIT_HRF_GLMdenoise_RR = results{4};

% Plot 1 slice of brain data
slice = 20; % adjust this number when using different datasets
val2plot = {'meanvol';'R2';'HRFindex';'FRACvalue'};
cmaps = {gray;hot;parula;copper};
figure(3);clf

for v = 1 : length(val2plot)
    
    f=subplot(2,2,v);
    imagesc(models.FIT_HRF_GLMdenoise_RR.(val2plot{v})(:,:,slice)); axis off image;
    colormap(f,cmaps{v}) % Error message is related to this line
    colorbar
    title(val2plot{v})
    set(gca,'FontSize',20)
    
end

set(gcf,'Position',[1224 840 758 408])

%%
% Run a standard GLM.
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

%%
designALL = cat(1,design{:});

% compute a vector containing 1-indexed condition numbers in chronological 
% order.

corder = [];
for p=1:size(designALL,1)
    if any(designALL(p,:))
        corder = [corder find(designALL(p,:))];
    end
end

%%

model_names = fieldnames(models);
model_names = model_names([4 1 2 3]);
% We arrange models from least to most sophisticated (for visualization
% purposes)
%%
vox_reliabilities = cell(1,length(models));
for m = 1 : length(model_names)
    

    modelmd = models.(model_names{m}).modelmd;
    
    dims = size(modelmd);
    Xdim = dims(1);
    Ydim = dims(2);
    Zdim = dims(3);
    
    cond = size(design{1},2);
    reps = dims(4)/cond;
    
    
    betas = nan(Xdim,Ydim,Zdim,reps,cond);
    
    for c = 1 : length(unique(corder))
        
        indx = find(corder == c);
        betas(:,:,:,:,c) = modelmd(:,:,:,indx);
        
    end
    
    
    ii_reps = 10;
    vox_perm = nan(1,ii_reps);
    vox_reliability = NaN(Xdim, Ydim, Zdim);
    
   
        
        for i = 1:Xdim
            for j = 1:Ydim
                for k = 1:Zdim
                    if ROI(i,j,k) == 1
                        
                        vox_data = squeeze(betas(i,j,k,:,:));
                        
                        
                        for ii_rep = 1:ii_reps
                            
                            vox_data_shuffle = vox_data;
                            
                            for c = 1 : cond
                                
                                tmp = vox_data(:,c);
                                vox_data_shuffle(:,c) =  tmp(randperm(length(tmp)));
                                
                            end
                            
                            
                            even_data = vox_data_shuffle(1,:);
                            odd_data =  vox_data_shuffle(2,:);
                            r = corr(even_data', odd_data');
                            vox_perm(ii_rep) = r;
                            
                        end
                        
                        vox_reliability(i,j,k) = nanmean(vox_perm);
                       
                    end
                end
            end
        end
    
    
    
    
    vox_reliabilities{m} = vox_reliability;
    

end

%%
%% Plot split-half reliability.

% For each model we plot the results of reliablity as an overlay.
figure(4);clf
for m = 1 : length(model_names)
    
    vox_reliability = vox_reliabilities{m};
    underlay = data{1}(:,:,20,1);
    sliceofROI = ROI(:,:,20);
    sliceofROI(sliceofROI~=1) = NaN;
    overlay = vox_reliability(:,:,20);
    
    underlay_im = cmaplookup(underlay,min(underlay(:)),max(underlay(:)),[],gray(256));
    overlay_im = cmaplookup(overlay,-0.5,0.5,[],hot(256));
    
    mask = sliceofROI==1;
        
    subplot(2,2,m);
    hold on
    imagesc(underlay_im);
    imagesc(overlay_im, 'AlphaData', mask);
    hold off

    colormap hot
    axis image  off
    c = colorbar;
    c.Label.String = 'Split-half reliability (r)';
    c.Ticks = [0 0.5 1];
    c.TickLabels = {'-0.5';'0';'0.5'};
    set(gca,'FontSize',15)
    title(model_names{m},'Interpreter','None')

end
set(gcf,'Position',[418   412   782   605])

%%
%% Compare visual voxel reliabilities between beta versions.
figure(5);clf

cmap = [0.2314    0.6039    0.6980
        0.8615    0.7890    0.2457
        0.8824    0.6863         0
        0.9490    0.1020         0];
    
% For each GLM type we calculate median reliability for voxels within the
% visual ROI and plot it as a bar plot.
for m = 1 : 4
    bar(m,nanmedian(vox_reliabilities{m}(ROI==1)),'FaceColor','None','Linewidth',3,'EdgeColor',cmap(m,:)); hold on
end
ylabel('Median reliability')
legend(model_names,'Interpreter','None','Location','NorthWest')
set(gca,'Fontsize',16)
set(gca,'TickLabelInterpreter','none')
xtickangle(0)
xticks([])
% ylim([0.1 0.2])
set(gcf,'Position',[418   412   782   605])
title('Median voxel split-half reliability of GLM models')