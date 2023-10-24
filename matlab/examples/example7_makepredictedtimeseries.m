clc
clear
close all


% To run this script we recommend running Example1 first to create all the
% necessary outpouts from GLMsingle that are going to be reused here.

% Code here shows how to create a predicted timeseries for an example 
% voxel. This predicted timecourse is used in the GLM and is fitted to the
% data. Code below will also show the user how to concatante data and 
% designs across runs. For the purpose of this experiment we will use 
% the data from example 1, pick the voxel with highest variance explained 
% from the ON-OFF model and calculate the predicted timecourse. The
% predicted timecourse can be used for various things, e.g ilustrating 
% in the paper how well your prediction matches the fMRI time series.

% load data
load('./data/nsdflocexampledataset.mat')
% load results of TYPED model
results = load('./example2outputs/GLMsingle/TYPED_FITHRF_GLMDENOISE_RR.mat');

stimdur = 3;
tr = 1;

%%
% designSingle  - each cell holds a design marix for each run. Number of
% columns in each cell corresponds to the number of unique images showed
% during the fMRI acqusition.

% data          - is the raw time series.

% results       -  GLMsingle outpupts for the 4 models.

%%
%Find a voxel with highest variance explained of the ON-OFF model
[vexpl,ind] = max(results{1}.R2(:));
% locate it on the brain slice
[ri,ci] = ind2sub(size(results{1}.R2),ind);
% pick betas from the TYPED_FITHRF_GLMDENOISE_RR model to build the
% prediction.
betas = flatten(results{4}.modelmd(ri,ci,1,:));
% predict time series
hrflibrary = getcanonicalhrflibrary(stimdur,tr)';  % timepoinpts x HRFs
hrfii = results{4}.HRFindex(ri,ci);  % HRF index
meansignal = results{4}.meanvol(ri,ci);  % mean signal
pts = [];

temp = load('./example2outputs/GLMsingle/DESIGNINFO.mat');
designSINGLE = temp.designSINGLE;
for ii=1:length(designSINGLE)
    
    design0 = conv2(designSINGLE{ii},hrflibrary(:,hrfii));  % convolve HRF into design matrix
    design0 = design0(1:size(designSINGLE{ii},1),:);        % truncate
    betatemp = betas/100 * meansignal;
    pts = cat(1,pts,zeromean(design0*betatemp(:)));    % weight and sum
    
end


% get the demeaned raw data
dataVOX = cellfun(@(pts) zeromean(flatten(pts(ri,ci,1,:))),data,'UniformOutput',0);
dataVOX = cat(2,dataVOX{:});

figure; hold on;
plot(dataVOX,'k-');
plot(pts,'.-');
ylabel('%BOLD')
xlabel('TRs (concatanated across runs)')
set(gca,'Fontsize',15)
xlim([0 size(data{1},4)*length(data)])
xticks([0:size(data{1},4):size(data{1},4)*length(data)]);
legend({'raw timeseries';'predicted timeseries'})
legend box off
set(gcf,'Position',[1000        1047        1113         290])
% note: the influence of the glmdenoise nuisance regressors is not included here.
% also, the influence of the polynomial drift regressors is not included.
