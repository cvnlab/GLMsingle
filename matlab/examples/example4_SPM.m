% This tutorial builds on Example1 and Example2 where all the processing
% steps are explained in detail. We advise to run/read them first.
% The aim of this example is to introduce users how to use GLMsingle
% with SPM formatted data. This data comes from an openneuro database
% "flanker task" https://openneuro.org/datasets/ds000102/versions/00001
% from subj08. The design matricies are already preapred in the SPM format
% and are downloaded from a sepeate SPM tutorial
% https://andysbrainbook.readthedocs.io/en/latest/fMRI_Short_Course/fMRI_01_DataDownload.html
% Note that the data has not been preprocessed and we will not show the preprocessing
% steps here. This example only shows how to prepare SPM formated data for
% GLMsingle


% Start fresh
clear
clc
close all


this_dir    = fileparts(which('example4_SPM.m'));
SPM_folder  = sprintf('%s/data/SPM_example',this_dir);
TR          = 2;
stimdur     = 2;



%% Add dependencies and download the example dataset
% Add path to GLMsingle
run(fullfile(this_dir, '..', '..', 'setup.m'));
% Name of directory to which outputs will be saved
outputdir   = fullfile(this_dir, 'exampleSPMoutput');

% Download files to data directory
input_dir = fullfile(this_dir, 'data');
if ~exist(input_dir, 'dir')
    mkdir('data')
end

URL = 'https://osf.io/yqaeb/download';
input_file = fullfile(input_dir, 'SPM.zip');

download_data(URL, input_file);
unzip(sprintf('%s',input_file),input_dir);

load(sprintf('%s/SPM.mat',SPM_folder));
tr = SPM.xY.RT;

%%
% load design matrix
design = cell(1,length(SPM.Sess));
for zz=1:length(SPM.Sess)  % for each run

  ncond = length(SPM.Sess(zz).U);    % number of conditions
  nvol = length(SPM.Sess(zz).row);   % number of volumes

  design{zz} = zeros(nvol,ncond);
  
  for yy=1:length(SPM.Sess(zz).U)    % for each condition
    design{zz}(round(SPM.Sess(zz).U(yy).ons/tr)+1,yy) = 1;  % set all of the onsets
  end

end
 
%%
% load fMRI data
data = cell(1,length(SPM.Sess));
datafiles = dir(sprintf('%s/*.gz',SPM_folder));
for zz=1:length(datafiles)
  tmp = niftiread(sprintf('%s/%s',SPM_folder,datafiles(zz).name));
  data{zz} = tmp;
end

%%
opt = struct('wantmemoryoutputs',[1 1 1 1]);
[results] = GLMestimatesingletrial(design,data,stimdur,tr,[outputdir '/GLMsingle'],opt);

