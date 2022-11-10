clc
clear all
close all

% To run this script we recommend running example2 first to create all the
% necessary outpouts from GLMsingle that are going to be reused here.

% This script shows how to save GLMsingle outputs as NIFTI files. NIFTI is 
% neuroimaging format often used to store neuroimaging data. This script 
% requires a NiFTi toolbox
% https://www.mathworks.com/matlabcentral/fileexchange/8797-tools-for-nifti-and-analyze-image
% or alternatively one can use matlab's native support for nifti files
% (explaiend at the end of the script).

% load GLMsingle results
load('./example2outputs/GLMsingle/TYPED_FITHRF_GLMDENOISE_RR.mat')


% we will save following estiamtes (modelmd corresponds to single trial betas)
val2save = {'modelmd';'R2';'HRFindex';'FRACvalue'};
niftidir = './example2outputs/GLMsingle/GLMDENOISE_RR_nifti_files';
mkdir(niftidir);

voxelsize = [1.8 1.8 1.8];
origin = [0 0 0];

for v = 1 : length(val2save)
    

    nii = make_nii(eval(val2save{v}),voxelsize,origin);
    save_nii(nii,sprintf('%s/%s.nii',niftidir,val2save{v}))
    
    
end

% if you don't want to change the reference frame and position of the data
% (done by setting a new origin) you can reuse a NIFTI file (e.g T1 image)
% as a template for saving your results. This is usefull if your functional
% data is already aligned with the anatomy.

% template = load_nii('T1.nii')
% template is a matlab structure where "vol" field stores the data matrix.
% template.vol = eval(val2save{v}); swap the data to your GLMsingle output
% save_nii(template,sprintf('%s/%s.nii',niftidir,val2save{v})); save nifti

% additionally you can use matlab's native support for reading and saving
% nifti files (niftiread and niftiwrite). This skips making the nifti file
% and the header information can be directly added to niftiwrite function. 
% info.Description = 'Modified using MATLAB R2017b';
% niftiwrite(nii,sprintf('%s/%s.nii',niftidir,val2save{v})), info);
% where info is a header.
