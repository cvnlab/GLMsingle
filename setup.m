% This script adds GLMsingle to the MATLAB path.

% Add GLMsingle to the MATLAB path (in case the user has not already done so).
GLMsingle_dir = fileparts(mfilename('fullfile'));

addpath(fullfile(GLMsingle_dir, 'matlab'));
addpath(fullfile(GLMsingle_dir, 'matlab', 'utilities'));

% if the submodules were installed we try to add their code to the path
addpath(fullfile(GLMsingle_dir, 'matlab', 'fracridge', 'matlab'));

% check that the dependencies are in the path
tmp = which('fracridge.m');
if isempty(tmp)
  error('fracridge is missing. Please install from: https://github.com/nrdg/fracridge.git')
end

clear GLMsingle_dir;
