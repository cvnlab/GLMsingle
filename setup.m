% This script adds GLMsingle to the MATLAB path.

% Add GLMsingle to the MATLAB path (in case the user has not already done so).
path0 = strrep(which('setup.m'),'/setup.m','/matlab');
addpath(genpath(path0));
clear path0;
