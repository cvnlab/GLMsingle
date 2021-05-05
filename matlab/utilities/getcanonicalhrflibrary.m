function hrfs = getcanonicalhrflibrary(duration,tr)

% function hrfs = getcanonicalhrflibrary(duration,tr)
%
% <duration> is the duration of the stimulus in seconds.
%   should be a multiple of 0.1 (if not, we round to the nearest 0.1).
%   0 is automatically treated as 0.1.
% <tr> is the TR in seconds.
%
% generate a library of 20 predicted HRFs to a stimulus of
% duration <duration>, with data sampled at a TR of <tr>.
%
% the resulting HRFs are returned as 20 x time. the first point is 
% coincident with stimulus onset. each HRF is normalized such 
% that the maximum value is one.
%
% example:
% hrfs = getcanonicalhrflibrary(4,1);
% figure; plot(0:size(hrfs,2)-1,hrfs,'o-');

% inputs
if duration == 0
  duration = 0.1;
end

% load the library
file0 = strrep(which('getcanonicalhrflibrary'),'getcanonicalhrflibrary.m','getcanonicalhrflibrary.tsv');
hrfs = load(file0)';  % 20 HRFs x 501 time points

% convolve to get the predicted response to the desired stimulus duration
trold = 0.1;
hrfs = conv2(hrfs,ones(1,max(1,round(duration/trold))));

% resample to desired TR
hrfs = interp1((0:size(hrfs,2)-1)*trold,hrfs',0:tr:(size(hrfs,2)-1)*trold,'pchip')';  % 20 HRFs x time

% make the peak equal to one
hrfs = hrfs ./ repmat(max(hrfs,[],2),[1 size(hrfs,2)]);

%%%%%%%%%%%%%%%%%%% FOR OUR RECORDS BELOW:

% % params taken from the Natural Scenes Dataset
% a1 = load('~/nsd/nsddata/templates/hrfparams.mat');
% 
% % obtain canonical response to a 0.1-s stimulus
% hrfs = [];
% for p=1:size(a1.params,1)
%   hrfs(p,:) = spm_hrf(0.1,a1.params(p,:));
% end
% 
% dlmwrite('getcanonicalhrflibrary.tsv',hrfs','delimiter','\t','precision',5);
% test=load('getcanonicalhrflibrary.tsv');
% figure; plot(hrfs');
% figure; plot(test);
