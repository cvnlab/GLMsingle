function responses = GLMpredictresponses(model,design,tr,numtimepoints,dimdata)

% function responses = GLMpredictresponses(model,design,tr,numtimepoints,dimdata)
%
% <model> is one of the following:
%   A where A is X x Y x Z x conditions x time with the timecourse of the 
%     response of each voxel to each condition.  XYZ can be collapsed.
%   {B C} where B is time x 1 with the HRF that is common to all voxels and
%     all conditions and C is X x Y x Z x conditions with the amplitude of the 
%     response of each voxel to each condition
%   Note that in both of these cases, the first time point is assumed to be 
%   coincident with condition onset.
% <design> is the experimental design.  There are three possible cases:
%   1. A where A is a matrix with dimensions time x conditions.
%      Each column should be zeros except for ones indicating condition onsets.
%      (Fractional values in the design matrix are also allowed.)
%   2. {A1 A2 A3 ...} where each of the A's are like the previous case.
%      The different A's correspond to different runs, and different runs
%      can have different numbers of time points.
%   3. {{C1_1 C2_1 C3_1 ...} {C1_2 C2_2 C3_2 ...} ...} where Ca_b
%      is a vector of onset times for condition a in run b.  Time starts at 0 
%      and is coincident with the acquisition of the first volume.  This case 
%      is compatible only with the common-HRF <model>.
% <tr> is the sampling rate in seconds
% <numtimepoints> is a vector with the number of time points in each run
% <dimdata> indicates the dimensionality of the voxels.
%   A value of 3 indicates X x Y x Z, and a value of 1 indicates XYZ.
%
% Given various inputs, compute the predicted time-series response.
%
% Return:
% <responses> as X x Y x Z x time or a cell vector of elements that are 
%   each X x Y x Z x time.  The format of <responses> will be a matrix in the
%   case that <design> is a matrix (case 1) and will be a cell vector in
%   the other cases (cases 2 and 3).
%
% History:
% - 2013/05/12: allow <design> to specify onset times; add <tr>,<numtimepoints> as inputs
% - 2013/05/12: update to indicate fractional values in design matrix are allowed.
% - 2012/12/03: *** Tag: Version 1.02 ***
% - 2012/11/2 - Initial version.

% calc
ismatrixcase = ~iscell(design);
dimtime = dimdata + 2;
if iscell(model)
  xyzsize = sizefull(model{2},dimdata);
else
  xyzsize = sizefull(model,dimdata);
end

% make cell
if ~iscell(design)
  design = {design};
end

% loop over runs
responses = {};
for p=1:length(design)

  % if onset-time case
  if iscell(design{p})

    % check that we have the case of common-HRF model
    assert(iscell(model));

    % calc
    alltimes = linspacefixeddiff(0,tr,numtimepoints(p));
    hrftimes = linspacefixeddiff(0,tr,length(model{1}));
  
    % loop over conditions
    temp = zeros(numtimepoints(p),length(design{p}));  % this will be time x conditions
    for q=1:length(design{p})

      % onset times for qth condition in run p
      otimes = design{p}{q};
    
      % intialize
      yvals = 0;
    
      % loop over onset times
      for r=1:length(otimes)
        
        % interpolate to find values at the data sampling time points
        yvals = yvals + interp1(otimes(r) + hrftimes,model{1}',alltimes,'pchip',0);

      end

      % record
      temp(:,q) = yvals;

    end
    
    % weight by the amplitudes
    responses{p} = reshape((temp * squish(model{2},dimdata)')',[xyzsize numtimepoints(p)]);  % X x Y x Z x time
  
  % if regular matrix case
  else

    % case of shared HRF
    if iscell(model)
    
      % convolve with HRF
      temp = conv2(full(design{p}),model{1});  % make full just in case design is sparse

      % extract desired subset of time-series
      temp = temp(1:numtimepoints(p),:);  % time x conditions

      % weight by the amplitudes
      responses{p} = reshape((temp * squish(model{2},dimdata)')',[xyzsize numtimepoints(p)]);  % X x Y x Z x time
  
    % case of individual timecourses
    else

      % length of each timecourse (L)
      len = size(model,dimtime);
    
      % expand design matrix using delta functions
      temp = constructstimulusmatrices(design{p}',0,len-1,0);  % time x L*conditions

      % weight design matrix by the timecourses
      responses{p} = reshape((temp * squish(permute(squish(model,dimdata),[3 2 1]),2))',[xyzsize numtimepoints(p)]);  % X x Y x Z x time

    end

  end

end

% undo cell if necessary
if ismatrixcase
  responses = responses{1};
end
