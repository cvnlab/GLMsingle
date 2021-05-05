function [f,mns,sds,gmfit] = findtailthreshold(v,wantfig)

% function [f,mns,sds,gmfit] = findtailthreshold(v,wantfig)
% 
% <v> is a vector of values
% <wantfig> (optional) is whether to plot a diagnostic figure. Default: 1.
%
% Fit a Gaussian Mixture Model (with n=2)
% to the data and find the point that is greater than
% the median and at which the posterior probability
% is equal (50/50) across the two Gaussians.
% This serves as a nice "tail threshold".
%
% To save on computational load, we take a random subset of
% size 1000000 if there are more than that number of values.
% We also use some discretization in computing our solution.
%
% return:
% <f> as the threshold
% <mns> as [A B] with the two means (A < B)
% <sds> as [C D] with the corresponding std devs
% <gmfit> with the gmdist object (the order might not
%   be the same as A < B)
%
% example:
% [f,mns,sds,gmfit] = findtailthreshold([randn(1,1000) 5+3*randn(1,500)]);

% internal constants
numreps = 3;      % number of restarts for the GMM
maxsz = 1000000;  % maximum number of values to consider
nprecision = 500; % linearly spaced values between median and upper robust range

% inputs
if ~exist('wantfig','var') || isempty(wantfig)
  wantfig = 1;
end

% quick massaging of input
v = v(isfinite(v));
if length(v) > maxsz
  warning('too big, so taking a subset');
  v = picksubset(v,maxsz);
end

% fit mixture of two gaussians
gmfit = fitgmdist(v(:),2,'Replicates',numreps);

% figure out a nice range
rng = robustrange(v(:));

% evaluate posterior
allvals = linspace(median(v),rng(2),nprecision);
checkit = zeros(length(allvals),2);
for qq=1:length(allvals)
  checkit(qq,:) = posterior(gmfit,allvals(qq));
end

% figure out crossing
assert(any(checkit(:,1) > .5) && any(checkit(:,1) < .5),'no crossing of 0.5 detected');
[mn,ix] = min(abs(checkit(:,1)-.5));

% return it
f = allvals(ix);

% prepare other outputs
mns = flatten(gmfit.mu);
sds = sqrt(flatten(gmfit.Sigma));
if mns(2) < mns(1)
  mns = mns([2 1]);
  sds = sds([2 1]);
end

% start the figure
if wantfig

  % calc
  x = linspace(rng(1),rng(2),100);   %.01 means when you sum you get a lot  .001 means you get 10x
  vals0 = pdf(gmfit,x');

  % make figure
  figure;
  subplot(2,1,1); hold on;
  binsize = x(2)-x(1);
  [n,x] = hist(v(:),x);
  plot(x,n/sum(n),'k-','LineWidth',3);
  plot(x,vals0/sum(vals0),'r-','LineWidth',2);
  ax = axis;
  valstemp = [];
  for zz=1:2
    valstemp(zz,:) = gmfit.PComponents(zz)*(normpdf(x,gmfit.mu(zz),sqrt(gmfit.Sigma(:,:,zz))) * binsize);
    plot(x,valstemp(zz,:),'c-');
  end
  plot(x,sum(valstemp,1),'b-','LineWidth',2);
  set(straightline(allvals(ix),'v','k-'),'LineWidth',2);
  title('Histogram (black); Full GMM fit (red); Two Dists (cyan) and their sum (blue)');

  % another visualization
  subplot(2,1,2); hold on;
  plot(allvals,checkit);
  ax2 = axis;
  axis([ax(1:2) ax2(3:4)]);
  set(straightline(allvals(ix),'v','k-'),'LineWidth',2);
  title('Posterior Probabilities');

end
