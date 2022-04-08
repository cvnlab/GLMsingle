% make sure we have no NaN / Inf in our expected results
% aslo plots a couple of figures for quick visual inspection

clear
clc
close all

load('TYPED_FITHRF_GLMDENOISE_RR.mat')

assert(any(isnan(R2(:))) == 0);
assert(any(isinf(R2(:))) == 0);

figure('name', 'histogram R2')
hist(R2(:), 100);
print('histogram_R2.tif', '-depsc','-tiff');

figure('name', 'R2')
imagesc(R2, [0 100])

figure('name', 'HRFindex')
imagesc(HRFindex, [1 20])