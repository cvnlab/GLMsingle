clc
clear
close all

% This example shows how to interpolate the fMRI timecourse if the stimulus
% duration is not equal to the duration of the TR. Imagine that there are 
% 100 TRs and each lasts for 1s, while your stimulus duration is only 0.5s
% The stimulus starts with the TR onset. To correctly code the stimulus 
% onset in your design matrix you need 200 rows in your design matrix 
% (but you only have 100 fMRI volumes). Interpolation of the timecourse is 
% necessary to match the size of rows in your design matrix. 
%%

% Simulated data properties.
TRs = 100;
TR = 1;
stimdur = 0.5;
TRs_after_resampling = TR/stimdur*TRs;

% Use an example hrf and create an example fMRI time series.
cond1 = zeros(TRs,1);
cond1(1:20:end) = 1;

hrf = getcanonicalhrf(0.5,1);
tcs = conv(cond1,hrf);
tcs = tcs(1:TRs);
figure(1);clf
plot(0:TR:TRs-TR,tcs,'-','LineWidth',2); hold on
ylabel('%BOLD')
xlabel('TRs')

% The following line resmaples the timecourse so that each timepoint will
% correspond to 0.5 s instead of 1 s. The tseriesinterp is not a function
% available in GLMsingle but you can download it form github 
% https://github.com/cvnlab/knkutils/blob/master/timeseries/tseriesinterp.m

tcs_interp = tseriesinterp(tcs,1,0.5);

% plot the interpolated timeseries
plot(0:stimdur:TRs-stimdur,tcs_interp,'o','MarkerSize',3,'LineWidth',2)
legend box off
set(gca,'FontSize',15)

%%
whos tcs tcs_interp

% Notice that the lenght of the tcs_interp is double the length of tcs.

% With an interpolated timecourse now you can code your design matrix
% correctly. The design matrix is going to consist of 200 columns were 1 will
% specify the stimulus onset. Remember to specify the stimduration for
% GLMsingle as 0.5 s instead of 1 s.

%% show the dm as stem plot with correct length

dm = repelem(cond1,2);
find_rep = diff(dm);
dm(find_rep==-1) = 0;
dm(dm==0) = NaN;
stem(0:stimdur:TRs-stimdur,dm,'filled','LineWidth',2)
legend({'Original tcs';'Interpolated tcs';'Stimulus onset'},'Location','EastOutside')
set(gcf,'Position',[1000        1090        1234         247])
