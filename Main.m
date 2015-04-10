%% Example Script to sample from Generalized extreme value choice model
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated for use or adaptation, please cite as:
% Cohen, M. A. (2015). MCMC for Hierarchical GEV Choice Model [Computer software]. 
% Retrieved from http://www.macohen.net/software or https://github.com/mcohen05007/McmcHierGEVChoice
clear
clc

%% 
% Seed Random number geenrator and use the new SIMD-oriented Fast Mersenne
% Twister only for use with MATLAB 2015a or newer
% rng(0,'simdTwister')
rng(100,'twister')

%% Generate Data (Change dgp.m if you want to experiment with the data generation process)
Data = dgp();

%%
Mcmc = struct('R',1e4,'keep',5);
k = size(Data{end}.Xpooled,2)+1;
nz = size(Data{end}.Z,2);
nu = k+3;
Prior = struct('deltabar',zeros(nz,k),'Ad',.0001*eye(nz),'nu',nu,'V',nu*eye(k));

%%
out=GEVhybrid(Data, Prior, Mcmc);

%% Plot Estimated vs Actual Parameters
% First the Betas
betadraw = out.betadraw;
figure
ss=.2;
ncoef = k-1;
for jj=1:ncoef
    bd = betadraw(:,:,jj);
    x = min(bd(:)):ss:max(bd(:));
    subplot(2,ncoef,jj), hist(bd(:),x)
    for i=1:size(out.betadraw,2)
        bt(i) = Data{i}.Beta(jj);
    end
    subplot(2,ncoef,ncoef+jj), hist(bt,x)
end
% Second Skewness
figure
xi = out.xidraw;
ss=.1;
x = min(xi(:)):ss:max(xi(:));
for i=1:size(out.betadraw,2)
    xt(i) = Data{i}.Xi;
end
subplot(2,1,1), hist(xi(:),x)
subplot(2,1,2), hist(xt(:),x)