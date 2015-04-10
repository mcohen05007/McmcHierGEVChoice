function out = GEVhybrid(Data, Prior, Mcmc) 
%% MCMC for Hierarchical GEV Choice Model
% Michael A. Cohen, PhD
% W: www.macohen.net
% E: michael@macohen.net
% Proper citation is appreciated for use or adaptation, please cite as:
% Cohen, M. A. (2015). MCMC for Hierarchical GEV Choice Model [Computer software]. 
% Retrieved from 

%% Unpack Estimation Arguments (Get the sampler ready)

%Data
    p = Data{end}.p;
    ypooled = Data{end}.ypooled;
    Xpooled = Data{end}.Xpooled;
    Z = Data{end}.Z;
    k = size(Xpooled,2);
    NC = length(Data)-1;
    nd = size(Z,2);
    
%Prior
    deltabar = Prior.deltabar; 
    Ad = Prior.Ad;
    nu = Prior.nu;
    V = Prior.V;

%MCMC params
    R = Mcmc.R;
    keep = Mcmc.keep;          
    
%% Define scale parameters for RW Metropolis-Hastings Draws

%If initial sequence of draws use Fractional likelihood to estimate
%Unit-level aysmptotic covaraince
x_0 = zeros(k,1);
options = optimset('Display','off','LargeScale','off','MaxIter',10000);
[mle,~,~,~,~,H] = fminunc(@(x) -llmnl(x,ypooled,Xpooled),x_0,options);
betapooled = mle;
rootH = chol(H);

%Allocate Space for Random Walk scaling parameters
candcovi = cell(NC,1);
candcov = cell(NC,1);
betaold = zeros(NC,k);
for i = 1:NC
   wgt = length(Data{i}.y)/length(ypooled);
   [mle,~,~,~,~,H] = fminunc(@(x) -llmnlFRACT(x, Data{i}.y, Data{i}.X, betapooled, rootH, 0.1, wgt),betapooled,options);
   COV = eye(k+1)*10;
   COV(1:k,1:k) = H;
   candcovi{i} = COV;
   betaold(i,:) = mle;
   if mod(i,50)==0
        disp(['Completed Unit #','  ', num2str(i)])
   end
end
% Set initial scaling parameter
s = 1/sqrt(k+1);

%% Allocate Space
%Allocate space for draws (Change to sv is larger storage is required)
betadraw = zeros(R/keep,NC,k);
xidraw = zeros(R/keep,NC);
Sigmadraw = zeros(R/keep,(k+1)^2);
deltadraw = zeros(R/keep,(k+1)*nd);
loglikedraw = zeros(R/keep,NC);
%Allocate space for other variables
wnew = cell(1,NC);
wold = cell(1,NC);
for i = 1:NC
   wnew{i} = zeros(size(Data{i}.X,1),1); 
   wold{i} = zeros(size(Data{i}.X,1),1); 
end
xiold = -.1*ones(NC,1);
deltaold = zeros(nd,k+1);
rootSigmaInv = eye(k+1);

%% Initiate Draws
logl = repmat(-Inf,NC,1);
naccept = zeros(NC,1);
naccepto = zeros(NC,1);
tic
disp('MCMC Iteration (Estimated time to end)')
for rep=1:R
    %Prior mean of unit level parameters
    priormean = (Z*deltaold)';    
    %Draw from unit-level model
    parfor i = 1:NC
        root = eye(k+1)/chol(candcovi{i} + rootSigmaInv*rootSigmaInv');
        if rep>5000
            root = candcov{i};
        end
        %Metropolis Hastings for draw of beta
        [theta, ll, w,naccept(i)] = MHthetaRW(wold{i},Data{i}.y,Data{i}.X,[betaold(i,:) xiold(i)]',naccept(i),priormean(:,i),rootSigmaInv,root,s,p,k);
        wold{i} = w;
        betaold(i,:) = theta(1:k)';
        xiold(i) = theta(end);
        logl(i) = ll;       
    end
    % Draw from first stage prior
    [deltaold, Sigma, rootSigmaInv] = bmultireg([betaold xiold], Z, deltabar, Ad, nu, V);  
    if (mod(rep,keep) == 0) 
        mkeep = rep/keep;
        betadraw(mkeep,:,:) = betaold;
        xidraw(mkeep,:) = xiold;
        loglikedraw(mkeep,:) = logl;
        Sigmadraw(mkeep,:) = Sigma(:);
        deltadraw(mkeep,:) = deltaold(:)';        
    end
    if (mod(rep,100) == 0)
        mkeep = rep/keep;
        b = round(.75*mkeep);
        disp('Mean of last 75% of Xi Draws')
        disp(num2str(mean(mean(xidraw(b:mkeep,:)))))
        disp('Mean of last 75% of Beta Draws')
        disp(num2str(reshape(mean(mean(betadraw(b:mkeep,:,:))),1,k)))
        timetoend = (toc/rep) * (R + 1 - rep);
        hours = floor(timetoend/60/60);
        mins = floor((timetoend/60)-hours*60);
        secs = floor(((timetoend/60)-floor(timetoend/60))*60);
        disp('MCMC Iteration (Estimated time to end)')
        disp([ '    ', num2str(rep), '          ',num2str(hours),' ', 'Hours',' ',num2str(mins),' ', 'Minutes',' ',num2str(secs),' ','Seconds'])
        disp(['% of accepted draws', '  ', num2str(round(100*sum(naccept)/rep/NC)),'%'])
    end     
    if (mod(rep,5000) == 0) 
        % Update posterior covariance utility parameters
        bi = round(.5*(rep/keep)); % Use the most recent X% of draws to avoid initial conditions
        for i =1:NC
           thetadraw = [reshape(betadraw(bi+1:rep/keep,i,:),rep/keep-bi,k) xidraw(bi+1:rep/keep,i)];
           covt = cov(thetadraw);
           [C,pd]=chol(covt);
           if pd==0
                candcov{i} = C;
           else
                candcov{i} = eye(k+1)/chol(candcovi{i} + rootSigmaInv*rootSigmaInv');
           end
        end
    end
    if (mod(rep,100) == 0)       
        % Adjust random walk step size to maintain efficient acceptance rate
        acceptr = (sum(naccept)-sum(naccepto))/100/NC;
        naccepto = naccept;
        if acceptr<.23
            s = .9*s;
        elseif acceptr>.23
            s = 1.25*s;
        end            
        disp(num2str(s))
    end
end
disp(['Total Time Elapsed: ', num2str(round(toc/60)),' ','Minutes']) 
out = struct('betadraw',betadraw,'xidraw',xidraw,'loglikedraw',...
    loglikedraw,'naccept',naccept);
end  

function [thetao, oll, wo, naccept] = MHthetaRW(wo,y,X,thetao,naccept,priormean,rootpi,root,s,p,k)
%Draw w conditional on the old theta
w = drawev(wo,X*thetao(1:k),thetao(k+1),p,y);
%Compute log likelihood for old draw
oll = EVloglike(w,X*thetao(1:k),thetao(k+1));
%Random Walk Metropolis draw of candidate theta
thetac = thetao+s*root*randn(k+1,1);
cll = EVloglike(w,X*thetac(1:k),thetac(k+1)); 
clpost = cll + lndMvn(thetac, priormean, rootpi); 
ldiff = clpost - oll - lndMvn(thetao, priormean, rootpi);
if rand(1)<exp(ldiff)
    thetao = thetac;
    naccept = naccept + 1;
    oll = cll;
    wo = w;
end
end

function wnew = drawev(w,mu,xi,p,y)
% Draw the latent values from GEV distribution
n = length(y);
w = reshape( w, p, n);
mu = reshape( mu, p, n);
indx = 1:p;
for i=indx
    temp = w(indx~=i,:);%[zeros(1, n);w(indx~=i,:)];
    bounds = max(temp,[],1)';
    ubounds = bounds;
    lbounds = bounds;
    ubounds(y==i) = Inf;
    lbounds(y~=i) = -Inf;
    w(i,:) = rtrunev(mu(i,:),xi,lbounds',ubounds');
end
wnew = reshape(w,p*n,1);
end

function z = rtrunev(mu,xi,a,b)
%Inverse CDF method to draw from truncated generalized extreme value
if xi==0
    m = psi(1);
else
    l = -.5;
    m = (1-gamma(1-l))/l;
end
FA = gevcdf(a-mu,xi,1,m);
FB = gevcdf(b-mu,xi,1,m);
z = mu+gevinv(rand(1,numel(mu)) .* (FB-FA) + FA,xi,1,m);
end

function ll = llmnl(beta, y, X)
n = length(y);
j = size(X,1)/n;
Xbeta = X * beta;
Xbeta = reshape(Xbeta, j, n)';
ind = sub2ind([n j],1:n, y');
xby = Xbeta(ind)';
Xbeta = exp(Xbeta);
iota = ones(j,1);
denom = log(Xbeta * iota);
ll = sum(xby - denom);
end

function ll = llmnlFRACT(beta, y, X, betapooled, rootH, w, wgt) 
z = rootH * (beta - betapooled);
ll = (1 - w) * llmnl(beta, y, X) + w * wgt * (-0.5 *(z(:)' * z(:)));
end

function p = lndMvn(x, mu, rooti) 
z = rooti' * (x - mu);
p = -(length(x)/2) * log(2 * pi) - 0.5 * (z(:)' * z(:)) + sum(log(diag(rooti)));
end

function ll = EVloglike(w,Xb,xi)
% Generalized extreme value function syntax gevcdf(resid,shape,scale,loc)
if xi==0
    m = psi(1);
else
    l = -.5;
    m = (1-gamma(1-l))/l;
end
pdf = gevpdf(w-Xb,xi,1,m);
if sum(pdf==0)>0
    ll = -Inf;
else
    ll = sum(log(pdf));
end
end

function beta = breg(y, X, betabar, A) 
% single draw from posterior of multivariate Bayesian regression model
k = length(betabar);
RA = chol(A);
W = [X;RA];
z = [y;RA*betabar];
IR = eye(k)/chol(W'*W);
beta = (IR*IR')*W'*z + IR*randn(k,1);
end

function [beta, Sigma, rootSigInv] = bmultireg(y, X, betabar, A, nu, V) 
% single draw from posterior of multivariate Bayesian regression model
n = size(y,1);
m = size(y,2);
k = size(X,2);
RA = chol(A);
W = [X;RA];
Z = [y;RA*betabar];
IR = eye(k)/chol(W'*W);
Btilde = (IR*IR')*W'*Z;
res = Z-W*Btilde;
S = res'*res;
rwout = rwishart(nu+n,eye(m)/(V+S));
Sigma = rwout.IW;
rootSigInv = rwout.CI;
beta = Btilde + IR*randn(k,m)*rootSigInv';
end

function rout=rwishart(nu, V) 
%Single random draw from wishart distribution
m = size(V,1);
df = (nu + nu - m + 1) - (nu - m + 1):nu;
if (m > 1)
    T = diag(sqrt(chi2rnd(df,m,1)));
    T = tril(ones(m,m),-1).* randn(m) + T;
else 
    T = sqrt(chi2rnd(df));
end
U = chol(V);
C = T' * U;
CI = eye(m)/C;
W = C'*C;
IW = CI*CI';
rout = struct('C',C,'CI',CI,'W', W,'IW',IW);
end