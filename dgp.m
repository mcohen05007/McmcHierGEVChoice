function Data = dgp()
%Generate Data
p = 3;                      % # of choice alterns
T = 50;                     % # of market periods
NC = 300;                   % # of consumers
nz = 2;                     % # of demographic variables
nt = 4;                     % # of utility parameters

Z = rand(NC,nz);                     % Simulate Cross-Sectional Explainitory Variables
Z = Z-repmat(mean(Z,1),NC,1);        % Demean Them
Z = [ones(NC,1) Z];

Delta= [1,1,0;2,1,0;-2,1,1;-.1,0,.1];% Cross-sectional model parameters
sigb = repmat(0.1,nt,1);
 
% Simulate data
Data=cell(NC+1,1);
y = [];X=[];
for i = 1:NC 
    thetai = Delta*Z(i,:)'+ diag(sigb)*randn(nt,1);
    simdata = simgev(p,T,thetai(1:3),thetai(end));
    y = [y;simdata.y];
    X = [X;simdata.X];
    Data{i} = struct('y',simdata.y,'X',simdata.X,'Beta',simdata.beta,'Xi',simdata.Xi);
end
Data{end} = struct('p', p, 'ypooled', y, 'Xpooled', X, 'Z', Z);
end

function out = simgev(p,n,beta,k)
% function generates gev choices
X = randn(n*p,1);
ints = repmat([zeros(1,p-1);eye(p-1)],n,1);
X = [ints X];
Xbeta = X*beta;
if k==0
    m = psi(1);
else
    m = (1-gamma(1-k))/k;
end
wout = Xbeta+gevrnd(k,1,m,n*p,1);
w = reshape(wout,p,n);
wmax = max(w,[],1);
y=zeros(n,1);
for i=1:p
    ind = w(i,:)==wmax;
    y(ind)=i;
end
out = struct('w',wout,'y',y,'X',X,'beta',beta,'Xi',k);
end