function [mu,sigma,ind,pp]=GMM(X,K)
% Expectation Maximization 
% X : Input data, n x d                                                     
% N : Sample number
% K : Cluster number
% D : Feature number
% mu: Cluster means
% sigma: Covariance matrix of clusters
% ind: Cluster labels to plot
% pp: Posterior probs.

% Initialization with Kmeans
[ind,mu] = kmeans(X,K);
% Size of Input matrix
[N,~]=size(X);
% Posterior prob. Init.
pp=rand(N,K);
% Normalizing prob.
pp=pp./repmat(sum(pp,2),[1 K]);

% Prior prob. and cov cell array init.
p=zeros(K,1);
sigma=cell(1,K);
% Iteration number to obtain a small convergence 
for iter=1:10000
    fprintf('Exp. and Max. Iteration %d\n', iter);
    mprev=mu;
    %% Maximization Part
    S=sum(pp);
    for i=1:K
        mu(i,:)=sum(bsxfun(@times,pp(:,i),X))/S(i);
        Xm=bsxfun(@minus,X,mu(i,:));
        sigma{i} = bsxfun(@times,pp(:,i),X)'*Xm/S(i);
        p(i)=S(i)/N;
    end
    
    %% Convergence Check
    if sum((mu-mprev).^2)<1e-10
        indc=zeros(N,K);
        for i=1:K
            indc(:,i)=sum(abs(bsxfun(@minus,X,mu(i,:))),2);
        end
        for i=1:N
            [~,ind(i)]=min(indc(i,:));
        end
        break;
    end

    %% Expectation Part
    for j=1:K
        pp(:,j)=p(j)*gaussianProb(X,mu(j,:),sigma{j});
    end
    pp=pp./repmat(sum(pp,2),[1 K]);
end

%% If convergence cond. is not met
indc=zeros(N,K);
for i=1:K
    indc(:,i)=sum(abs(bsxfun(@minus,X,mu(i,:))),2);
end
for i=1:N
    [~,ind(i)]=min(indc(i,:));
end
