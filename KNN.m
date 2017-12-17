function [ class ] = KNN(X,K,data)
% class : class guess made for each data point
% X : Training Set
% K : Neighbour number
% data: Test Set
[n,~] = size(data);
class = zeros(n,1);
    % Eucledian distance utilized.
    for i =1:n
      dist = sqrt(sum(bsxfun(@minus,X(:,1:2),data(i,1:2)).^2,2));
      sortedDist = sortrows([dist X(:,3)]);
      class(i) = mode(sortedDist(1:K,2));
    end
end

