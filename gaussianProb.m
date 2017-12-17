function [ pdf ] = gaussianProb(X, mu, Sigma)

%      X - Matrix of data points.
%     mu - Mean vector
%  Sigma - Covariance matrix.

% Get the vector length.
n = size(X, 2);

% Subtract the mean from every data point.
meanDiff = bsxfun(@minus, X, mu);

% Calculate the multivariate gaussian.
pdf = 1 / sqrt((2*pi)^n * det(Sigma)) * exp(-1/2 * sum((meanDiff /(Sigma) .* meanDiff), 2));

end

