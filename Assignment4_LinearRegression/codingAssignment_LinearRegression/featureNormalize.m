function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is a sample data. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
% NOTE: By textbook definition, we will be using the standardization method
% and not min-max normalization.

% Determine the mean and standard deviation of each feature. The result is
% a 1x3 matrix.
mu = mean(X_norm,1);
sigma = std(X_norm,1);

for j = 1:3
    for i = 1:length(X)
        X_norm(i,j) = ( X_norm(i,j) - mu(j) ) / sigma(j);
    end
end


%% Verify that the mean is 0 and standard deviation is 1
    % mean(X_norm)
    % std(X_norm)
    % Mean of X_norm is a really small number (10^-15 essentially 0)
    % standard deviation of X_norm can be rounded to 1.


% ============================================================

end
