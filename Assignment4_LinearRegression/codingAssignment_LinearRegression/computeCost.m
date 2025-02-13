function J = computeCost(X, y, beta)
%COMPUTECOST computes the cost of using beta as the
%   parameter for linear regression to fit the data points in X and y

%% Initialize some useful values
%m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

%% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of beta
%               You should set J to the cost.
lambda = 0.1;
JB = sum(beta.^2);
JD = sum((  y - X * beta).^2 );
J = JD + lambda * JB;

% =========================================================================

end
