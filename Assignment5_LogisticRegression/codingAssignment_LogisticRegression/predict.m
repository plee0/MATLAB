function p = predict(beta, X)
%PREDICT Predict whether the label is 0 or 1 using learned logistic 
%regression parameters beta
%   p = PREDICT(beta, X) computes the predictions for X using a 
%   threshold at 0.5 (i.e., if probability >= 0.5, predict 1)

m = size(X, 1); % Number of training examples

% You need to return the following variables correctly
p = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters. 
%               You should set p to a vector of 0's and 1's
%

% Calculate the probability
U = sigmoid(X*beta);
% Evaluate vs. 0.5
for i = 1:m
    if U(i) >= 0.5
        p(i) = 1;
    else
        p(i) = 0;
    end
end

% =========================================================================


end
