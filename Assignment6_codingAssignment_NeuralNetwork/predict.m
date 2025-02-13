function p = predict(beta1, beta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(beta1, beta2, X) outputs the probability of the output to 
%   1, given input X and trained weights of a neural network (beta1, beta2)

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1); % 
[m, ~] = size(p);
X = [ones(m,1) X]; % append bias term
% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. 
%
%

for i = 1:m
    h = sum(beta1 .* X(i,:), 2);
    h = sigmoid(h);
    h = [1; h];
    out = sigmoid(beta2 * h);
    if out >= .5
        p(i) = 1;
    else
        p(i) = 0;
end


% =========================================================================


end
