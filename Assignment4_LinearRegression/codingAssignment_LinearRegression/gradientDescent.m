function [beta, J_history] = gradientDescent(X, y, beta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn beta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = length(beta); % number of features
Total = 0;
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               beta. 
    % Gradient Descent Formula: Beta - alpha * -2 SUMMATION(y_i - x_i' *
    % Beta) x_ik
    
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    beta = beta' + alpha * 2 * ( (y - X * beta)' * X);
    beta = beta';
    % ============================================================

    % Save the cost J in every iteration 
    % ====================== YOUR CODE HERE ======================
    J_history(iter) = computeCost(X, y, beta);

    % ============================================================
end

end
