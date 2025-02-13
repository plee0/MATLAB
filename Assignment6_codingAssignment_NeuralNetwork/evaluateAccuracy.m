function accuracy = evaluateAccuracy(beta1, beta2, X, y)
%EVALUATEACCURACY calculates the prediction accuracy of the learned 
%neural network model using the testing data 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the percentage of accurately predicted examples 
%
%

num = length(y);
p = predict(beta1, beta2, X);

accuracy = sum( ~xor(p, y) ) / num;

% ============================================================

end