function accuracy = evaluateAccuracy(beta, X, y)
%EVALUATEACCURACY calculates the prediction accuracy of the learned 
%logistic regression model using the testing data 

num = length(y); % number of testing examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the percentage of accurately predicted examples 
%
% To predict accuracy, first we must compute results using our estimated
% parameters.
p = predict(beta, X);

% Now, we may begin to compare p with y
%{
% Method using for loop // DEBUG PURPOSES ONLY
temp = 0;
for j = 1:num
    if y(j) == p(j)
        temp = temp+1; %counter
    end
end
accuracy = temp / num;
%}
%% XOR method
% By using xor, any incorrect answers will be set to logical 1. ~(NOT)
% operator is used to flip the logic to set correct predictions to 1 and
% incorrect predictions to 0
accuracy = sum( ~xor(p, y) ) / num;
% ============================================================
end