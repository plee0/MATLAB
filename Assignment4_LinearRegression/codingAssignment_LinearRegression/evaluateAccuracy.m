function error = evaluateAccuracy(beta, X, y)
%EVALUATEACCURACY calculates the average prediction error of the learned 
%linear regression model using the testing data 

m = length(y); % number of testing examples

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the prediction error for each testing example and
%               then take the mean
% Our equation is now y_hat = beta * X.
% The prediction error for each testing sample, e, will be y - y_hat
% Then the mean will be computed.

y_hat = X * beta;

e = y - y_hat;
error = mean(e);
% ============================================================

end