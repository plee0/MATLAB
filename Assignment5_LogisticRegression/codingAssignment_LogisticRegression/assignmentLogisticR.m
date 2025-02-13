%% SDSU Machine Learning Course (EE600/CompE596)
%% Programming Assignment:  Logistic regression 
%  Paulie Lee
%  10/20/2021
%  Dataset comes from: 
%   http://networkrepository.com/pima-indians-diabetes.php
% 
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  logistic regression assignment. 
%
%  You will need to complete the following functions in this 
%  assignment
%
%     loadData.m
%     featureNormalize.m
%     gradientAscent.m
%     likelihoodFunction.m
%     evaluateAccuracy.m 
%     predict.m
%     sigmoid.m
%
%  For this part of the assignment, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%
% Initialization
clear ; close all; clc

%% ================ Part 1: Data Preprocessing ================
% Instructions: The following code loads data into matlab, splits the 
%               data into two sets, and performs feature normalization. 
%               You will need to complete code in loadData.m, and 
%               featureNormalize.m
%
%

%% Load data
fprintf('Loading data ...\n');

% ====================== YOUR CODE HERE ======================
[X_train, y_train, X_test, y_test] = loadData();

% ============================================================

[n, m] = size(X_train); % n is the number of total data examples
                        % m is the number of features
                  
% Print out some data points
fprintf('First 10 examples from the training dataset: \n');
fprintf(' x = [%.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f], y = %.0f \n',...
    [X_train(1:10,:) y_train(1:10,:)]');
fprintf('\n');

% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

%% Normalize the features. 
% ====================== YOUR CODE HERE ======================
[Xn_train, mu, sigma] = featureNormalize(X_train);

% ============================================================

num_train = length(y_train); % number of training examples

% Add intercept term to X_train and X_test
Xn_train = [ones(num_train, 1) Xn_train];

%% ========== Part 2: Maximum Likelihood & Gradient Ascent =============

fprintf('Running gradient ascent ...\n');

% Instructions: The following code applies gradient ascent to 
%               estimate the parameters in a logistic regression 
%               model based on the idea of maximum likelihood estimation. 
%               You should complete code in gradientAscent.m,
%               likelihoodFunction.m
%
%               Try running gradient ascent with 
%               different values of alpha and see which one gives
%               you the best result.
%

% ====================== YOUR CODE HERE ======================
% Choose some alpha value and number of iterations
alpha = .01;
num_iters = ceil( .05 * length(X_train) );

% ============================================================

% Init beta
beta = zeros(m+1, 1);

%% Run Gradient Descent 
% ====================== YOUR CODE HERE ======================
[beta, l_history] = gradientAscent(Xn_train, y_train, beta, alpha, num_iters);

% ============================================================

% Plot the convergence graph
figure;
plot(1:numel(l_history), l_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('log likelihood l');

% Display gradient descent's result
fprintf('beta computed from gradient descent: \n');
fprintf(' %f \n', beta);
fprintf('\n');


%% ========== Part 3: Evaluate performance =============

fprintf('Evaluate the prediction accuracy ...\n');

% Instructions: The following code evaluates the performance of
%               the trained logistic regression model. You should 
%               complete code in evaluateAccuracy.m, predict.m, and sigmoid.m
%

num_test = length(y_test); % number of testing examples

% normalize input features of the testing set
Xn_test = (X_test - mu)./sigma;

% Add intercept term to Xn_test
Xn_test = [ones(num_test, 1) Xn_test];

% ====================== YOUR CODE HERE ======================
accuracy = evaluateAccuracy(beta, Xn_test, y_test);

% ============================================================

% Display the prediction accuracy
fprintf('Accuracy:\n %f\n', accuracy);
fprintf('\n');
