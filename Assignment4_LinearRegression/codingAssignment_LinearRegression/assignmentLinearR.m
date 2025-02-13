%% SDSU Machine Learning Course (EE600/CompE596)
%% Programming Assignment:  Linear regression 
% PAULIE LEE
% 10/01/2021 -- ASSIGNMENT 4
% Dataset comes from: 
% https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  linear regression assignment. 
%
%  You will need to complete the following functions in this 
%  exericse:
%
%     loadData.m
%     normalEqn.m
%     evaluateAccuracy.m
%     featureNormalize.m
%     gradientDescent.m
%     computeCost.m
%
%  For this part of the exercise, you will need to change some
%  parts of the code below for various experiments (e.g., changing
%  learning rates).
%
% Initialization
clear ; close all; clc

%% ================ Part 1: Normal Equations ================

fprintf('Solving with normal equations...\n');

% Instructions: The following code computes the closed form 
%               solution for linear regression using the normal
%               equations. You should complete the code in loadData.m,
%               normalEqn.m
%
%               After doing so, you should complete this code 
%               to predict the price of a house that is 2 year 
%               old, 500 meter to the nearest MRT station, and 
%               has 8 convenience stores in the living circle 
%               on foot

%% Load data and split data into a training set and testing set
% ====================== YOUR CODE HERE ======================
[X_train, y_train, X_test, y_test] = loadData();

% ============================================================

% Print out some data points
fprintf('First 10 examples from the training dataset: \n');
fprintf(' x = [%.0f %.0f %.0f], y = %.0f \n', [X_train(1:10,:) y_train(1:10,:)]');
fprintf('\n');

num_train = length(y_train); % number of training examples

% Add intercept term to X_train
X_train = [ones(num_train, 1) X_train];

%% Calculate the parameters from the normal equation
% ====================== YOUR CODE HERE ======================
beta = normalEqn(X_train, y_train);

% ============================================================

% Display normal equation's result
fprintf('beta computed from the normal equations: \n');
fprintf(' %f \n', beta);
fprintf('\n');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ========== Part 2: Evaluate performance =============

fprintf('Evaluate the prediction accuracy ...\n');

% Instructions: The following code evaluates the performance of
%               the trained linear regression model. You should 
%               complete code in evaluateAccuracy.m
%

num_test = length(y_test); % number of testing examples

% Add intercept term to Xn_test
X_test = [ones(num_test, 1) X_test];

% ====================== YOUR CODE HERE ======================
error = evaluateAccuracy(beta, X_test, y_test);

% ============================================================

% Display the average prediction error
fprintf('Average prediction error (using normal equations):\n %f\n', error);
fprintf('\n');

%% Estimate the price of a house 
% that is 2 year old, 500 meter to the nearest MRT station, has 8 
% convenience stores in the living circle on foot
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized.
price = [1, 2, 500, 8] * beta;

% ============================================================

fprintf(['Predicted price of the house ' ...
         '(using normal equations):\n $%f\n'], price);
fprintf('\n');


fprintf('Program paused. Press enter to continue.\n');
pause;


%% ================ Part 3: Gradient Descent ================

fprintf('Running gradient descent ...\n');

% Instructions: The following code applies gradient descent to 
%               estimate the parameters in a linear regression 
%               model. You should complete code in featureNormalize.m,
%               gradientDescentMulti.m and normalEqn.m
%
%               Try running gradient descent with 
%               different values of alpha and see which one gives
%               you the best result.
%
%               After that, you should complete the code at the end
%               to predict the price of a house.
%
%


% Scale features and set them to zero mean
fprintf('Normalizing Features ...\n');

% Normalize the features. 
% ====================== YOUR CODE HERE ======================
[Xn_train mu sigma] = featureNormalize(X_train(:,2:end));

% ============================================================

% Add intercept term to X_train
Xn_train = [ones(num_train, 1) Xn_train];

% Choose some alpha value
% ====================== YOUR CODE HERE ======================
alpha = 0.001;
num_iters = 20;
% ============================================================

% Init beta 
beta = zeros(4, 1);

%% Run Gradient Descent 
% ====================== YOUR CODE HERE ======================
[beta, J_history] = gradientDescent(Xn_train, y_train, beta, alpha, num_iters);

% ============================================================

% Plot the convergence graph
figure;
plot(1:numel(J_history), J_history, '-b', 'LineWidth', 2);
xlabel('Number of iterations');
ylabel('Cost J');

% Display gradient descent's result
fprintf('beta computed from gradient descent: \n');
fprintf(' %f \n', beta);
fprintf('\n');

%% Evaluate the accuracy of the derived model
Xn_test = (X_test - [0, mu])./[1, sigma];

error = evaluateAccuracy(beta, Xn_test, y_test);

% Display the average prediction error
fprintf('Average prediction error (using gradient descent):\n %f\n', error);
fprintf('\n');

%% Estimate the price of a house 
% that is 2 year old, 500 meter to the nearest MRT station, has 8 
% convenience stores in the living circle on foot
% ====================== YOUR CODE HERE ======================
% Recall that the first column of X is all-ones. Thus, it does
% not need to be normalized. But the other columns need to be normalized

predict = [2, 500, 8];
% Normalize the features:
predict_norm = (predict - mu) ./ sigma;

price = [1, predict_norm] * beta;

% ============================================================

fprintf(['Predicted price of the house ' ...
         '(using gradient descent):\n $%f\n'], price);
