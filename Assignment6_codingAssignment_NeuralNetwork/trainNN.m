function [beta1, beta2, J_history] = trainNN(X, y, beta1, beta2, alpha, num_epochs)
%trainNN train the neural network model using backpropagation algorithm. It
%updates the weights, beta1 and beta2 using the training examples. It also
%generates the cost computed after each epoch. 

% useful values
[n, ~] = size(X); % n is number of training examples
num_hidden = length(beta1(:,1)); % number of hidden units (bias not included)
num_output = length(beta2(:,2)); % number of output units

X = [ones(n,1) X]; % append the intercept X0 = 1 to each training example.
% Note that beta1 has 9 weights therefore X MUST also have 9 features
J_history = zeros(num_epochs,1);

for epoch = 1:num_epochs
% for each training example, do the following
    Jd = 0;
    for d = 1:n
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the weights beta1 and
    %               beta2. The key steps are indicated as follows
    %
    %
   
    
        %% Step 1: forward propagate to generate the network output
        % Let "H" denote the hidden layer.
        % H(2) shall be the first hidden unit; H(3) shall be the second hidden unit.
        % Up to n hidden units... % Recall H(1) shall be the bias unit equal to
        % one.
        H = ones(1,num_hidden+1);
        for i = 2:num_hidden+1
            H(i) = sigmoid( dot( beta1(i-1,:), X(d,:) ) );
        end

        out = sigmoid( dot(beta2, H) );
        
        %% Step 2: for each output unit, calculate its error term
        % Recall that the number of output units is num_output
        % For this dataset, we only have one output
        
        err_o = out * (1-out) * (y(d) - out);

        %% Step 3: for each hidden unit, calculate its error term
        % Recall that number of hidden units is num_hidden+1, including
        % bias unit
        err_h = ones(1, num_hidden+1);
        
        for j = 1:num_hidden
            err_h(j+1) = H(j+1) * (1 - H(j+1)) * err_o * beta2(j+1);
        end
        
        
        %% Step 4: update the weights using the error terms
        for k = 1:num_hidden
            beta1(k,:) = beta1(k,:) + ( alpha * err_h(k+1) * X(d,:) );
        end
        beta2 = beta2 + ( alpha * err_o .* H );
        %% calculate the cost per epoch
        Jd = Jd + ( y(d) - out)^2;

    end
    J_history(epoch) = Jd/2;
end