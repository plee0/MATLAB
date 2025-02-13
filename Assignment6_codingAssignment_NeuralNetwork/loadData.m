function [X_train, y_train, X_test, y_test] = loadData()
%   LOADDATA imports data downloaded from 
%   http://networkrepository.com/pima-indians-diabetes.php
%   and splits the dataset into two sets: training set and testing set
%

 % ====================== YOUR CODE HERE ======================
    % Instructions: Import spreadsheets data, extract the first
    % 8 columns and store them as X. Extract the last column and 
    % store it as y. 
    filename = 'pima-indians-diabetes.csv';
    temp_table = readtable(filename);
    temp_arr = table2array(temp_table);
    X = temp_arr(:, 1:8);
    Y = temp_arr(:,9);
    % Randomly pick 70% of the data examples as the training set and the 
    % the rest as the testing set
    L = length(Y);
    R = randperm(L); % generates random permutation from 1 to L
    Sep = ceil(.70 * L); % Round 70% upwards
    i = 1;
    j = 1;
    while i < L + 1
        if i < Sep
            % First 70% of data is stored as X/Y training data
            X_train(i,:) = X(R(i), :);
            y_train(i,1) = Y(R(i));
        else
            % Latter 30% of data is stored as X/Y test data
            X_test(j,:) = X(R(i), :);
            y_test(j,1) = Y(R(i));
            j = j + 1;
        end
        i = i + 1;
    end
        
    % Hint: You might find the 'readtable' and 'table2array' functions useful.
    %




% ============================================================
end