function [X_train, y_train, X_test, y_test] = loadData()
%   LOADDATA imports data downloaded from 
%   https://archive.ics.uci.edu/ml/datasets/Real+estate+valuation+data+set
%   and splits the dataset into two sets: training set and testing set
%
%   We only use three features as the input X: 
%       X2=the house age (unit:year)
%       X3=the distance to the nearest MRT station (unit:degree)
%       X4=the number of convenience stores in the living circle on foot (integer)
%   The output y is:
%       y=house price of unit area (10000 New Taiwan Dollar/Ping, where 
%         Ping is a local unit, 1 Ping = 3.3 meter squared)

 % ====================== YOUR CODE HERE ======================
    %% Instructions: Import spreadsheets data and extract the columns
    % corresponding to X2, X3, X4 and store them as X. Extract the last
    % column and store it as y.
    clear, clc
    T = readtable('housePriceData.xlsx', 'ReadRowNames', true);
    A = table2array(T);
    X2 = A(:,2);
    X3 = A(:,3);
    X4 = A(:,4);
    y = A(:,7);
    %% Randomly pick 70% of the data examples as the training set and the 
    % the rest as the testing set
    % Establish Training as the number of data points to divert to the
    % training set.
    Total = length(A);
    Training = ceil(.7*Total);
    r = randperm(414);
    i = 1; j = 1; k = 1;
    while i <= Total
        % First while loop stores training data
        while i <= Training
            j = r(i);
            X_train(i, 1) = X2(j);
            X_train(i, 2) = X3(j);
            X_train(i, 3) = X4(j);
            y_train(i, 1) = y(j);
            i = i+1;
        end
        % Second while loop stores test data
        while k <= Total - Training
            j = r(i);
            X_test(k, 1) = X2(j);
            X_test(k, 2) = X3(j);
            X_test(k, 3) = X4(j);
            y_test(k, 1) = y(j);
            k = k+1;
            i = i+1;
        end
    end
    

    % Hint: You might find the 'readtable' and 'table2array' functions useful.
    
 % ============================================================   

    
end