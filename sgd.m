function [X, objval, accuracy] = sgd(A, x, y, options)
% This function performs stochastic gradient descent for logistic regression.
% Inputs:
%   A       - data matrix (m x n), rows are examples
%   x       - initial parameter vector (n x 1)
%   y       - label vector (m x 1), labels are +1/-1 (and possibly padded with 0s)
%   options - struct with fields:
%               eta            : learning rate
%               batch_size     : mini-batch size
%               maxiters       : maximum number of SGD iterations
%               print_interval : iterations between logging
% Outputs:
%   X       - matrix collecting iterates at logging times (n x k)
%   objval  - vector of objective values at logged iterates
%   accuracy- vector of training accuracies (percent) at logged iterates
%
% Notes:
% - The code assumes the first portion of y contains true labels (+1/-1)
%   and any trailing zeros are padding. m_unpadded finds the length of the
%   unpadded portion for reporting training metrics.
% - A_scaled_y is precomputed as A .* y to simplify the gradient for the
%   logistic loss when labels are +1/-1.
% - The sigmoid, gradient, and objective functions are defined as function
%   handles below for clarity and vectorized computation.
% - Mini-batches are defined by batch_ptr; iteration cycles through batches.
% - X starts with the initial x and appends iterates every print_interval.
% - This comment block is intentionally concise and placed at the top of the
%   function to explain inputs, outputs, and key implementation choices.
    [m, ~] = size(A);
    iters = 0;
    A_scaled_y = A.*y;
    sigmoid_fn = @(z) ones(size(z))./(ones(size(z)) + exp(-z));
    gradient_fn = @(matrix, vector) -matrix'*sigmoid_fn(-matrix*vector);
    objective_fn = @(matrix, vector) sum(log(1 + exp(-matrix*vector)));
    num_batches = ceil(m/options.batch_size);
    batch_ptr = [0, min(m,cumsum(options.batch_size*ones(1,num_batches)))];
    idx = 1;
    objval = [];
    accuracy = [];
    % compute initial objective error and accuracy
    m_unpadded = find(y == 0, 1)-1;
    objval(end+1) = objective_fn(A_scaled_y(1:m_unpadded,:), x)/m_unpadded;
    probabilities = sigmoid_fn(A*x);
    probabilities(probabilities > 0.5) = 1;
    probabilities(probabilities <= 0.5) = -1;
    accuracy(end+1) = 100*(sum(probabilities(1:m_unpadded) == y(1:m_unpadded))/m_unpadded);
    prev_objval = objval(end);
    X = x;
    while(iters < options.maxiters)
        batch_indices = batch_ptr(idx)+1:batch_ptr(idx+1);
        x = x - options.eta*gradient_fn(A_scaled_y(batch_indices,:), x);
        idx = idx + 1;
        % SGD with cyclic sampling
        if(idx == length(batch_ptr))
            idx = 1;
        end
        iters = iters + 1;
        % print and store training information
        if(mod(iters, options.print_interval) == 0)
            X(:,end+1) = x;
            objval(end+1) = objective_fn(A_scaled_y(1:m_unpadded,:), x)/m_unpadded;
            probabilities = sigmoid_fn(A*x);
            probabilities(probabilities > 0.5) = 1;
            probabilities(probabilities <= 0.5) = -1;
            accuracy(end+1) = 100*(sum(probabilities(1:m_unpadded) == y(1:m_unpadded))/m_unpadded);
            fprintf("Iters: %d\t Objective: %.8f\t Training Accuracy: %.4f%%\t Obj val diff %1.15e.\n", ...
            iters, objval(end), accuracy(end), abs(prev_objval - objval(end)));
            prev_objval = objval(end);
        end
    end
    fprintf("\n");
end
