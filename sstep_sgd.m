function [X, objval, accuracy] = sstep_sgd(A, x, y, options)
% This function performs s-step stochastic gradient descent for logistic regression.
% Inputs:
%   A        - data matrix (m x n)
%   x        - initial parameter vector (n x 1)
%   y        - labels vector (m x 1), uses +1/-1 and may be padded with zeros
%   options  - struct with fields:
%                batch_size     - size of each minibatch
%                s              - number of sequential steps per gradient computation
%                eta            - step size (learning rate)
%                precision      - numeric precision indicator (0.5, 1, or 2)
%                maxiters       - maximum number of iterations
%                print_interval - interval (in iterations) to record/print progress
%
% Outputs:
%   X        - matrix whose columns are recorded iterates of x at print intervals
%   objval   - vector of objective values recorded at print intervals
%   accuracy - vector of training accuracies recorded at print intervals
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
    objective_fn = @(matrix, vector) sum(log(1 + exp(-matrix*vector)));
    prev_objval = objective_fn(A_scaled_y, x)/m;
    num_batches = ceil(m/options.batch_size);
    batch_ptr = [0, min(m,cumsum(options.batch_size*ones(1,num_batches)))];
    idx = 1;
    objval = [];
    accuracy = [];    
    X = x;
    m_unpadded = find(y == 0, 1) - 1;
    objval(1) = objective_fn(A_scaled_y(1:m_unpadded,:), x)/m_unpadded;
    probabilities = sigmoid_fn(A*x);
    probabilities(probabilities > 0.5) = 1;
    probabilities(probabilities <= 0.5) = -1;
    accuracy(1) = 100*(sum(probabilities(1:m_unpadded) == y(1:m_unpadded))/m_unpadded);

    while(iters < options.maxiters)

        batch_indices = batch_ptr(idx)+1:batch_ptr(idx+options.s);
        gradient = compute_sstep_gradient(options.batch_size, options.s, options.eta, A_scaled_y(batch_indices,:), x);
        x = x - options.eta*gradient;
        idx = idx + options.s;
        iters = iters + options.s;
        % cyclic s-step SGD
        if(idx == length(batch_ptr))
            idx = 1;
        end
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

function gradient = compute_sstep_gradient(batch_size, s, eta, A, x)
    % The following block is intentionally left for explanatory comments
    % about the s-step gradient computation implemented below.
    %
    % This function computes an s-step gradient for a minibatch
    % of logistic loss terms. The input A is expected
    % to be a vertically stacked matrix containing s blocks each of size
    % batch_size-by-n (so A has s*batch_size rows). The algorithm uses a
    % compact recurrence to perform s sequential
    % stochastic gradient updates without updating x in between.
    %
    % Key quantities:
    %  - G = A*A'  : block Gram matrix between all samples in the s blocks.
    %  - correction: initially A*x (vector of linear predictions for all
    %                samples). It is updated blockwise to implicitly propagate the
    %                previous updates.
    %
    % For each block i = 1..s:
    %  - accumulate contributions from previous blocks j < i via the
    %    term eta * G(i_block, j_block) * correction(j_block). This models
    %    how an update computed on block j would affect inputs
    %    for block i after a gradient update.
    %  - apply the elementwise sigmoid mapping appropriate for the gradient
    %    of the logistic loss.
    %
    % After processing all s blocks, the returned gradient is a linear combination
    % of A transpose times the final correction vector, which matches the
    % sum of s gradients of classical SGD.
    %
    % The implementation below relies on the precomputed G and performs
    % block-index arithmetic using batch_size to walk through the stacked
    % blocks in A.
    G = A*A';
    correction = A*x;
    i_start = 1;
    i_end = batch_size;
    for i = 1:s
        j_start = 1;
        j_end = batch_size;
        for j = 1:i-1
            correction(i_start:i_end) = correction(i_start:i_end) ...
                     + eta*G(i_start:i_end, j_start:j_end)*correction(j_start:j_end);

            j_start = j_start + batch_size;
            j_end = j_end + batch_size;
        end
        correction(i_start:i_end) = sigmoid(correction(i_start:i_end));
        i_start = i_start + batch_size;
        i_end = i_end + batch_size;
    end
    gradient = -A'*correction;
end

function result = sigmoid(z)
    result = ones(size(z))./(ones(size(z)) + exp(z));
end
