function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    %% fprintf('θ and J(θ) found by gradient descent: ');
    %% fprintf('%f %f  %f %f \n', theta(1), theta(2), theta(3), computeCostMulti(X, y, theta));
    
    s = zeros(size(X, 2), 1);
    
    for i = 1:m
        s = s + (theta' * X(i, :)' - y(i, :)) .* X(i, :)';
    end
    
    %%% simultaneously change θ0, θ1, θ2
    temp = theta - (alpha / m) * s;
    theta = temp;  

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
