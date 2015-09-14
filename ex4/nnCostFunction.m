function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% --------------------------------------------------------------------------
% Forward Propagation Calculations
a1 = [ones(m, 1), X]; % a1 = x adding a0 of 1
z2 = Theta1 * a1'; a2 = sigmoid(z2); a2 = [ones(m, 1) a2'];
z3 = Theta2 * a2'; a3 = sigmoid(z3);

% compute the unregularized part
for i = 1:m,
    yk = zeros(num_labels, 1);
    yk(y(i)) = 1;
    J = J + (-yk' * log(a3(:, i)) - (1 -yk') * log(1 - a3(:, i)));
end
J = J / m;

% compute the regularization term
theta1 = Theta1(:, 2:end); % exclude 1st column (don't reg. bias)
theta2 = Theta2(:, 2:end); % exclude 1st column (bias)

reg1 = 0; reg2 =0;

for i = 1: input_layer_size,
    reg1 = reg1 + theta1(:, i)' * theta1(:, i);
end

for i = 1 : hidden_layer_size,
    reg2 = reg2 + theta2(:, i)' * theta2(:, i);
end

J = J + (lambda / (2 * m)) * (reg1 + reg2);

% -------------------------------------------------------------
Delta_1 = 0; Delta_2 = 0; % initialize Δ(1) ανδ Δ(2). No need to initialize as matrix; it will take correct dimensions afte 1st addition

for t = 1:m,
  % Forward-prop for (x_t, y_t)
  a1 = [1 X(t, :)]; %a1(t) = X(t, :) plus +1 for the bias term
  a2 = sigmoid(Theta1 * a1'); a2 = [1 a2']; % compute a2 and add bias term
  a3 = sigmoid(Theta2 * a2'); % computed a3 = h_Θ(x)
  
  yk = zeros(num_labels, 1); yk(y(t)) = 1;
  delta3 = a3 - yk;
  % don't compute δ for the bias terms, so we use theta2
  delta2 = (theta2' * delta3) .* sigmoidGradient(Theta1 * a1');
  Delta_1 = Delta_1 + delta2 * a1;
  Delta_2 = Delta_2 + delta3 * a2;
end

Theta1_grad = (1 / m) * Delta_1;
Theta2_grad = (1 / m) * Delta_2;
% =========================================================================
% Add regularization
no_bias_Theta1 = Theta1; no_bias_Theta1(:, 1) = 0;
no_bias_Theta2 = Theta2; no_bias_Theta2(:, 1) = 0;
Theta1_grad = Theta1_grad + (lambda / m) * no_bias_Theta1;
Theta2_grad = Theta2_grad + (lambda / m) * no_bias_Theta2;

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
