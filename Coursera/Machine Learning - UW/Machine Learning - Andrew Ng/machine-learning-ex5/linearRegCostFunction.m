function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%   LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%   regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
%    Instructions: Compute the cost and gradient of regularized linear 
%    regression for a particular choice of theta.
%
%    You should set J to the cost and grad to the gradient.
%

htheta = X*theta;
hthetay1 = htheta - y;
hthetay = (hthetay1).^2;
thetas = theta.^2;
J =(1/(2*m)) * sum(hthetay) + lambda/(2*m) * sum(thetas(2:end));


alpha = (1/m);
grad = alpha * hthetay1'*X;
%grad(1) = alpha * sum(hthetay1);
grad(2:end) = grad(2:end) + (lambda/m)*theta(2:end)';



% =========================================================================

grad = grad(:);

end
