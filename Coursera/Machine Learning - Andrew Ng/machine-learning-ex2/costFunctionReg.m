function [J, grad] = costFunctionReg(theta, X, y, lambda)
%   COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



thetaX = theta'*X';
h_theta = sigmoid(thetaX);

logthetaX = log(h_theta);
logthetaXm = log (1-h_theta);

costsum = logthetaX*y + logthetaXm*(1-y);
J=-(1/m)*costsum;
thetahelp=theta(2:length(theta));
J=J+(lambda/(2*m))*sum(thetahelp.^2);

h_thetay=h_theta-y';
grad = (1/m)*h_thetay*X;
grad(2:length(grad))= grad(2:length(grad))+(lambda/m)*theta(2:length(theta))';

% =============================================================

end
