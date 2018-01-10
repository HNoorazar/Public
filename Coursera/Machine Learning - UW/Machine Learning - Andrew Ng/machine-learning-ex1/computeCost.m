function J = computeCost(X, y, theta)
%   COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.


h = theta'*X';            % Compute h_theta(x)
%for ii=1:m
 %   h(ii)=theta(1)+theta(2)*X(ii);
%end
h=h';
denom = 2*m;
frac = 1/denom;
inside = h-y;

J= frac*sum(inside.^2);

% =========================================================================

end
