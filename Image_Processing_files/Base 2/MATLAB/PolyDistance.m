% Define a distance function
function  PolyDist = PolyDistance(x,y,s, k, A)
% x: vector
% y: vector 
% x and y should have the same dimension
% s: number of layer
% k: degree of the polynomial minus 1 since there is a constant term
% A: store the coefficients for the polynomial
% length of A should be k


% =============================
% Compute the length of vector x (y)
l = length(x);
% Compute the length of signature of each layer
d = l/s;


% =============================
% Create the weight function
% Define the layer numbers
idx = (1:1:s);



% =============================
% Create all the polynomials, e.g. idx^1, idx^2, ... , idx^k 
% Use Vandermone Matrix to construct the polynomials
tempPD = vander(idx,k);
PD = fliplr(tempPD);
PD = PD*A';


% Contruct a diagonal matrx with each layer the weight in A
temp = repmat(PD,1,d);
temp = temp';
temp = reshape(temp, 1, l);
weight = diag(temp);

% print the result
PolyDist = x*weight*y';


endfunction