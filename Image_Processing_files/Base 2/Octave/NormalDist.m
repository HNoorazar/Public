% This is normal distance function 
function  NormalDist = NormalDist(x,y,mu, sigma, alpha, s)
 % x: a vector
 % y: a vector
 % mu: the mean
 % sigma: the standard deviation
 % s: the number of layers
 % alpha: some constant
 % =============================
 % Compute the length of vector x (y)
 l = length(x);
% Compute the length of signature of each layer
 d = l/s;



% Construct the normal distribution as a function as the layer number s 
 idx = (1:1:s);
 normalfun  = alpha*exp(-(idx-mu).^2/(2*sigma));
 normaldist = diag(reshape(repmat(normalfun, d,1),1,l));



% return the normal distance between x and y
 NormalDist = x*normaldist*y';
endfunction