% This is correlation distance between x and y
function  cdist = CorrelationDist(x,y)
 % x: a vector
 % y: a vector
 % =============================
 % Normalize vector x and y
 
 x = (1/sqrt(sum(x.^2)))*x;
 y = (1/sqrt(sum(y.^2)))*y;



% =============================
%  return the correlation distance between x and y
 cdist = x'*y;
 endfunction