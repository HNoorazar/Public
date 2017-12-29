function [x,res] = Gmresd(A,b,kmax,tol)
%
%   Usage     [x,rho] = Gmres(A,b,tol);
%
%   Solves  Ax = b
%
%   Input:    A  -- an n by n matrix
%
%
%          kmax  --  max no. GMRES steps 
%
%           tol  -- stopping tolerance 
% 
%
%   Output:   x  -- an n  vector... approx solution Ax = b
%
%           res  -- Residual history res(j) = (norm(b -Ax_j)  
    k = kmax;
    q = eye(k+1,1);
%
%   The rhs b is scaled to unit length 
%   size will be restored after convergence
%
    theta = norm(b);
    v = b/theta;
%
    n = length(v);
    R = zeros(k);
    V = zeros(n,k);

    v = v/norm(v);
    w = A*v;
    alpha = v'*w;
    
    f = w - v*alpha;
        c = v'*f;
        f = f - v*c;
        alpha = alpha + c;

    R(1,1) = alpha;
    V(:,1) = v; 
    rho = 1;

    Q = eye(k+1);
    res = [];    

        

    j = 1;
    while (rho > tol && j < k),
       
        j = j+1;
        beta = norm(f);
        v = f/beta;
%
%       Compute the Givens transformation G to zero
%       last row of updated H (H is not stored)
%       to get H = QR
%        
        [G,t] = qr([R(j-1,j-1) ; beta]);
        Q(:,j-1:j) = Q(:,j-1:j)*G;

%
%       Update rhs  q = Q'*e_1
%
%       Q should not be stored, it should be 
%       represented by the Givens transformations
%
        q(j-1:j) = G'*q(j-1:j);
        R(j-1,j-1) = t(1);
        rho = abs(q(j));

        res = [res; rho];

        V(:,j)   = v;

        if (rho > tol),
           w = A*v;
           h = V(:,1:j)'*w;
           f = w - V(:,1:j)*h;
               c = V(:,1:j)'*f;
               f = f - V(:,1:j)*c;
               h = h + c;

           R(1:j,j) = Q(1:j,1:j)'*h;
        end

    end 

%   After convergence (or max iter exceeded)
%   compute the solution x.  Must multiply by 
%   theta to restore rhs b to original length
%

%   res = res*theta;
    res = res;  % relative residual

    y = R(1:j-1,1:j-1)\q(1:j-1);
    x = V(:,1:j-1)*y*theta;
end