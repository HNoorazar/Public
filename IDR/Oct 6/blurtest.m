function [A1,Q1] = blurtest(A,Q,n,m,top,bot,watch_chase)
%
% This code checks for shift blurring.  It is the same as downchase.m, 
% except that it computes the eigenvalues of the intermediate bulge 
% pencils at the beginning and end of the iteration and compares them
% with the shifts.
%
% This file does a QR iteration of degree m 
% on the submatrix A(top:bot,top:bot) of the
% matrix A of dimension n.
% This program will malfunction unless 0 < top, bot < n+1,
% and bot - top > m.
% Standard shifts are used.
% We "cheat" by using eig to compute the shifts.
shift = eig(A(bot-m+1:bot,bot-m+1:bot));
% Build vector to create initial bulge.
x = [1;zeros(m,1)];
for k = 1:m
    x(1:k+1) = A(top:top+k,top:top+k-1)*x(1:k) - shift(k)*x(1:k+1);
    x = x/norm(x);
end
lr = top + m;
% Start main loop
for k = top:bot-1
    if k > top
        lr = min(lr+1,bot);
        x =A(k:lr,k-1);
    end
    [v,beta,s] = reflector(x,1);
    y = v'*A(k:lr,k:n); % apply transformation on left.
    z = v*beta;
    A(k:lr,k:n) = A(k:lr,k:n) - z*y;
    if k > top, A(k:lr,k-1) = zeros; A(k,k-1) = s; end
 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    if k == top | k == bot - m - 1 % compute bulge pencil eigenvalues
        if k == top 
            disp('at top')
        elseif k == bot - m - 1
            disp('at bottom')
        end
        F = eye(m+1,m);
        y = v'*F; z = v*beta;
        F = F - z*y; F = F(2:m+1,:);
        E = A(k+1:lr,k:lr-1);
        bulgeig = eig(E,F);
        shiftdif = max(abs(sort(shift)-sort(bulgeig)))
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    lrp = min(lr+1,bot);
    y = A(1:lrp,k:lr)*v; % apply transformation on right.
    A(1:lrp,k:lr) = A(1:lrp,k:lr) - y*z';
    y = Q(1:n,k:lr)*v; % update transforming matrix.
    Q(1:n,k:lr) = Q(1:n,k:lr) - y*z';
    
    if watch_chase == 1, showmatrix(A); end

end
A1 = A; Q1 = Q;

