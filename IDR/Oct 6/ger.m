% This function wil transfer row vector v to e_n^T.


function [G,Ginv]=ger(v)
n=length(v);

if(v(n)==0)
    sprintf('impossible')
else
    G=eye(n);
    G(n,1:n-1)=-v(1:n-1)/v(n);
    Ginv=G;
    Ginv(n,1:n-1)=-Ginv(n,1:n-1);
end