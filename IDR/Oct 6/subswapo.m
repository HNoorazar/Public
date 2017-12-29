% This function swaps block of a matrix by Sylvester equation.

function [L, Linv, s]=subswapo(E,F,G)

k=length(E);
m=length(F);
D=zeros(m,k);
n=m+k;
M=zeros(m+k,m+k);
M(1:k,1:k)=E;
M(k+1:n,k+1:n)=F;
M(1:k,k+1:n)=G;

X=lyap(-E,F,-G);
X=[X;eye(m)];

[L, Linv, U]=myLU(X);

s=1;
end


