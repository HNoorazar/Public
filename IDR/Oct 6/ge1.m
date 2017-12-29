% This is Gauss transformation that transfers column vector v to alpha*e1.
% G*v=alpha*e_1 or v=Ginv*(alpha*e_1)

function [G,Ginv]=ge1(x)
k=length(x);
u=-x/x(1); u(1)=1;
G=eye(k);
G(:,1)=u;
Ginv=G; Ginv(2:k,1)=-Ginv(2:k,1);
end

