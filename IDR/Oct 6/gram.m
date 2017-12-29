% This function takes A as a matrix and orthogonalizes the last column of A
% with previous columns

function [r,v]=gram(A)
[~,m]=size(A);
v=zeros(m-1,1);

for ii=1:m-1
   v(ii)=(dot(A(:,m),A(:,ii)))/dot(A(:,ii),A(:,ii));
end
r=A(:,m);

for ii=1:m-1
   r=r-v(ii)*A(:,ii);
end
end