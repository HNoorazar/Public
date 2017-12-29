% This function will get A as an n-by-5 matrix, and will orthogonalize the
% last column of A with the first four columns.

function [r,v]=gramOR1(A)
[~,m]=size(A);
v=zeros(m-1,1);

for ii=1:m-1
    v(ii)=dot(A(:,m),A(:,ii))/dot(A(:,ii),A(:,ii));
end
r=A(:,m);
for ii=1:m-1
    r=r-v(ii)*A(:,ii);
end
end