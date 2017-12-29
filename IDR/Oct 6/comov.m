% This function takes quasi triangular matrix A and counts 
% the nonzero elements of subdiagonal. 

function cv=comov(A)
cv=0;
m=length(A);

for ii=1:m-1
    if (A(ii+1,ii)~=0)
    cv=cv+1;
    end
end
end
