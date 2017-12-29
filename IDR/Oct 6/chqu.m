% This function will check subdiagonal elements of A to see whether A is
% quasi triangular or not.

% l is a vector (of length n) whose components are corresponding to the columns whose
% subdiogonals are either zero or are asocciated with complex eigenvalues.
% gr (Green Light) tells us whether A is quasi or not. (gr will be sum of components of l)

function [gr,l,lm]=chqu(A,B)
n=length(A);
l=zeros(1,n);
l(n)=n;

if(A(2,1)==0)
    l(1)=1;
end

if (A(2,1)~=0 && A(3,2)==0)
 v=eig(A(1:2,1:2),B(1:2,1:2));
 if(imag(v(1))~=0)
     l(1)=1;
     l(2)=2;
 end
end


for jj=2:n-2
    if(A(jj+1,jj)==0)
        l(jj)=jj;
    elseif(A(jj,jj-1)==0 && A(jj+1,jj)~=0 && A(jj+2,jj+1)==0)
        v=eig(A(jj:jj+1,jj:jj+1),B(jj:jj+1,jj:jj+1));
     if(imag(v(1))~=0)
         l(jj)=jj;
     end
    end   
end


if(A(n,n-1)==0)
    l(n-1)=n-1;
end

if(A(n,n-1)~=0 && A(n-1,n-2)==0)
    v=myeigg2(A(n-1:n,n-1:n),B(n-1:n,n-1:n));
    if(imag(v(1))~=0)
        l(n-1)=n-1;
    end   
end
gr=l*ones(n,1);
lm=l(l~=0);
end
