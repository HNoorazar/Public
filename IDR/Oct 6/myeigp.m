
% This function computes eigenvalues of pencil (A,B) where both are
% (quasi) upper-triangular.

function v=myeigp(A,B)
n=length(A);
v=zeros(1,n);
    
    if (A(2,1)==0)
         v(1)=myeigg2(A(1,1),B(1,1));
    elseif (A(2,1)~=0)
        vc=myeigg2(A(1:2,1:2),B(1:2,1:2));
        v(1)=vc(1);
        v(2)=vc(2);
    end
    
for ii=2:n-1
    
   if (A(ii,ii-1)==0 && A(ii+1,ii)==0)    
       v(ii)=myeigg2(A(ii,ii),B(ii,ii));    
   elseif (A(ii+1,ii)~=0) 
       vc=myeigg2(A(ii:ii+1,ii:ii+1),B(ii:ii+1,ii:ii+1));
       v(ii)=vc(1);
       v(ii+1)=vc(2);
   end       
end


if (A(n,n-1)==0)
    v(n)=myeigg2(A(n,n),B(n,n));
end

%if (A(n-1,n-2)==0 && A(n,n-1)==0)
%    v(n)=myeigg2(A(n,n),B(n,n));
%    v(n-1)=myeigg2(A(n-1,n-1),B(n-1,n-1));
    
%elseif(A(n-1,n-2)~=0)
%    vc=myeigg2(A(n-2:n-1,n-2:n-1),B(n-2:n-1,n-2:n-1));
 %   v(n-2)=vc(1);
  %  v(n-1)=vc(2);
   % v(n)=A(n,n)/B(n,n);
%elseif (A(n,n-1)~=0)
 %   vc=myeigg2(A(n-1:n,n-1:n),B(n-1:n,n-1:n));
  %  v(n-1)=vc(1);
   % v(n)=vc(2);
%end


end

