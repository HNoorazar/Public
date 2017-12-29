% This function gives eigenvalues of a 2-by-2 or 1-by-1 pencil.

function v=myeigg2(A,B)

if (length(A)==2)
% a=B(1,1)*B(2,2)-B(1,2)*B(2,1);
% b=A(2,1)*B(1,2)+A(1,2)*B(2,1)-A(2,2)*B(1,1)-A(1,1)*B(2,2);
% c=A(1,1)*A(2,2)-A(2,1)*A(1,2);
[v1,v2]=myeig(A,B);
v=[v1,v2];

elseif(length(A)==1)
   if (B==0)
       v=inf;
   elseif(B~=0)
       v=A/B;
   end
    
end
end
