% Finds eigenvalues of (A,B) a 2-by-2 Hessenberg, Upper pencil

function [e1,e2]=myeig(A,B)
%A
%B
a=B(1,1)*B(2,2);
b=A(2,1)*B(1,2)-A(1,1)*B(2,2)-A(2,2)*B(1,1);
c=(A(1,1)*A(2,2))-(A(2,1)*A(1,2));

delta=(b^2)-(4*a*c);

if (delta == 0)
     
    e1=-b/(2*a);
    e2=e1;
elseif(delta>0)
     q= -b - sign(b)* sqrt(delta); 
     e1=(2*c)/q;
     e2=q/(2*a);
elseif(delta<0)
    im=sqrt(delta)/(2*a);
    r=-b/(2*a);
    e1=r+im;
    e2=r-im;
else
    sprintf('Input has inf or NaN')
end
end
