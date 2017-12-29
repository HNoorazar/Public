%This function checks subdiagonal of a hessenebrg matrix, extracts
%positions of zeros and also the place of those 2-by2 submartices with
%imaginary eigenvalues


function l=subchk(H,U)

n=length(H);
l=zeros(1,n-1);

for jj=1:n-1          % This loop looks for zeros
    
    if (H(jj+1,jj)==0)
        l(jj)=jj;
    end
end

if(n==2 && H(2,1)~=0)
   [e1]=myeig(H,U);
 if(imag(e1)~=0)
     l(1)=1; 
 end
end

if(n>2)
if(H(2,1)~=0 && H(3,2)==0) %This "if" checks whether the eigenvalues of (H(1:2,1:2),U(1:2,1:2)) are real or imaginary
 
 [e1]=myeig(H(1:2,1:2),U(1:2,1:2));
 if(imag(e1)~=0)
     l(1)=1;
 end
end

if(H(n-1,n-2)==0 && H(n,n-1)~=0 )
  e1=myeig(H(n-1:n,n-1:n),U(n-1:n,n-1:n));
  if(imag(e1)~=0)
     l(n-1)=n-1;
  end
end

end

%In this loop we look for 2-by-2 submatrices 
if (n>3)
for zz=1:n-3
  if (H(zz+1,zz)==0 && H(zz+2,zz+1)~=0 && H(zz+3,zz+2)==0 )
     e1=myeig(H(zz+1:zz+2,zz+1:zz+2),U(zz+1:zz+2,zz+1:zz+2));
     if (imag(e1) ~= 0)
         l(zz+1)=zz+1;
     end
  end
end
end

 
end  
