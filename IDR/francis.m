n=10;
A=rand(n,n);

H=hess(A);

for w=1:210

rho=H(n,n);

d1=H(1,1)-rho;
d2=H(2,1);

c=d1/sqrt(d1^2+d2^2);
s=d2/sqrt(d1^2+d2^2);


for j=1:n
    b=H(1,j);
    H(1,j)=c*b+s*H(2,j);
    H(2,j)=c*H(2,j)-s*b;
end

for i=1:3
    
        b=H(i,1);
        H(i,1)=b*c+H(i,2)*s;
        H(i,2)=H(i,2)*c-b*s;
end


for k=1:n-2
    
    if H(k+2,k)==0
        break 
    else
        b=H(k+1,k);
        c=H(k+1,k)/sqrt(H(k+1,k)^2+H(k+2,k)^2);
        s=H(k+2,k)/sqrt(H(k+1,k)^2+H(k+2,k)^2);
    end
    
        for j=k:n
            b=H(k+1,j);
            H(k+1,j)= b*c + H(k+2,j)*s;
            H(k+2,j)= H(k+2,j)*c-b*s;
        end
        
        for i=1:n
           b=H(i,k+1);
           H(i,k+1)=b*c+H(i,k+2)*s;
           H(i,k+2)=H(i,k+2)*c-b*s;
        end
end
end

H
            
        
        
        