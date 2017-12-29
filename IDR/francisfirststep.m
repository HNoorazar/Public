n=10;
A=rand(n,n);
H=hess(A);
HB=H;
Ro=eye(n,n);

c=H(1,1)/sqrt(H(1,1)^2+H(2,1)^2);
s=H(2,1)/sqrt(H(1,1)^2+H(2,1)^2);

Ro(1,1)=c; Ro(2,2)=c;
Ro(1,2)=s; Ro(2,1)=-s; 

H=Ro*H;
H=H*transpose(Ro)

for j=1:n
    b=HB(1,j);
    HB(1,j)=c*b+s*HB(2,j);
    HB(2,j)=c*HB(2,j)-s*b;
end

for i=1:3
    
        b=HB(i,1);
        HB(i,1)=b*c+HB(i,2)*s;
        HB(i,2)=HB(i,2)*c-b*s;
end
