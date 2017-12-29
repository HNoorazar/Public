% This function computes v=(AB^-1-shift1*I)e_1
% So, we have only one shift


function v=sv1s(A,B)
n=length(A);
[sh1,sh2]=myeig(A(n-1:n,n-1:n),B(n-1:n,n-1:n));
w=eig(A(n,n),B(n,n));

d=[sh1-w sh2-w];
ad1=abs(d(1));
ad2=abs(d(2));

if (ad1<ad2)
    shift=sh1;
else
    shift=sh2;
end

v1=A(1,1)/B(1,1)-shift;
v2=A(2,1)/B(1,1);
v=[v1 v2]';
v=real(v);
end
