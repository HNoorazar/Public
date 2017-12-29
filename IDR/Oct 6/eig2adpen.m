% This funtion computes eigenvalues of a 2-by-2 pencil using excersise
% 5.6.31 of F.M.C. (It can be more accurate, by taking some situation into 
% account. look at page 383, ex. 5.6.31, part b)

function v=eig2adpen(H,U)

Uinv=[1/U(1,1)  -U(1,2)/(U(1,1)*U(2,2)); 0 1/U(2,2)];
A=H*Uinv;

if (A(1,2)+A(2,1)==0)
   c=1/sqrt(2);
   s=c;
else
    that=(A(1,1)-A(2,2))/(A(1,2)+A(2,1));
    den=sqrt(1+that^2);
    t=that/(1+den);
    c=1/sqrt(1+t^2);
    s=c*t;
end

Q=[c s;-s c];
A=Q'*A*Q;
v1=A(1,1)+sqrt(A(1,2)*A(2,1));
v2=A(1,1)-sqrt(A(1,2)*A(2,1));
v=[v1 v2]';
end