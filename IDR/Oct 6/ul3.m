% M is a 3-by-3 matrix. P is a matrix such that M*P=U where U is upper
% triangular



function P=ul3(M)
x=M(3,:);
G=ger3(x);
M=M*G;
M(3,1:2)=zeros(1,2);

Gh=ger(M(2,1:2));
Gti=eye(3);
Gti(1:2,1:2)=Gh;
P=G*Gti;


end