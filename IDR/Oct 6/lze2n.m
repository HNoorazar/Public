% This function takes (C,D) as a 2-by-2 pencil, where D is upper triangular
% and if the eigenvalues are real, it transfers (C,D) to (U,T) where U and T
% both are upper-triangular.

function [C,D, L2, Linv2, Z2]=lze2n(C,D)

%First column of shift polynomial:

[e1,~]=myeig(C,D);
Z2=eye(2);
L2=eye(2);
Linv2=eye(2);
ep=10^(-16);
if (imag(e1)==0)
   while (abs(C(2,1))>ep*(abs(C(1,1))+abs(C(2,2))))
       v=[C(1,1)/D(1,1)-e1; C(2,1)/D(1,1)];
       
       [G,Ginv]=ge1(v);
       C=G*C;
       D=G*D;
       L2(2,1)=L2(2,1)+G(2,1);
       Linv2(2,1)=Linv2(2,1)+Ginv(2,1);
       [~,W]=ul(D);
       D(1,1)=D(1,:)*W(:,1);
       D(2,1)=0;

       C(:,1)=C*W(:,1);
       Z2(2,1)=Z2(2,1)+W(2,1);
       
   end
end
C(2,1)=0;
end
