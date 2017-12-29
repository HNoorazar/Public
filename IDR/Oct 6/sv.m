% This function will get A and B matrices as upper hess and upper trian.
% and two shifts ro1 and ro2 and it will compute
% (AB^(-1)-ro2I)(AB^(-1)-ro1I)e1. for special case x is 3-by-1.

%sv=Starting Vector
function [x]=sv(A,B,s1,s2)

sa=s1+s2;
sp=s1*s2;
e1=[1 0 0]';
y=(1/B(1,1))*([A(1,1) A(2,1) 0])';
v=sp*e1-sa*y;
z2=y(2)/B(2,2);
z1=(y(1)-B(1,2)*z2)/B(1,1);
%z=[z1 z2 0]';
Az=A(1:3,1:2)*[z1;z2];
x=Az+v;

%y=[(A(1,1)/B(1,1))-s1  A(2,1)/B(1,1) 0]';
%z2=y(2)/B(2,2);
%z1=(y(1)-(B(1,2)*z2))/B(1,1);
%z=[z1 z2]';
%w=(1/B(1,1))*(y(1)-((y(2)*B(1,2))/B(2,2)))*[A(1,1); A(2,1); 0]+(y(2)/B(2,2))*A(1:3,2)-s2*y;
%z=A(1:3,1:2)*z;
%y=s2*y;
%x=z-y;

end
