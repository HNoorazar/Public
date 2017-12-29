function v=sv_idrminNEW(U,D,sh)
sa=sh(1)+sh(2);
sp=sh(1)*sh(2);

x1=1/U(1,1);

x=[x1-sh(2) -x1 0]';

z1=x(1)/U(1,1);
z2=x(2)/U(2,2);

z2=z2-z1;
z3=-z1;

z=[z1 z2 z3]';

v=z-sh(1)*x;


end