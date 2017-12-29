function x=sv_idrmin(U,D,sh)
sa=sh(1)+sh(2);
sp=sh(1)*sh(2);

kasr1=1/(D(1,1)^2);

x1=kasr1+sp-sa/D(1,1);

x2=-1/D(1,1)-1/D(2,2)+sa;
x2=x2/(D(1,1)*U(1,1));

x3=U(1,1)*U(2,2)*D(1,1)*D(2,2);
x3=1/x3;

x=[x1 x2 x3]';

end