double;
%%m = number of intervals in x (10)
% k = delta t = time step size
% D = diffusion coeff (alpha 2)= 0.5
%
%%
function forward_heat(m,k,N,D)%D=0.5 m = 10
m=10;
N=2020;
k=.0005;
D=1;

h=1/m;%h=x step size
% k = time step size

mu = (D*k)/(h^2);
%%% t = 0 init
w=zeros(m+1,1);
for i = 2:m
%w(i)=f((i-1)*h);
w(i)=test_f((i-1)*h);
end
%disp(w);
%%%%%

%%%%x = 0 init
%???
%%%%%
v=zeros(m+1,1);
t=0;
exactanswer = zeros(m+1,1);
for j = 1:N %stepping through time
t=(j-1)*k;
v(1)=0;v(m+1)=0;
for i = 2:m %step through space
v(i)=mu*w(i-1) + (1-(2*mu))*w(i)+mu*w(i+1);
%exactanswer(i)=exact_answer(t,(i-1)*h,D);
exactanswer(i)=exact_answer_test(t,(i-1)*h);
end
if(t==0.5)
disp(v);
disp(exactanswer);
end
w=v;
end

function out = f(x)% x boundary condition
if(x >= 0 && x <= 0.5)
out=x;
end
if(x > 0.5 && x <= 1)
out = 1-x;
end

function out = exact_answer(t,x,D)%is this correct?
sum=0;
for k=1:13
vxt=4*(1/((k*pi).^2))*sin(k*pi/2)*sin(k*pi*x)*exp(-D*(k*pi).^2*t);
sum = sum+vxt;
end
out = sum;

function out = test_f(x)
if(x >= 0 && x <= 1)
out = sin(pi*x);
end

function out = exact_answer_test(t,x)
out = exp(-((pi^2) * t))*sin(pi*x);