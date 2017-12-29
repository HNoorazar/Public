format long

 

h=.1;      %delta x 

D=.5;%dissusion coefficient

t=5;

N=100;

k=1/N;    %delta t

r=50*k;    %1/(2*(h^2))

gu=zeros(11,5);

uint=zeros(11,1);     %initial u

for i=0:10

    if i*h<=.5

        uint(i+1,1)=i*h;

    end

    if i*h>.5

        uint(i+1,1)=1-i*h;

    end

end

uold=uint;

unew=zeros(11,1);

for n=1:(t/k)

    for i=1:9

       unew(1,1)=uold(1 ,1);

       unew(11,1)=uold(11,1);

       unew(i+1,1)=r*uold(i,1)+(1-2*r)*uold(i+1,1)+r*uold(i+2,1);

    end

    uold=unew;
    
    figure(1) ;   
    plot(1:length(uold),uold,'b.-');
    title(sprintf('iteration %4d',n));
    axis([1 11 0 0.01]);

end 

    uold
    
figure(2)    
plot(1:length(uold),uold,'b.-')
% legend('computed','true')
title('Numerical result for Forward difference')
% axis([1 11 0 1e-12])