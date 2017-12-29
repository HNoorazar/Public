% This function takes x=[x1 x2 x3] and returns G such that x*G=[0 0 x3]

function G=ger3(x)
x=x';

if(x(3)==0)
    sprintf('Impossible, last component is zero')
else
    u=[x(1)/x(3) x(2)/x(3) 0]';
    v=[0 0 1];
    G=eye(3)-u*v;
    G=G';
end
end