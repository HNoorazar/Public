function v=vec(M)

[m,n]=size(M);
sv=m*n;
v=zeros(sv,1);

for ii=1:n
    
    k=(ii-1)*m+1;
    v(k:ii*m)=M(:,ii);
    
end
end
