function [e1,e2]=quad(a,b,c)
delta=b^2-(4*a*c);

if (b == 0)
     
    e1=sqrt(-c/a);
    e2=-sqrt(-c/a);
    
else
     q=-b - sign(b)* sqrt(delta); 
     e1=(2*c)/q;
     e2=q/(2*a);
end

end