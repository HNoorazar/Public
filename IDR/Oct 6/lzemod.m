% UH=L*A*Z so, Linv*UH=A*Z

function [UH, UT, La, Linva, Za]=lzemod(A,B)

%checks subdiaginal entries, and if they are not small enought then it does the lz
%algorithm steps again and again.
pp=length(A);
Linva=eye(pp);
La=eye(pp);
Za=eye(pp);
[c,~,~,A]=chk(A);
dummy=0;
while (c==0)
    dummy=dummy+1;
    [A, B, L, Linv, Z]=lzsonwaypivot(A,B);
    [c,~,~,A]=chk(A);
    Linva=Linva*Linv;
    Za=Za*Z;
    La=L*La;
end
[~,~,~,A]=chk(A);
UH=A;
UT=B;

end

