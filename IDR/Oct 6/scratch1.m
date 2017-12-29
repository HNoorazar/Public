clear all;
clc;

format short

n=7;
A=rand(n,n);
B=rand(n,n);
A=hess(A);
B=triu(B);
[C,D,L,Linv,Z]=lzff(A,B);
C
e=myeigp(C,D)'
vs=findind(e,3)
%[T,S]=move(C,D,2);
%T

