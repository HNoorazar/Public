function v=lz_shift_idrmin(U,D)
d=D(1,1);
v2=1/d;
v1=v2+v2*(U(1,2)/(U(1,1)*U(2,2)));
v=[v1; v2];

end