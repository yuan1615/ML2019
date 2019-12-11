clear ; close all; clc
%%  cvx ÊµÀý
m = 20; n = 10; p = 4;
rng('default');
rng(1);
A = randn(m,n);
rng(1);
b = randn(m,1);
rng(1);
C = randn(p,n); 
rng(1);
d = randn(p,1);
rng(1);
e = rand;

cvx_begin
    variable x(n)
    minimize( norm( A * x - b, 2 ) )
    subject to
        C * x == d
        norm( x, Inf ) <= e
cvx_end
%%

