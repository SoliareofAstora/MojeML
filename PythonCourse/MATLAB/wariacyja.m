a = [2,3,10,5,12];
B = [1 1 1 1;1 1 1 1; 1 1 1 1];
xd =-5

cov(B)
arrayfun(@(b)(b-xd),B(:,:))


%wariacja(a)



function var = wariacja(x)
xd = sum(x)./length(x);
var = sum(arrayfun(@(a)(a-xd).^2,x))./(length(x)-1);
end

%function cov = covariance(x)
