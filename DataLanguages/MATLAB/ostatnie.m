f([-4 1 3]);

B = [-5 8 1; 4 0 8]
solve(B)
c = [-2 0 ; 1 3]
solve(c)

%https://www.mathworks.com/help/matlab/matlab_prog/find-array-elements-that-meet-a-condition.html
function res = solve(x)
res = x;
res(res<0)=-1;
%istotne jest ¿eby ten warunke wstawiæ przed ostatnim warunkiem!
res(res>=1)=2;
res(res>=0 & res<1)=1;
end

%czy x którejest macierz¹ jest mniejsze od zera??
%sprawdza tylko ostatni element. 
function y = f(x)
if x < 0
   y = -1;
elseif x < 1
   y = 1;
else
   y = 2;
end
end

