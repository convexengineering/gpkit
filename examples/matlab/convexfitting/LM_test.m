function [params, RMStraj, X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT] = LM_test()

npt = 1000;
dimx = 2;
x = randn(npt,dimx);
y = sum(x.^2, 2);
%y = max(sum(x.^2, 2), 5*max(x,[],2));

K = 4;
ba = max_affine_init(x, y, K);
alphainit = 10;

rfun = @(p) generic_resid_fun(@softmax_affine, x, y, p);

%tic;
[params, RMStraj] = LM(rfun, [ba(:); alphainit]);
myLMtime = toc;
tic;
[X,RESNORM,RESIDUAL,EXITFLAG,OUTPUT] = lsqnonlin(rfun,[ba(:); alphainit],[],[],optimset('Jacobian', 'on'));
trusttime = toc;

disp('        finalRMS    iterations     time');
disp(['my LM   ', num2str(norm(rfun(params))/sqrt(npt)), '       ', num2str(length(RMStraj)), '        ', num2str(myLMtime)]);
disp(['trust   ', num2str(sqrt(RESNORM/npt)), '       ', num2str(OUTPUT.iterations), '        ', num2str(trusttime)]);

figure(1); clf(1);
plot(x(:,1), x(:,1).^2, '.'); hold on;
xplo = [linspace(min(x(:,1)), max(x(:,1)), 1000)', zeros(1000, dimx-1)];
plot(xplo(:,1), softmax_affine(xplo, params), 'r');

end