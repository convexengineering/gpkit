function s = compare_fits(xdata, ydata, Ks, ntry)

[npt, dimx] = size(xdata);
s = struct();
s.Ks = Ks;
alphainit = 10;

for ik = 1:length(Ks)
    for t = 1:ntry
        k = Ks(ik);
        
        bainit = max_affine_init(xdata, ydata, k);
        
        rfun = @(p) generic_resid_fun(@max_affine, xdata, ydata, p);
        tic;
        [params, RMStraj] = LM(rfun, [bainit(:)]);
        store_results('maxaffine');
        
        rfun = @(p) generic_resid_fun(@softmax_affine, xdata, ydata, p);
        tic;
        [params, RMStraj] = LM(rfun, [params(:); alphainit]);
        store_results('softmax_optMAinit');
        
        rfun = @(p) generic_resid_fun(@softmax_affine, xdata, ydata, p);
        tic;
        [params, RMStraj] = LM(rfun, [bainit(:); alphainit]);
        store_results('softmax_originit');

        rfun = @(p) generic_resid_fun(@implicit_softmax_affine, xdata, ydata, p);
        tic;
        [params, RMStraj] = LM(rfun, [bainit(:); alphainit*ones(k,1)]);
        store_results('implicit_originit');
        
    end
end


    function store_results(fieldname)
        s.(fieldname).resid(ik,t) = min(RMStraj);
        s.(fieldname).iter(ik,t) = length(RMStraj);
        s.(fieldname).params{ik,t} = params;
        s.(fieldname).time(ik,t) = toc;
    end


end