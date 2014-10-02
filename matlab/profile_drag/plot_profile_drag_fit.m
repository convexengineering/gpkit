function plot_profile_drag_fit()

d = load ('../NACA24xx_fits.mat');
data = d.NACA24xx;

%plotting setup
fs = 18;
fnum = 0;
leg = @(c,l) legend(c, 'fontsize', fs, 'location', l);
xlab = @(s) xlabel(s, 'fontsize', fs, 'fontweight', 'bold');
ylab = @(s) ylabel(s, 'fontsize', fs, 'fontweight', 'bold');
tit = @(s) title(s, 'fontsize', fs, 'fontweight', 'bold');
txt = @(x, y, s) text(x, y, s, 'fontsize', 14, 'fontweight', 'bold');
txtrot = @(x, y, s, r) text(x, y, s, 'fontsize', 14, 'fontweight', 'bold', 'rotation', r);
myplot = @(x,y,varargin) plot(x,y,varargin{:},'linewidth',2);
myloglog = @(x,y,varargin) loglog(x,y,varargin{:},'linewidth',2);
mysemilogy = @(x,y,varargin) semilogy(x,y,varargin{:},'linewidth',2);

    function savefig(name)
        dir = '$HOME/berkeley/research/aircraft_design/talks/mit_new_trends12/figs/';
        reldir = '../../figs/';
        print(gcf, '-dpsc', [reldir, name, '.eps']);
        unix(['epstopdf ', dir, name, '.eps']);
    end
fnum = 1;
    function newplot()
        figure(fnum); clf(fnum);
        fnum = fnum + 1;
        set(gca, 'fontsize', fs);
    end

%nk = size(residsave,1);
%bestresid = min(residsave(2:end, :, :), [], 2);
%avetime = mean(timesave(2:end, :, :), 2);
colors = 'brm';

figure(1); clf(1);
set(gca, 'fontsize', fs);
xlab('K');
ylab('RMS log error');
tit('Quality of Fit');
hold on;
myplot(data.Ks, data.maxaffine.resid, 'b');
myplot(data.Ks, data.softmax.resid, 'r');
myplot(data.Ks, data.implicit.resid, 'm');
myplot(data.Ks, data.maxaffine.resid, 'b.', 'markersize', 16);
myplot(data.Ks, data.softmax.resid, 'r.', 'markersize', 16);
myplot(data.Ks, data.implicit.resid, 'm.', 'markersize', 16);
%
% for i = 1:3
%     plot(2:nk, bestresid(:,:,i), [colors(i), '-+'], 'linewidth', 2, 'markersize', ms);
% end
leg({'max-affine', 'scaled softmax', 'implicit softmax'}, 'northeast');
savefig('profile_drag_fit_error');

% figure(2); clf(2);
% for i = 1:3
%     semilogy(2:nk, avetime(:,:,i), [colors(i), '-+'], 'linewidth', 2, 'markersize', ms);
%     hold on;
% end
% set(gca, 'fontsize', fs);
% xlab('K');
% ylab('average fitting time (s)');
% leg({'max-affine', 'scaled softmax', 'implicit softmax'}, 'southeast');
% 
% print(gcf, '-depsc', [reldir, 'profile_drag_fit_time', '.eps']);
% unix(['epstopdf ', dir, 'profile_drag_fit_time', '.eps']);

end