function [data, stalldata] = import_polars(path, relim)%, old_data)

     function savefig(name)
         dir = '$HOME/berkeley/research/aircraft_design/papers/aiaa_journal/figs/';
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

%defaults
if(nargin < 2 || isempty(relim))
    relim = inf*[-1 1];
elseif(length(relim) == 1)
    relim = [relim relim];
end
fs = 12;
xlab = @(s) xlabel(s, 'fontsize', fs);
ylab = @(s) ylabel(s, 'fontsize', fs);
zlab = @(s) zlabel(s, 'fontsize', fs);
tit = @(s) title(s, 'fontsize', fs);

%first pass through, get info from file names
files = dir(path);
nfiles = length(files);
fnameres = zeros(nfiles, 1);
fnametaus = zeros(nfiles, 1);
arftypes = cell(nfiles, 1);
arfcodes = cell(nfiles, 1);
for i = 1:nfiles
    fnameinfo = regexp(files(i).name, '(?<arftype>[a-zA-Z]+)(?<arfcode>\d+)_Re(?<rek>\d+)k', 'names');
    %fields of above are arftype, arfcode, rek
    fnameres(i) = str2double(fnameinfo.rek)*1000;
    fnametaus(i) = str2double(fnameinfo.arfcode(end-1:end));
    arftypes{i} = fnameinfo.arftype;
    arfcodes{i} = fnameinfo.arfcode;
%     if(fnameres(i) < relim(1) || fnameres(i) > relim(2))
%         filteredfiles(i) = [];  %remove file from consideration if outside re bounds
%     end
end
%filteredfiles(fnameres < relim(1) | fnameres > relim(2)) = [];
keepi = fnameres >= relim(1) & fnameres <= relim(2);
files = files(keepi);
nfiles = length(files);
arftypes = arftypes(keepi);
arfcodes = arfcodes(keepi);
fnameres = fnameres(keepi);
fnametaus = fnametaus(keepi);
uniqueres = unique(fnameres);
uniquetaus = unique(fnametaus)/100;
rei = containers.Map(uniqueres(1), 1);
for i = 2:length(uniqueres)
    rei(uniqueres(i)) = i;
end
taui = containers.Map(uniquetaus(1), 1);
for i = 2:length(uniquetaus)
    taui(uniquetaus(i)) = i;
end
m = colormap;
mgray = colormap('gray');
reclr = @(r)interp1(logspace(log10(min(uniqueres)),log10(max(uniqueres)),size(m,1))', m, r);
reclrgray = @(r).75*(1-interp1(logspace(log10(min(uniqueres)),log10(max(uniqueres)),size(m,1))', mgray, r));

slashpos = find(path == '/', 1, 'last');
if(isempty(slashpos))
    pathstr = '';
else
    pathstr = path(1:slashpos);
end
data.data = [];
% if(nargin < 3)
%     data.data = [];
% else
%     data.data = old_data;
% end

%go through the data again, now opening each file
%figure to plot polar traces
%figure(1); clf(1);
clslices = [.2 .5 .8 1.2];
for i = 1:nfiles
    d = importdata([pathstr, files(i).name], ' ', 12);
    arf = regexp(d.textdata{4}, 'Calculated polar for: (?<type>[a-zA-Z]+) ?(?<code>\d+)', 'names');
    if(~strcmp(arf.type, arftypes{i}) || ~strcmp(arf.code, arfcodes{i}))
        error('airfoil mismatch between filename and file header');
    end
    headerdata = regexp(d.textdata{9}, 'Re = *(?<restr>\d+.\d+ e \d) +Ncrit = *(?<Ncrit>\d+.\d+)', 'names');
    headerdata.re = str2double(headerdata.restr(headerdata.restr ~= ' '));
    headerdata.Ncrit = str2double(headerdata.Ncrit);
    if(headerdata.re ~= fnameres(i))
        disp(files(i).name)
        disp(headerdata.re);
        disp(fnameres(i));
        error('Re mismatch between filename and file header');
    else
        re = headerdata.re;
        tau = fnametaus(i)/100;
    end
    if(re < relim(1) || re > relim(2))
        error('file outside relim, this should not happen');
    end
    if(~strmatch(arf.type, 'NACA'))
        error('expected NACA airfoil');
    end
    if(length(fnameinfo.arfcode) ~= 4)
        error('expected 4 digit airfoil code');
    end
    cl = d.data(:, xfoil_polar_ind('cl'));
    [maxcl, maxcli] = max(cl);
    %mini = find(cl > 0.1, 1, 'first');
    ii = cl >= .1 & cl < .95*maxcl & (1:length(cl))' <= maxcli;
    d.data = d.data(ii, :);
    if size(d.data,1) == 0
        disp([files(i).name, ' has no useful data -- skipping']);
        continue;
    end
    cd = d.data(:, xfoil_polar_ind('cd'));
    cl = d.data(:, xfoil_polar_ind('cl'));
    clstall(taui(tau), rei(re)) = maxcl;
    o = ones(size(d.data,1),1);
    data.data = [data.data; [d.data, re*o, tau*o, headerdata.Ncrit*o]];
    %[~, sorti] = sort(cl);
    badi = find(cl ~= [unique(cl); zeros(length(cl)-length(unique(cl)),1)], 1, 'first');
    if(~isempty(badi))
        cl = cl(1:badi(1)-1);
        cd = cd(1:badi(1)-1);
    end
    if(any(unique(cl) ~= cl))
        error('multiple cls');
    end
    if length(cl) > 5
        for c = 1:length(clslices)
            cdatclslice(taui(tau), rei(re), c) = interp1(cl, cd, clslices(c));
        end
    end
end

data.a = data.data(:, xfoil_polar_ind('a'));
data.cl = data.data(:, xfoil_polar_ind('cl'));
data.cd = data.data(:, xfoil_polar_ind('cd'));
data.tau = data.data(:, xfoil_polar_ind('tau'));
data.re = data.data(:, xfoil_polar_ind('re'));

tauplots = [.08 .10 .12 .16];
newplot();
for i = 1:length(tauplots)
    subplot(2,2,i)
    tauplot = tauplots(i);
    for re = uniqueres'
        inds = (data.re == re) & (data.tau == tauplot);
        loglog(data.cl(inds), data.cd(inds), '.', 'color', reclr(re));
        hold on;
        loglog(data.cl(inds), data.cd(inds), 'color', reclr(re));
    end
    set(gca, 'fontsize', fs);
    xlab('c_l');
    ylab('c_d');
    tit(['\tau = ', num2str(tauplot)]);
    axis([min(data.cl) max(data.cl) min(data.cd) max(data.cd)]);
    set(gca, 'xtick', [.2, .4:.4:2]);
    set(gca, 'ytick', [.005, .01:.01:.05]);
end
savefig('cdp_slicecl_color');

newplot();
for i = 1:length(tauplots)
    subplot(2,2,i)
    tauplot = tauplots(i);
    for re = uniqueres'
        inds = (data.re == re) & (data.tau == tauplot);
        loglog(data.cl(inds), data.cd(inds), '.', 'color', reclrgray(re));
        hold on;
        loglog(data.cl(inds), data.cd(inds), 'color', reclrgray(re));
    end
    set(gca, 'fontsize', fs);
    xlab('c_l');
    ylab('c_d');
    tit(['\tau = ', num2str(tauplot)]);
    axis([min(data.cl) max(data.cl) min(data.cd) max(data.cd)]);
    set(gca, 'xtick', [.2, .4:.4:2]);
    set(gca, 'ytick', [.005, .01:.01:.05]);
end
savefig('cdp_slicecl_gray');

newplot();
for i = 1:length(clslices)
    subplot(2,2,i);
    %[TAU, RE] = meshgrid(uniquetaus, uniqueres);
    %mesh(TAU, RE, cdatclslice');
    for re = uniqueres'
        loglog(uniquetaus, cdatclslice(:, rei(re), i), '.', 'color', reclr(re));
        hold on;
        loglog(uniquetaus, cdatclslice(:, rei(re), i), 'color', reclr(re));
    end
    axis([min(uniquetaus) max(uniquetaus) min(data.cd) max(data.cd)]);
    xlab('\tau');
    ylab('c_d')
    tit(['c_l = ', num2str(clslices(i))]);
    set(gca, 'fontsize', fs);
    set(gca, 'xtick', min(uniquetaus):.02:max(uniquetaus));
    set(gca, 'ytick', [.005, .01:.01:.05]);
end
%savefig('cdp_slicetau');

newplot();
for i = 1:length(clslices)
    subplot(2,2,i);
    for tau = uniquetaus'
        loglog(uniqueres, cdatclslice(taui(tau), :, i), '.');
        hold on;
        loglog(uniqueres, cdatclslice(taui(tau), :, i));
    end
    axis([min(uniqueres) max(uniqueres) min(data.cd) max(data.cd)]);
    xlab('Re');
    ylab('c_d')
    set(gca, 'fontsize', fs);
    tit(['c_l = ', num2str(clslices(i))]);
    set(gca, 'ytick', [.005, .01:.01:.05]);
end
%savefig('cdp_slicere');

% figure(4); clf(4);
% for tau = uniquetaus'
%     loglog(uniqueres, 1./clstall(taui(tau), :), '.');
%     hold on;
%     loglog(uniqueres, 1./clstall(taui(tau), :));
% end
% xlab('Re');
% ylab('1/C_{L_{stall}}');

% figure(5); clf(5);
% for re = uniqueres'
%     loglog(uniquetaus, 1./clstall(:, rei(re)), '.', 'color', reclr(re));
%     hold on;
%     loglog(uniquetaus, 1./clstall(:, rei(re)), 'color', reclr(re));
% end
% xlab('1/tau');
% ylab('1/C_{L_{stall}}');

[tt,rr] = meshgrid(uniquetaus, uniqueres);
clstalls = zeros(size(tt));
for i = 1:numel(tt)
    clstalls(i) = clstall(taui(tt(i)), rei(rr(i)));
end
stalldata.tau = tt(:);
stalldata.re = rr(:);
stalldata.clstall = clstalls(:);

newplot();
mesh(log10(tt), log10(rr), log10(1./clstalls));
xlabel('log10(tau)', 'fontsize', 16);
ylabel('log10(Re)', 'fontsize', 16);
zlabel('log10(1/cl_{stall})', 'fontsize', 16);
view([-144 15]);
%savefig('clstall');

% close all;
% clc;

%figure(2); %clf(2);
% for i = 1:length(files)

%
%     %figure(1);
%     %hold on;
%     %plot(alpha, CL, '*', 'color', reclr(re));
%     %figure(2); hold on;
%     %plot(CL, CD, '*', 'color', reclr(re));
% end
%
%
% figure(1); clf(1);
% scatter(data.a, data.cl, [20], log10(data.re), 'filled');
% colorbar; set(gca, 'fontsize', 14);
% hold on;
% plot(data.a, 2*pi*(data.a+2.2)*pi/180, 'k');
% xlabel('alpha [deg]', 'fontsize', 14);
% ylabel('CL', 'fontsize', 14);
% title(path, 'fontsize', 14);
% figure(2); clf(2);
% scatter(data.cl, data.cd, [20], log10(data.re), 'filled');
% colorbar; set(gca, 'fontsize', 14);
% xlabel('CL', 'fontsize', 14);
% ylabel('CD', 'fontsize', 14);
% title(path, 'fontsize', 14);
% figure(3); clf(3);
% scatter(log10(data.cl(data.cl > 0)), log10(data.cd(data.cl > 0)), [20], log10(data.re(data.cl > 0)), 'filled');
% colorbar; set(gca, 'fontsize', 14);
% xlabel('log10(CL)', 'fontsize', 14);
% ylabel('log10(CD)', 'fontsize', 14);
% title(path, 'fontsize', 14);
%
% figure(4); clf(4);
% scatter(log10(data.tau(data.cl > 0)), log10(data.cd(data.cl > 0)), [20], (data.cl(data.cl > 0)), 'filled');
% colorbar; set(gca, 'fontsize', 14);
% xlabel('log10(tau)', 'fontsize', 14);
% ylabel('log10(CD)', 'fontsize', 14);
% title(path, 'fontsize', 14);
%
% figure(5); clf(5);
% scatter3(log10(data.cl), log10(data.tau), log10(data.cd));
%
% figure(6); clf(6);
% scatter3(log10(data.cl), log10(data.re), log10(data.cd));

end
