function print_table(tab, rowhdr, colhdr)
%prints a formatted table with row and column labels
[nr, nc] = size(tab);
assert(length(rowhdr) == nr);
assert(length(colhdr) == nc);
rowhdrlen = max(cellfun(@length, rowhdr));
fstr = [repmat(' ', 1, rowhdrlen), ' |', repmat(' %-10s', 1, nc), '\n'];
fprintf(fstr, colhdr{:});
fprintf([repmat('-', 1, rowhdrlen), '-|-', repmat('-', 1, nc*11), '\n']);
for i = 1:nr
    fstr = ['%-', num2str(rowhdrlen), 's |', repmat(' %-10.4g', 1, nc), '\n'];
    fprintf(fstr, rowhdr{i}, tab(i,:));
end
end