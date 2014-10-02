function options = process_options(defaults, varargin)

options = defaults;
%process inputs
i = 1;
while(i < length(varargin))
    options.(varargin{i}) = varargin{i+1};
    i = i+2;
end

end