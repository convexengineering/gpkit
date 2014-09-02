classdef mec < gp_constraint
    %monomial equality constraint
    %implies 1 == c*(vars.^a)
    
    %Woody Hoburg, November 2012, whoburg@alum.mit.edu
    
    properties
        %(defined in superclass)
    end
    
    methods
        function obj = mec(vars, c, a, name)
            if ischar(vars)
                %for single variable, allows {} to be omitted from input
                vars = {vars};
            end
            if nargin < 3
                %default exponents if not provided
                a = ones(1, length(vars));
            end
            if nargin < 2
                %default constant if not provided
                c = 1;
            end
            %check sizes
            [nt, nv] = size(a);
            assert(nt == 1);
            assert(nv == length(vars));
            assert(numel(c) == 1);
            %fill in obj fields
            obj.vars = vars;
            obj.a = a;
            obj.c = c;
            if nargin > 3
                obj.name = name;
            end
        end 
        
    end
    
end