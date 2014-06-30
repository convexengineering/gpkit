classdef mono < gp_constraint
    %monomial constraint
    %implies 1 >= c*(vars.^a)
    
    %Woody Hoburg, November 2012, whoburg@alum.mit.edu
    
    properties
        %(defined in superclass)
    end
    
    methods
        function obj = mono(vars, c, a, name, type)
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
            if nargin > 4
                obj.type = type;
            end
        end
        
%         function s = str(obj)
%             %convert to string
%             inds = 1:length(obj.a);
%             s = ' \geq ';
%             if min(obj.a) >= 0
%                 s = strcat('1', s);
%             else
%                 for i = inds(obj.a < 0)
%                     if obj.a(i) == -1
%                         ss = [obj.vars{i}, ' '];
%                     else
%                         ss = [obj.vars{i}, '^{', num2str(-obj.a(i)), '} '];
%                     end
%                     s = strcat(ss, s);
%                 end
%             end
%             if obj.c ~= 1
%                 s = strcat(s, [' ', num2str(obj.c)]);
%             end
%             for i = inds(obj.a > 0)
%                 if obj.a(i) == 1
%                     ss = [' ', obj.vars{i}];
%                 else
%                     ss = [' ', obj.vars{i}, '^{', num2str(obj.a(i)), '}'];
%                 end
%                 s = strcat(s, ss);
%             end
%         end
        
    end
    
end
