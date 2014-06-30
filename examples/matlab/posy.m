classdef posy < gp_constraint
    
    %posynomial constraint class
    %Woody Hoburg, November 2012, whoburg@alum.mit.edu
    
    properties
        %(defined in superclass)
    end
    
    methods
        
        function obj = posy(vars, c, a, lhs, name, type)
            %implies lhs > sum_i c_i*(vars.^a(i,:))
            %if entered, lhs must be a monomial
            
%             if ischar(vars)
%                 %for single variable
%                 %actually, this should never happen, since we assert
%                 % nt > 1 -- keeping code just in case, but ok to delete
%                 % entire commented out section if things seem to be working
%                 vars = {vars};
%             end
%             %don't allow default c or a, since #terms ambiguoius, and
%             %doesn't add significant convenience
            
            %check sizes
            [nt, nv] = size(a);
            assert(length(c) == nt);
            assert(length(vars) == nv);
            assert(nt > 1);
            
            %handle lhs if entered
            if nargin > 3 && ~isempty(lhs)
                %check that lhs is a monomial
                assert(strcmp(class(lhs), 'mono'));
                augvars = unique([vars, lhs.vars]); %add in any extra lhs vars
                %if #vars increased, we need to reshape a
                if length(augvars) > nv
                    auga = zeros(nt, length(augvars));
                    for i = 1:nv
                        auga(:,strcmp(augvars, vars{i})) = a(:,i);
                    end
                    a = auga;
                    vars = augvars;
                end
                %now divide a and c through by lhs
                c = c/lhs.c;
                for i = 1:length(lhs.vars)
                    iv = strcmp(vars, lhs.vars{i});
                    a(:,iv) = a(:,iv) - lhs.a(:,i);
                end
            end
            
            %fill in obj fields
            obj.vars = vars;
            obj.c = c;
            obj.a = a;
            if nargin > 4
                obj.name = name;
            end
	    if nargin > 5
		obj.type = type;
	    end
        end
        
%         function obj = breakdown(vars)
%             %don't think this ever worked -- ok to delete
%             obj = posy(vars, [1 1], ones(2, length(vars)));
%         end
        
    end
    
end

