classdef gp < handle
    
    %convenience class for defining, modifying, and solving GPs
    %Woody Hoburg, November 2012, whoburg@alum.mit.edu
    
    %note: throughout this class definition, 'flight conditions' (fcs)
    %refer to a vector-valued variable, with a single name. its indices in
    %vars are determined using vari(name), which returns a vector for fcs
    %when a constraint is added involving one or more fcs, it is enforced
    %for all fcs (unless ow specified via optional params)
    
    properties
        
        %gp data (mosek format)
        c
        a
        map
        
        %constant hash map (values can be vectors)
        const
        
        %indexing
        vari
        fcs   %fc names for vector vars and consts
        vars
        constraints
        constraint_types
        
        tables %keep track of vector vars assigned to flight conditions (fcs)
        
        %counts
        nvar
        nmono
        nconstraint
    end
    
    methods
        function obj = gp(nm, nv)
            obj.a = sparse(nm, nv, min(5,nv)*nm);
            obj.c = zeros(nm, 1);
            obj.map = zeros(nm, 1);
            obj.const = containers.Map();
            obj.vari = containers.Map();
            obj.fcs = containers.Map();
            obj.vars = cell(1,nv);
            obj.constraints = cell(nm, 1);
            obj.constraint_types = cell(nm, 1);
            obj.tables = {};    %holds flight conditions and varnames
            obj.nvar = 0;
            obj.nmono = 0;
            obj.nconstraint = 0;
        end
        
        function trim(obj)
            %storage is overallocated on construction for faster runtime;
            %trims unused rows and columns accordning to counts
            %call this before attempting to call solve()
            obj.a(obj.nmono+1:end, :) = [];
            obj.a(:, obj.nvar+1:end) = [];
            obj.c(obj.nmono+1:end) = [];
            obj.map(obj.nmono+1:end) = [];
            obj.vars(obj.nvar+1:end) = [];
            obj.constraints(obj.nconstraint+1:end) = [];
            obj.constraint_types(obj.nconstraint+1:end) = [];
            assert(isempty(intersect(keys(obj.vari), keys(obj.const))));
        end
        
        function res = solve(obj, cmd)
            %solve the GP; returns mosek res struct
            %make sure we're trimmed
            assert(length(obj.c) == obj.nmono);
            assert(length(obj.map) == obj.nmono);
            assert(all(size(obj.a) == [obj.nmono, obj.nvar]));
            assert(any(obj.map == 0), 'no objective defined');
            
            %solve using mosek
            if(nargin == 1)
                cmd = 'minimize echo(0)'; %default -- supresses output
                %pass in cmd = 'minimize' to see mosek output
            end
            res = mskgpopt(obj.c, obj.a, obj.map, [], cmd);
            if res.rcode ~= 0
                %warning(['mosek returned code ', res.rcodestr]);
            end
        end
        
        function addvar(obj, varname, n)
            %add variable varname; n specifies vector length if not scalar
            assert(~any(varname=='@'), 'variable names should not include @ symbol');
            assert(~isKey(obj.vari, varname), ['variable ', varname, ' already exists']);
            assert(~isKey(obj.const,varname), ['variable ', varname, ' is a constant.']);
            if nargin == 2
                n = 1;
            end
            assert(n == 1 || isKey(obj.fcs, varname), ['vector var ', varname, ' is not listed in fcs']);
            ii = obj.nvar+1:obj.nvar+n;
            obj.vari(varname) = ii;
            obj.vars(ii) = {varname};
            obj.nvar = obj.nvar+n;
        end
        
        function addvars(obj, varnames, fcnames)
            % adds multiple variables of same length
            % length(fcnames) fixes vector length, or omit for scalar vars
            % also assigns vars to a cell in obj.tables
            % (i.e., makes this a set of flight conditions), 
            % and updates varfc to reflect fcnames
            if nargin == 3
                obj.tables{end+1} = {varnames, fcnames};
                n = length(fcnames);
                for i = 1:length(varnames)
                    var = varnames{i};
                    obj.fcs(var) = fcnames;
                    if isKey(obj.const, var)
                        % allows defn of vector consts w/ associated fcs;
                        % (const must be defined beforehand)
                        assert(length(obj.const(var)) == n);
                    else
                        obj.addvar(var, n);
                    end
                end
            else    % no fcnames specified
                % assume n is one -- scalar var
                for i = 1:length(varnames)
                    obj.addvar(varnames{i});
                end
            end
        end
        
        function add_constraint(obj, cc, fc, name)
            %adds the gp_constraint cc to this gp
            %optional fc specifies which fcs to add constraints for
            %default is to add constraints for all common fcs shared among
            %cc.vars
            %name is also optional -- overwrites cc.name
            assert(isa(cc, 'gp_constraint'));
            if ~(isa(cc, 'posy') || isa(cc, 'mono') || isa(cc, 'mec'))
                error(['unexpected constraint type ', class(cc)]);
                %doesn't currently handle max-monomial, because fillin_helper
                %requires a scalar map input
            end
            nt = size(cc.a,1);
            common_fcs = scanvars(obj, cc.vars);    %adds missing vars
            if nargin < 3
                fc = common_fcs; %default adds constraint to all common_fcs
            elseif ischar(fc)
                fc = {fc};
            end
            if isempty(fc)  %hack to make fc have length 1 in case of all scalar vars (no fcs shared)
                %in this case, an empty fc={} will get passed to fillin,
                %which is fine.
                fc = {fc};
            else
                assert(length(intersect(fc, common_fcs)) == length(fc));    %make sure all entries in fc are included in common_fcs
            end
            if nargin < 4 || isempty(name)
                if isempty(cc.name)
                    name = ['constraint ', num2str(obj.nconstraint + 1)];
                else
                    name = cc.name;
                end
            end
            if isa(cc, 'mec')
                cdirvals = 1:2;
                cdir_name_adder = {' (1> dir)', ' (1< dir)'};
            else
                cdirvals = 1;
                cdir_name_adder = {''};
            end
            
            for i = 1:length(fc)
                for cdir = cdirvals %this isn't a loop (only does 1 iter), unless cc is a mec.
                    ii = obj.nmono+1:obj.nmono+nt;
                    if cdir == 1
                        obj.fillin(ii, cc.c, cc.a, obj.nconstraint+1, cc.vars, fc{i});
                    elseif cdir == 2
                        obj.fillin(ii, 1/cc.c, -cc.a, obj.nconstraint+1, cc.vars, fc{i});
                    else
                        error(['unexpected cdir: ', num2str(cdir)]);
                    end
                    obj.nmono = obj.nmono + nt;
                    obj.nconstraint = obj.nconstraint + 1;
                    if isempty(fc{i})
                        obj.constraints{obj.nconstraint} = [name, cdir_name_adder{cdir}];
                    else
                        obj.constraints{obj.nconstraint} = [name, cdir_name_adder{cdir}, ' ', fc{i}];
                    end
                    obj.constraint_types{obj.nconstraint} = cc.type;
                end
            end
            
        end
        
        function add_objective_term(obj, cc, fc)
            nt = size(cc.a, 1);
            assert(isa(cc, 'gp_constraint'));
            common_fcs = scanvars(obj, cc.vars);    %adds missing vars
            ii = obj.nmono+1:obj.nmono+nt;
            if nargin == 2
                assert(isempty(common_fcs), 'found vector var(s), but objective fc not specified.');
                obj.fillin(ii, cc.c, cc.a, 0, cc.vars);
            else
                assert(ischar(fc));
                obj.fillin(ii, cc.c, cc.a, 0, cc.vars, fc);
            end
            obj.nmono = obj.nmono + nt;
        end
        
        function fillin(obj, rowi, c, a, map, vars, fc)
            %helper function for add_constraint and add_objective_term
            %fills in obj.{c,a,map} in rows rowi; does *not* update counts
            %rowi - the rows to fill in
            %c, a, map -- the constraint
            %vars - a cell array of variables
            %fc - (optional) fc at which to enforce constraint
            %if fc is not inputted, all vars must be scalars
            %this function also accepts var@fc forms
            
            [nt, nv] = size(a);
            assert(length(rowi) == nt);
            assert(length(c) == nt);
            assert(numel(map) == 1);
            assert(length(vars) == nv);
            
            %TODO handle fc not inputted case
            %also handle char case (map to {cell})
            
            if size(c,1) == 1
                c = c';
            end
            for iv = 1:nv
                var = vars{iv};
                %goals: set var, fci
                if any(var == '@')
                    ati = find(var == '@');
                    vpart = var(1:ati-1);
                    fpart = var(ati+1:end);
                    assert(isKey(obj.fcs, vpart));
                    fci = strcmp(obj.fcs(vpart), fpart);  %index into fcs, eg 1, 2, 3, ...
                    assert(any(fci), ['variable ', var, ' not found']);
                    var = vpart;
                elseif isKey(obj.fcs, var)
                    assert(~isempty(fc));
                    fci = strcmp(obj.fcs(var), fc);
                else
                    fci = [];
                end
                if isKey(obj.vari, var)
                    ia = obj.vari(var);
                    if length(ia) > 1
                        ia = ia(fci);
                        assert(numel(ia) == 1);
                    else
                        assert(isempty(fci));
                    end
                    obj.a(rowi, ia) = a(:,iv);
                elseif isKey(obj.const, var)
                    con = obj.const(var);
                    if length(con) > 1
                        con = con(fci);
                        assert(numel(con) == 1);
                    else
                        assert(isempty(fci));
                    end
                    c = c.*(con.^a(:,iv));
                else
                    error(['expected var ', var, ' to be in const or vari, but it wasn''t in either.']);
                end
            end
            obj.c(rowi) = c;
            obj.map(rowi) = map;
        end
        
        function common_fcs = scanvars(obj, vars)
            %scans cell array of vars;
            %returns fcs common to all vars
            % *adds missing vars*
            common_fcs = {};
            for i = 1:length(vars)
                var = vars{i};
                if isKey(obj.fcs, var)
                    assert(isKey(obj.vari, var) || isKey(obj.const, var));
                    if isempty(common_fcs)
                        common_fcs = obj.fcs(var);
                    else
                        common_fcs = intersect(common_fcs, obj.fcs(var));
                    end
                elseif isKey(obj.vari, var)
                    %scalar
                    assert(numel(obj.vari(var)) == 1);
                elseif isKey(obj.const, var)
                    %scalar
                    assert(numel(obj.const(var)) == 1);
                elseif any(var == '@')
                    %consider these guys scalars
                    continue;
                else
                    obj.addvar(var);
                end
            end
        end
        
        function print_report(obj, res, varargin)
            %somewhat hacky; prints a detailed report given the soln res
            defaults.print_sensitivities = true;
            options = process_options(defaults, varargin{:});
 
            vars_to_print = keys(obj.vari); %used to keep track of table vars vs other vars
            
            fprintf('\n');
            fprintf('SOLUTION REPORT\n');
            fprintf(['prosta: ', res.sol.itr.prosta, '\n']);
            fprintf(['solsta: ', res.sol.itr.solsta, '\n']);
            fprintf('\n');
            
            fprintf('CONSTANTS\n');
            cnames = keys(obj.const);
            mnl = max(cellfun(@length, cnames));    %max name length
            for i = 1:length(cnames)
                cnst = obj.const(cnames{i});
                fstr = ['%-', num2str(mnl+4), 's', repmat(' %-10.4g', 1, length(cnst)), '\n'];
                fprintf(fstr, cnames{i}, cnst);
            end
            fprintf('\n');
            
            if options.print_sensitivities
                fprintf('CONSTANT SENSITIVITIES\n');
                [s, si] = sort(-res.sol.itr.y*100, 'descend');
                for i = 1:obj.nconstraint
                    %disp([num2str(res.sol.itr.y(i)*100), '%   ', obj.constraints{i}]);
                    %continue here if constraint should not be printed
                    if strcmp(obj.constraint_types{si(i)}, 'const')
                        fprintf('%6.2f%%  %s\n', s(i), obj.constraints{si(i)});
                    end
                end
                fprintf('\n');
            end
            
            %print tables -- vars-by-fc version
            %to do -- print fc consts in table instead of consts list?
            for i = 1:length(obj.tables)
                vs = obj.tables{i}{1};
                fc = obj.tables{i}{2};
                vals = zeros(length(vs), length(fc));
                for vi = 1:length(vs)
                    if isKey(obj.vari, vs{vi})
                        vals(vi,:) = exp(res.sol.itr.xx(obj.vari(vs{vi})));
                    elseif isKey(obj.const, vs{vi})
                        vals(vi,:) = obj.const(vs{vi});
                    else
                        error(['variable ', vs{vi}, ' not found.']);
                    end
                    vars_to_print(strcmp(vars_to_print, vs{vi})) = [];
                end
                %print_table(vals', fc, vs);
                print_table(vals, vs, fc);
                fprintf('\n');
            end
            
            fprintf('VARIABLES\n');
            for i = 1:length(vars_to_print)
                v = vars_to_print{i};
                vi = obj.vari(v);
                fstr = ['%-10s', repmat(' %-12.4g', 1, length(vi)), '\n'];
                fprintf(fstr, v, exp(res.sol.itr.xx(vi)));
            end
            fprintf('\n');
           
            if options.print_sensitivities 
            fprintf('CONSTRAINT SENSITIVITIES\n');
            [s, si] = sort(-res.sol.itr.y*100, 'descend');
            for i = 1:obj.nconstraint
                %disp([num2str(res.sol.itr.y(i)*100), '%   ', obj.constraints{i}]);
            %continue here if constraint should not be printed
                if any(strcmp(obj.constraint_types{si(i)}, {'const','nodisplay'}))
                    continue;
                end
                fprintf('%6.2f%%  %s\n', s(i), obj.constraints{si(i)});
            end
            end
            
        end
        
        function [sols, objective] = sweep_pareto(obj, constraint_id, fc, u)
            %constraint_id may either be a number (index into map), or name
            %fc allows indexing into multi-fc constraints
            %u is relaxation param as defined in gp_tutorial
            %3 options for u:
            % a) unspecified, in which case default 0.8 is used
            % b) scalar, e.g. .95, in which case the c for constraint_name
            %    is swept from currentC/u to currentC*u
            % c) vector of u's to divide 
            if ischar(constraint_id)
                constraint_id = find(strcmp(obj.constraints, constraint_id));
            end
            if numel(constraint_id) > 1 && nargin > 2
                constraint_id = constraint_id(fc);
            end
            assert(numel(constraint_id) == 1);
            assert(constraint_id <= obj.nconstraint);
            rows = find(obj.map == constraint_id);
            objrows = find(obj.map == 0);
            cold = obj.c(rows);
            if nargin < 4
                u = 0.8;
            end
            if numel(u) == 1
                if u > 1, u = 1/u; end
                u = logspace(log10(u), log10(1/u), 31); 
            end            
            sols = nan*ones(obj.nvar, length(u));
            objective = nan*ones(1,length(u));
            for i = 1:length(u)
                obj.c(rows) = cold/u(i);
                res = obj.solve();
                xx = res.sol.itr.xx;
                if strcmp(res.sol.itr.prosta, 'PRIMAL_AND_DUAL_FEASIBLE')
                    sols(:,i) = exp(xx);
                    objective(i) = sum(obj.c(objrows).*exp(obj.a(objrows,:)*xx));
                end
            end
            obj.c(rows) = cold;
            
            figure(1); clf(1);
            if numel(rows) == 1 && nnz(obj.a(rows,:)) == 1 && abs(obj.a(rows,find(obj.a(rows,:)))) == 1
                ai = find(obj.a(rows,:));
                aval = obj.a(rows,ai);
                varname = obj.vars{ai};
                assert(numel(cold) == 1);
                if aval == -1
                    xvals = cold./u;
                elseif aval == 1
                    xvals = u/cold;
                else
                    error(['unexpected aval = ', num2str(aval)]);
                end
                plot(xvals, objective, 'linewidth', 2);
                hold on;
                plot(xvals, objective, '.', 'markersize', 20);
                title([obj.constraints{constraint_id}, ' Pareto frontier']);
                xlabel(varname);
                ylabel('objective');
            else
                plot(u, objective, 'linewidth', 2);
                hold on;
                plot(u, objective, '.', 'markersize', 20);
                title([obj.constraints{constraint_id}, ' Pareto frontier']);
                xlabel('u (loosening parameter)');
                ylabel('objective');
            end
            
        end
        
    end
end

