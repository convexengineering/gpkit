function i = ind(str)
switch(str)
    case {'a', 'alpha', 'alfa'}
        i = 1;
    case {'cl', 'CL'}
        i = 2;
    case {'cd', 'CD'}
        i = 3;
    case 'CDp'
        i = 4;
    case {'cm', 'CM'}
        i = 5;
    case 'Top_Xtr'
        i = 6;
    case 'Bot_Xtr'
        i = 7;
    case {'re', 'Re', 'RE'}
        i = 8;
    case {'tau', 'thickness'}
        i =  9;
    case {'N', 'Ncrit'}
        i = 10;
    otherwise
        error(['Unrecognized input: ', str]);
end