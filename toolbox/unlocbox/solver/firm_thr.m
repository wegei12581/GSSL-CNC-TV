 function p = firm_thr(x,lambda, mu)
%function p = prox_abs(x, gamma)
%
% This procedure computes the proximity operator of the function
%
%                       f(x) = gamma * |x|
%
% When the input 'x' is an array, the output 'p' is computed element-wise.
%
%  INPUTS
% ========
%  x     - ND array
%  gamma - positive, scalar or ND array with the same size as 'x'

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Version : 1.0 (27-04-2017)
% Author  : Giovanni Chierchia
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2017
%
% This file is part of the codes provided at http://proximity-operator.net
%
% By downloading and/or using any of these files, you implicitly agree to 
% all the terms of the license CeCill-B (available online).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% check input
if any( lambda(:) <= 0 ) || ~isscalar(lambda) && any(size(lambda) ~= size(x))
    error('''lambda'' must be positive and either scalar or the same size as ''x''')
end

if any( mu(:) <= 0 )||  (mu(:) <= lambda(:) ) || ~isscalar(lambda) && any(size(lambda) ~= size(x))
    error('''mu'' must be positive, greater than ''lambda''and either scalar or the same size as ''x''')
end
%-----%

p=0.*(abs(x)<= lambda)+(sign(x).* mu.*(abs(x)-lambda)./(mu-lambda)).*(abs(x) > lambda & abs(x) < mu)+x.*(abs(x) >= mu);


% if abs(x) <= lambda 
%     p = 0;
% elseif  abs(x) > lambda & abs(x) < mu
%     p = sign(x)*mu*(abs(x)-lambda)/(mu-lambda);
% else 
%     p = x;
% end