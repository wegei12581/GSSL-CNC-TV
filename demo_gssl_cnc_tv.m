%   In this demo, We utilize the function to randomly generate a block graph, 
%   and then utilize the TV, Tikhonov, and CNC regularizer respectively to process the sample graph based on the generated graph,
%   with the aim of restoring the graph information, and then compare the results obtained by different models.
%
%   For this example, you need the unlocbox. You can download it here:
%   http://unlocbox.sourceforge.net/download
%
%  We can use the TV regularizer，the most classic regularizer to restore the graph signal value.
%  In this case, we solve:
%
%  argmin argmin_{x} ||Mx - y||_2^2  + epsilon * || x ||_{GTV}
%  i.e. argmin argmin_{x} ||Mx - y||_2^2  + epsilon * || x ||_{1}
%
%  We can also use the Tikhonov regularizer that will promote smoothness.
%  In this case, we solve:
%   
%  argmin   ||grad(x)||_2^2   s. t. ||Mx-b||_2 < epsilon
% 
%
%   
%   We can also use the CNC regularizer that will reduced the underestimation in l1 regularization.
%   In this case, we solve:
%
%   argmin argmin_{x} ||Mx - y||_2^2  + epsilon * {|| x ||_1 - min_{v}{|| v ||_1 + 1/2||B(x-v)||_2^2}}
%   
%
%
%



%% Initialisation

clear;
close all;

% Loading toolbox
init_unlocbox();

verbose = 1;    % verbosity level
sigma = 0.0;

N = 240; % size of the graph for the demo

rng(2024);



%% Create a random sensor graph

paramgraph.distribute = 1;
G = gsp_stochastic_block_graph(N,9);

G = gsp_adj2vec(G);
G = gsp_estimate_lmax(G);
G = gsp_compute_fourier_basis(G);

graph_value = sign(G.U(:,4));


%%
p = 0.8; %probability of having no label on a vertex.
%create the mask
M = rand(G.N,1);
M = M>p;


%applying the Mask to the datao
depleted_graph_value = M.*(graph_value+sigma*randn(G.N,1));

sol = gsp_regression_tv(G,M,depleted_graph_value,0.03);
sol2 = gsp_regression_tik(G,M,depleted_graph_value,0.03);
sol3 = gsp_regression_cnc(G,M,depleted_graph_value,0.03,2.25);

sol4 = sol - graph_value;
sol5 = sol2 - graph_value;
sol6 = sol3 - graph_value;

%% Print the result
paramplot.show_edges = 1;

% Let show the original graph
figure(1)
gsp_plot_signal(G,graph_value,paramplot)
caxis([-1 1])
title('Original signal')


% Let show depleted graph
figure(2)
gsp_plot_signal(G,depleted_graph_value,paramplot)
caxis([-1 1])
title('Measurement')



figure(3)
gsp_plot_signal(G,sol,paramplot)
caxis([-1 1])
title('Solution of the algorithm: TV')


figure(4)
gsp_plot_signal(G,sol2,paramplot)
caxis([-1 1])
title('Solution of the algorithm: Tikhonov')

figure(5)
gsp_plot_signal(G,sol3,paramplot)
caxis([-1 1])
title('Solution of the algorithm: CNC')

figure(6)
gsp_plot_signal(G,sol4,paramplot)
caxis([-1 1])
title('TV error')

figure(7)
gsp_plot_signal(G,sol5,paramplot)
caxis([-1 1])
title('Tikhonov error')

figure(8)
gsp_plot_signal(G,sol6,paramplot)
caxis([-1 1])
title('CNC error')
