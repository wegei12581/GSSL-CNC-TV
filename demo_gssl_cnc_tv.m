%GSP_DEMO_GRAPH_TV Reconstruction of missing sample on a graph using TV
%
%   In this demo, we try to reconstruct missing sample of a piece-wise
%   smooth signal on a graph. To do so, we will minimize the well-known TV
%   norm defined on the graph.
%
%   For this example, you need the unlocbox. You can download it here:
%   http://unlocbox.sourceforge.net/download
%
%   We express the recovery problem as a convex optimization problem of the
%   following form:
%
%   ..   argmin   ||grad(x)||_1   s. t. Mx = y
%
%   .. math:: arg \min_x  \|\nabla(x)\|_1 \text{ s. t. } Mx = y
%  
%   Where b represents the known measurements, M is an operator
%   representing the mask and $\epsilon$ is the radius of the l2 ball.
%
%   We set 
%
%   * $f_1(x)=||\nabla x ||_1$
%     We define the prox of $f_1$ as: 
%
%     .. prox_{f1,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma  ||grad(z)||_1
%
%     .. math:: prox_{f1,\gamma} (z) = arg \min_{x} \frac{1}{2} \|x-z\|_2^2 +  \gamma \| \nabla z \|_1
%
%   * $f_2$ is the indicator function of the set S define by $Mx = y$
%     We define the prox of $f_2$ as 
%
%     .. prox_{f2,gamma} (z) = argmin_{x} 1/2 ||x-z||_2^2  +  gamma i_S( x ),
%
%     .. math:: prox_{f2,\gamma} (z) = arg \min_{x} \frac{1}{2} \|x-z\|_2^2   + i_S(x) ,
%
%     with $i_S(x)$ is zero if x is in the set S and infinity otherwise.
%     This previous problem has an identical solution as:
%
%     .. argmin_{x} ||x - z||_2^2   s.t.  Mx = y
%
%     .. math:: arg \min_{x} \|x - z\|_2^2   \hspace{1cm} such \hspace{0.25cm} that \hspace{1cm} Mx = y
%
%     It is simply a projection on the B2-ball.
%
%   Results
%   -------
%
%   .. figure::
%
%      Original signal on graph
%
%      This figure shows the original signal on graph.
%
%   .. figure::
%
%      Depleted signal on graph
%
%      This figure shows the signal on graph after the application of the
%      mask and addition of noise. Half of the vertices are set to 0.
%
%   .. figure::
%
%      Reconstructed signal on graph usign TV
%
%      This figure shows the reconstructed signal thanks to the algorithm.
%
%   Comparison with Tikhonov regularization
%   ---------------------------------------
%
%   We can also use the Tikhonov regularizer that will promote smoothness.
%   In this case, we solve:
%   
%   ..   argmin   ||grad(x)||_2^2   s. t. ||Mx-b||_2 < epsilon
%
%   .. math:: arg \min_x \tau \|\nabla(x)\|_2^2 \text{ s. t. } \|Mx-b\|_2 \leq \epsilon 
%
%   The result is presented in the following figure:
%
%   .. figure::
%
%      Reconstructed signal on graph using Tikhonov 
%
%      This figure shows the reconstructed signal thanks to the algorithm.
%
%   We can also use the CNC regularizer that will reduced the underestimation in l1 regularization.
%   In this case, we solve:
%   argmin argmin_{x} ||Mx - y||_2^2  + || x ||_1 - min_{v}{|| v ||_1 + 1/2||B(x-v)||_2^2}
%   

%   We solve this problem by writing our own gsp_regression_cnc and firm_thr functions,  .. 
%   where cnc passes one more parameter mu than tv, ..
%   the choice of mu is given in the paper.
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


% Let show the reconstructed graph
figure(3)
gsp_plot_signal(G,sol,paramplot)
caxis([-1 1])
title('Solution of the algorithm: TV')

% Let show the reconstructed graph
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
