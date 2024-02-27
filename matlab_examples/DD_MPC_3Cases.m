function DD_MPC_3Cases
% tree cases of computing explicit solution for a double integrator example
% quadratic function
%
% This example uses native MPT3 interface to formulate the MPC problem.
%
% Francesco Borrelli July 05 2009
% M. Herceg, ETH Zurich, 2014
%
% requires MPT3

close all

%% printing parameters
label_font_size = 14;
tick_font_size = 10;
line_width = 0.8;
axeswidth=0.2;
% create new folder for storing figures
mkdir('figures');
fig_str = ['.',filesep,'figures',filesep,'double_int_case'];


%% core code
% use native mpt3 interface to formulate the problem
A = [1 1; 0 1];
B = [0; 1];
model = LTISystem('A',A,'B',B);
model.x.min = [-10;-10];
model.x.max = [ 10; 10];
model.u.min = -1;
model.u.max = 1;

% penalties
model.x.penalty = QuadFunction(eye(2));
model.u.penalty = QuadFunction(0.01);

% compute LQR terminal cost
PN = model.LQRPenalty;
model.x.with('terminalPenalty');
model.x.terminalPenalty = PN;

% Case 1
Tset = Polyhedron('Ae',eye(2),'be',zeros(2,1)); % equality constraints
model.x.with('terminalSet');
model.x.terminalSet = Tset;
% horizon = 2
ctrl1 = MPCController(model, 2);
% explicit solution
c{1} = ctrl1.toExplicit();
% simplify using merging
c{1}.simplify();


% Case 2
K = model.LQRGain();
X = Polyhedron([eye(2);-eye(2); K; -K],[model.x.max;-model.x.min;model.u.max; -model.u.min]);
aut_model = LTISystem('A',A+B*K);
Oinf = aut_model.invariantSet('X',X);
% use Oinf as terminal set
model.x.terminalSet = Oinf;
% horizon = 2
ctrl2 = MPCController(model, 2);
% explicit solution
c{2} = ctrl2.toExplicit();
% simplify using merging
c{2}.simplify()

% Case 3
model.x.without('terminalSet'); % remove terminal set constraint
% horizon = 6
ctrl3 = MPCController(model, 6);
% explicit solution
c{3} = ctrl3.toExplicit();
% simplify using merging
c{3}.simplify();

%% plot and save book figures
for ii = 1:3,
    figure;
    h=plot(c{ii}.optimizer,'colormap','lightgray','linewidth',line_width);
    axis([model.x.min(1),model.x.max(1),model.x.min(2),model.x.max(2)]);
    
    set(gca,'LineWidth',axeswidth);
    set(gca,'FontSize', tick_font_size);
    
    title('')
    hx=xlabel('$x_{1}$');
    set(hx, 'FontSize', label_font_size);
    hy=ylabel('$x_{2}$');
    set(hy, 'FontSize', label_font_size);
    
    set(gca,'XTickLabel',{''});
    set(gca,'YTickLabel',{''});
    
    % print
    figure_name = [fig_str, num2str(ii)];
    disp('Hit any key to print the figure and continue.');
    pause
    saveas(gcf, [figure_name,'_matlab'], 'fig');
    
    laprint(gcf, [figure_name,'_tex'],'scalefonts','off');   
end


end