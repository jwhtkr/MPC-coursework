function dint_inftynorm_unconstrained
% explicit MPC solution to double integrator with infinity norm using
% dynamic programming
%
% Example 10_1
% Francesco Borrelli July November 09 2009
% Revised by M. Herceg, Automatic Control Laboratory, ETH Zurich 2014
%
% requires MPT3

close all

%% printing parameters
label_font_size = 14;
tick_font_size = 10;
line_width = 0.8;
axeswidth=0.2;
figure_name = ['.',filesep,'figures',filesep,'fig_10_1'];

%% core code
% model dynamics
% x(k+1)=Ax(k)+Bu(k)
A = [1 1; 0 1];
B = [0; 1];

% no constraints on input and outputs

% specify cost function as a part of the model
Q = eye(2);
R = 20;

% dimensions
nu = 1; % number of inputs
nx = 2; % number of states


%  STEP N-1 Translation of problem :
%  min         eps
%     eps,epsu,epsx,u(N-1)  
%
%	s.t.    epsu+epsx			<= eps
%       	r*u(N-1)			<= epsu
%          -r*u(N-1)			<= epsu
%       	q*(x(N-1)+u(N-1)+1)	<= epsu
%          -q*(x(N-1)+u(N-1)+1)	<= epsu
%       	q*(x(N-1)+u(N-1)-1)	<= epsu
%          -q*(x(N-1)+u(N-1)-1)	<= epsu
%

% Result of solution of Step N-1 
%    Jopt=PWLf(x(N-1)) x(N-1).  Let nR the number of PWLfunctions

% Step N-2 Problem formulation:
%  min       |r*u(N-2)+|q*x(N-1)|+Jopt(x(N-1))  
%     u(N-2)  
%
% 7 - Translation of problem:
%  min         eps
%     eps,epsu,epsx,epsJopt,u(N-2)  
%
%	s.t.    epsu+epsx+epsJopt   <= eps
%  
%       "c1*x(N-1)=c1*(Ax(N-2)+Bu(N-2)) + d1 <= epsJopt"   
%                  c1*(Ax(N-2)+Bu(N-2)) + d2 <= epsJopt   
%                  c2*(Ax(N-2)+Bu(N-2)) + d3 <= epsJopt   
%                                :           <=   :
%                  cnR*(Ax(N-2)+Bu(N-2))     <= epsJopt   
%                 -c1*(Ax(N-2)+Bu(N-2)) - d1 <= epsJopt   
%                 -c2*(Ax(N-2)+Bu(N-2)) - d2 <= epsJopt
%                                :           <=   :
%                 -cnR*(x(N-1)+u(N-2)-1)     <= epsJopt   
%                            r*u(N-2)        <= epsu
%                           -r*u(N-2)	     <= epsu
%                    q*(x(N-2)+u(N-2)+1)     <= epsu
%                   -q*(x(N-2)+u(N-2)+1)     <= epsu
%                    q*(x(N-2)+u(N-2)-1)     <= epsx
%                   -q*(x(N-2)+u(N-2)-1)     <= epsx
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Algorithm Implementation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Inu=eye(nu);
xe=[0;0]; 
ue=0; 
maxiter=100;
%%%%%%%%%%%%%%%%%%%%%
%%% Intialization:
%%%%%%%%%%%%%%%%%%%
% Option 1 Pn=0
% nR=0;
% Option 2 Pn=Q
 nR=4;
 Joptl{1}=Q(1,:);
 Joptl{2}=-Q(1,:);
 Joptl{3}=Q(2,:);
 Joptl{4}=-Q(2,:);
 Joptc{1}=0;
 Joptc{2}=0;
 Joptc{3}=0;
 Joptc{4}=0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Algorithm Starts
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


notfound=1;
jj=0;
while notfound && (jj<maxiter)
    jj=jj+1;
    fprintf('************ STEP %d *************\n', jj);
    %At time j: Build cost function and constraint matrices
    % optvar=[ eps,epsJopt,epsuj,epsxj,uj]
    % f= epsJopt
    f=[1 0 0 0 zeros(1,nu)]';
    
    % to construct G optvar<=S+Wx
    % G=[- 1           	1				1             1               zeros(1,nu) ;
    %    zeros(2*nu,2) 					I1            zeros(2*nu,nu)  I2		  ;
    %    zeros(nR,1)  	Ij				zeros(nR,1)   zeros(nR,nu)    Ju          ;
    %    zeros(2*nu,2) 					zeros(2*nx,1) I3            	L 	      ;
    %    zeros(2*nu,2) 					zeros(o,2)     					Bc 		  ]
    % W=[0;zeros(2*nu,nx);Jx; W; -Ac];
    % S=[0;S1;Jc;S2;Cc];
    
    
    I1	=	-1*ones(2*nu,1);
    I2	=	[R;-R];
    S1	=	[-R*ue;R*ue];
    Ij	=	-1*ones(nR,1);
    if nR>0,
        Ju=[];
        Jx=[];
        Jc=[];
        for k=1:nR;
            Ju = [Ju; Joptl{k}*B];
            Jx = [Jx; -Joptl{k}*A];
            Jc = [Jc; -Joptc{k}];
        end
    else
        Ju=zeros(nR,nu);
        Jx=zeros(nR,nx);
        Jc=zeros(nR,1);
        Joptl=[];
    end
    
    
    I3 = -1*ones(2*nx,1);
    L  = [-Q*B;+Q*B];
    W  = [Q*A;-Q*A];
    S2 = [-Q*xe;Q*xe];
    
    %% CONSTRUCT NOW THE MATRICES%%%
    % optvar=[ eps,epsJopt,epsuj,epsxj,uj]
    % G optvar<=S+W xj+O wj
    
    G = [ -1           	1	            1             1               zeros(1,nu) ;
        zeros(2*nu,2) 					I1            zeros(2*nu,nu)  I2		  ;
        zeros(nR,1)  	Ij				zeros(nR,1)   zeros(nR,nu)    Ju          ;
        zeros(2*nx,2) 					zeros(2*nx,1) I3              L 	      ];
    W = [zeros(1,nx);zeros(2*nu,nx);Jx; W];
    S = [0;S1;Jc;S2];
    
    
    % Solve mp-LP
    % If its the first step and the cost to go is 0, remove the variable epsJopt
    if (jj==1) && (nR==0),
        G(:,2)=[];
        f(2)=[];
    end
    Matrices.f=f;
    Matrices.A=G;
    Matrices.b=S;
    Matrices.pB=W;
    Matrices.Ath=[1 0;-1 0;0 1; 0 -1];
    % Limits on exploration space, i.e. Ath*x <= bth
    Matrices.bth=[ 100;100;100;100];
    % formulate MPLP min f'x s.t A*x <= b + pB*th, Ath*x <= bth
    problem = Opt(Matrices);
    % solve MPLP
    solution = problem.solve;
    
    % number of regions
    nR = solution.xopt.Num;
    
    % Compute Joptf Joptc;
    Joptlnew=[];
    Joptcnew=[];
    for j=1:nR,
        % extract cost function from the solution for each region
        % cost = F*th + g
        Joptlnew{j}=solution.xopt.Set(j).Functions('obj').F;
        Joptcnew{j}=solution.xopt.Set(j).Functions('obj').g;
    end
    
    % Check if converged
    if length(Joptlnew)==length(Joptl),
        notequalJfound=0;
        kk=1;
        while (kk<=nR) && ~notequalJfound
            nl=0;
            ii=1;
            while (ii<=nR) && ~nl
                if (norm(Joptlnew{kk}-Joptl{ii},2)<1e-10) && (norm(Joptcnew{kk}-Joptc{ii},2)<1e-10),
                    nl=1;
                end
                ii=ii+1;
            end
            notequalJfound=~nl;
            kk=kk+1;
        end
    else
        notequalJfound=1;
    end
    
    notfound=notequalJfound;
    Joptl=Joptlnew;
    Joptc=Joptcnew;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% Algorithm Ends
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nR>0,
    Jx=[];
    for k=1:nR;
        Jx=[Jx;Joptl{k}];
    end
end

% plot the solution
figure
% sort regions according to radius of the Chebyshev ball
regions = solution.xopt.Set;
rd = regions.chebyCenter;
[srd,idsrt] = sort([rd.r]);
% plot according to radius
h1 = regions(idsrt).plot('colormap',gray,'LineWidth',line_width);
set(h1(end),'FaceColor',[0.9 0.9 0.9]);
hold on
% cost function
h2 = regions(idsrt).fplot('obj','colormap',gray,'LineWidth',line_width);

% use the same colors for PWA function as for the partition
c = linspace(0.1,0.9,20);
for i=1:numel(h2)
    cnew = [c(i) c(i) c(i)];
    set(h1(i),'FaceColor',cnew);
    set(h2(i),'FaceColor',cnew,'LineWidth',line_width);
end



%% save book figures
set(gca,'LineWidth',axeswidth)
set(gca,'FontSize', tick_font_size);
set(gca,'YTickLabel',{''})
set(gca,'XTickLabel',{''})
set(gca,'ZTickLabel',{''})

hx1 = xlabel('$x_1$');
set(hx1, 'FontSize', label_font_size);
hy1 = ylabel('$x_2$');
set(hy1, 'FontSize', label_font_size);
hz1 = zlabel('$J^{*}(x)$');
set(hz1, 'FontSize',label_font_size);
title('');

% generate FIG file
disp('Hit any key to print the figure and continue.');
pause
saveas(gcf, [figure_name,'_matlab'], 'fig');

% print 
laprint(gcf, [figure_name,'_tex'],'scalefonts','off');


end