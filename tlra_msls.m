function [Z,E,err ] = tlra_msls(X,A,max_iter,lambda ,Debug)

[n1,n2,n3]=size(X);% m n l
[~,n4,~]=size(A);% n

%% Z=J=Y1 n4 n2 n3
Z=zeros(n4,n2,n3);
J=Z;
Y1=Z;

%% E=Y2 n1 n2 n3
E = zeros(n1,n2,n3);
Y2=E;


beta = 1e-4;
max_beta = 1e+8;
tol = 1e-8;
rho = 1.1;
iter = 0;

Ain = t_inverse(A);
AT = tran(A);
while iter < max_iter
    iter = iter+1;
    
    %% update Zk
    Z_pre = Z;
    R1 = J-Y1/beta;
    [Z] = prox_tensor_pshrinkage(R1,1,-0.5);
    
    
    %% update Ek
    E_pre = E;
    R2=X-tprod(A,J)+Y2/beta;
    E = solve_l1l1l2( R2, lambda/beta );

    
    %% update Jk
    J_pre=J;
    Q1=Z+Y1/beta;
    Q2=X-E+Y2/beta;
    J=tprod(Ain, Q1+tprod(AT,Q2));

    
    %% check convergence
    leq1 = Z-J;
    leq2 = X-tprod(A,J)-E;
    leqm1 = max(abs(leq1(:)));
    leqm2 = max(abs(leq2(:)));
    
    difJ = max(abs(J(:)-J_pre(:)));
    difE = max(abs(E(:)-E_pre(:)));
    difZ = max(abs(Z(:)-Z_pre(:)));
    err(iter) = max([leqm1,leqm2,difJ,difZ,difE]);
    if (Debug && (iter==1 || mod(iter,20)==0))
        sparsity=length(find(E~=0));
    end
    if err < tol
        break;
        iter
    end
    
    %% update Lagrange multiplier and  penalty parameter beta
    Y1 = Y1 + beta*leq1;
    Y2 = Y2 + beta*leq2;
    beta = min(beta*rho,max_beta);
end
