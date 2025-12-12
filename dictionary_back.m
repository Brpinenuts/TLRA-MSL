function [LL,V] = dictionary_back(X)
%% directly using raw data as dictionary
    tho=100;
    [ ~,~,U,V,S ] = prox_low_rank(X,tho);
    LL=tprod(U,S); 
