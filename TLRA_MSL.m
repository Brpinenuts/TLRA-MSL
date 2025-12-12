close all;
clc;

addpath(genpath('./tSVD'));
addpath(genpath('./proximal_operator'));
%% Select datasets
datanames = 'HYDICE';
TIR = load([datanames,'.mat']);
DataTesttt = TIR.data;
DataTesttt = double(DataTesttt);
mask = double(TIR.map);

%% TLRA_MSL running
numb_dimension = 8; 
DataTestt = PCA_img(DataTesttt, numb_dimension);
[H,W,Dimm]=size(DataTestt);
num=H*W;
mask_reshape = reshape(mask, 1, num);
anomaly_map = logical(double(mask_reshape)>0);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
normal_map = logical(double(mask_reshape)==0);
for i=1:Dimm 
    DataTestt(:,:,i) = (DataTestt(:,:,i)-min(min(DataTestt(:,:,i)))) / (max(max(DataTestt(:,:,i))-min(min(DataTestt(:,:,i)))));
end 
X=DataTestt;  

tic;
[LL,V] = dictionary_back(X);
max_iter=100;
Debug = 0;
lambda=0.01; 
[~,A,~] = tlra_msls(X,LL,max_iter,lambda,Debug);
toc

%% TLRA_MSL Testing
E=reshape(A, num, Dimm)';
r_new=sqrt(sum(E.^2,1));

r_max = max(r_new(:));
taus = linspace(0, r_max, 5000);
PF4=zeros(1,5000);
PD4=zeros(1,5000);
for index2 = 1:length(taus)
  tau = taus(index2);
  anomaly_map_rx = (r_new> tau);
  PF4(index2) = sum(anomaly_map_rx & normal_map)/sum(normal_map);
  PD4(index2) = sum(anomaly_map_rx & anomaly_map)/sum(anomaly_map);
end
f_show=reshape(r_new,[H,W]);
f_show=(f_show-min(f_show(:)))/(max(f_show(:))-min(f_show(:)));
r_TLRA_MSL = reshape(f_show,num,1);
area_TLRA_MSL = sum((PF4(1:end-1)-PF4(2:end)).*(PD4(2:end)+PD4(1:end-1))/2);
NBOXPLOT(:,6)=f_show(:);
fprintf('TLRA_MSL = %.4f', area_TLRA_MSL);
