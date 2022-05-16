%% This code is applied to establish SVR prediction models for the improvement of clinical symptom of patients
%% The predicting features identified in SVC prediction is taken as the mask for the redundant features
clc; clear all;
NumROI = 35; % nodes of your functional brain network
path = 'your_pathway\matrix';
file = dir([path,filesep, '*.mat']);
load('FD_response_prediction_030.mat', 'cons_feature_mask_permut') % prediction results with the optimal threshold
conn_msk = double(sum(cons_feature_mask_permut ~= 0,3) ==100); % Identify predicting features in the SVC model
Ind_01 = find(triu(ones(NumROI),1));
Ind_02 = find(conn_msk(Ind_01) ~= 0);

% load the matrix of the patients
data_all = zeros(length(file), length(Ind_02));
for i = 1:length(file)
    load([path,filesep, file(i).name])
    data_all(i,:) = R(Ind_01(Ind_02)); 
end
clear i conn_msk file path R cons_feature_mask_permut;

% load the label
load('your_pathway\label.txt');

% establish SVC models 
permut = 100;
h = waitbar(0,'please wait..'); 
for mn = 1:permut
    waitbar(mn/permut,h,['repetition:',num2str(mn),'/',num2str(permut)]);
    predictive_value = zeros(size(data_all,1),1);
    k =10;
    indices = crossvalind('Kfold',size(data_all,1),k);  
    for  i = 1:k
         test = (indices == i); train = ~test;
         train_data = data_all(train,:);
         train_label = label(train,:);
         test_data = data_all(test,:);
         test_label = label(test,:); 
         cmd = ['-s 3, -t 0, -c 1'];
         model = svmtrain(train_label,train_data, cmd);% SVR linear kernal
         prediction = svmpredict(test_label,test_data,model);
         predictive_value(indices == i) = prediction;
         clear  test_data  train_data test_label train_label model
    end
    
    R = corr(predictive_value, label); % calculate R
    R2 = R*R;
    mse = sum((predictive_value - label).^2)/length(label);  % calculate mean square error

   %% Permutation test for R2 and mse 
    Nsloop = 100;
    R_permut = zeros(Nsloop,1);
    mse_permut = zeros(Nsloop,1);
    for i = 1:Nsloop
        randlabel = randperm(size(data_all,1));
        label_r  = label(randlabel);
        predictive_value_r = zeros(size(data_all,1),1);
        k =10;
        indices2 = crossvalind('Kfold',size(data_all,1),k);
        for m = 1:k
            test = (indices2 == m); train = ~test;
            train_data = data_all(train,:);
            train_label = label_r(train,:);
            test_data = data_all(test,:);
            test_label = label_r(test,:);
            model = svmtrain(train_label,train_data, '-s 3, -t 0, -c 1');% SVR linear kernal
            predicted_r = svmpredict(test_label,test_data,model); 
            predictive_value_r(indices2 == m) = predicted_r;
            clear  test_data  train_data test_label train_label model predicted_r
         end  
            R_permut(i,1) = corr(predictive_value_r, label_r);
            mse_permut(i,1) = sum((predictive_value_r - label_r).^2)/length(label_r);
            clear randlabel label_r predictive_value_r
    end

    p_predict_R = mean(abs(R_permut) > abs(R));
    p_predict_mse = mean(abs(mse_permut) < abs(mse));

    R2_permut(mn,1) = R*R;
    MSE_permut(mn,1) = mse;
    p_R_permut(mn,1) = p_predict_R;
    p_mse_permut(mn,1) = p_predict_mse;
    predictive_value_permut(mn,:) = predictive_value;
end
close(h);
    clear h mn Ind_01 Ind_02 cmd conn_msk cons_feature cons_feature_mask cons_feature_mean data_all i indices indices2 k mse_permut R_permut;
    clear label m mse NDSI_improve p p_predict_mse p_predict_R prediction predictive_value R SID_improve sigInd test train;

