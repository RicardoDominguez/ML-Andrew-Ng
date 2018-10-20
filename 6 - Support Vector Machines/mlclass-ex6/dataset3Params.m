function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
% C = 1;
% sigma = 0.3;

C_p =0.01;
sigma_p = 0.01;
error = 10.^5;
errors = zeros(64,1);
pairs = zeros(64,3);
% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%
values = [0.01 0.03 0.1 0.3 1 3 10 30];
coun = 1;

for c = 1:8
    for s = 1:8
        pairs(coun,:) = [sigma_p C_p coun];
        model= svmTrain(X, y, C_p, @(x1, x2) gaussianKernel(x1, x2, sigma_p));
        predictions = svmPredict(model, Xval);
        error_p = mean(double(predictions ~= yval));
        errors(coun,:) = error_p;
        if error_p < error
            error = error_p;
            C = C_p;
            sigma = sigma_p;
        end
        
        %sigma_p = sigma_p * 3;
        sigma_p = values(s);
        coun = coun + 1;
    end
    %C_p = C_p * 3;
    C_p = values(c);
end

pairs
errors
C
sigma





% =========================================================================

end
