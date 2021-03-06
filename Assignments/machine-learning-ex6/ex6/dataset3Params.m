function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 0.01 ;
sigma = 0.01 ;

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
accuracy = 0 ;

for i = 1:7
    for j = 1:7
      model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
      pred = svmPredict(model, Xval) ;
      acc_new  = sum(yval == pred) ;
      if acc_new > accuracy
          accuracy = acc_new ;
          C_acc = C ;
          sigma_acc = sigma ;
      end
      C = C*3 ;
    end
    C = 0.01 ;
    sigma = sigma*3 ;
   
end
C = C_acc ;
sigma = sigma_acc ;


% =========================================================================

end
