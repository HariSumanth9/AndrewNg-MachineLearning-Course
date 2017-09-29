function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X = [ones(m,1) X] ;         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
%% ================ Part 1: Compute Cost (Feedforward) ================
a2 = sigmoid(X*Theta1') ;
v = a2;
a2 = [ones(m,1) a2] ;
size(a2) ;
a3 = sigmoid(a2*(Theta2')) ;
size(a3) ;
k = num_labels;

for i = 1:k
    h = a3(:,i);     
      
        p = (y==i);
        
        J = J + ( -p'*log(h) - (1 - p)'*log(1 - h)) ;
end
J = J/m;
if lambda ==0
J = J;
else
J = J + sum(sum(Theta1(:,2:input_layer_size + 1).*(Theta1(:,2:input_layer_size + 1))))*lambda/(2*m) + ...
        sum(sum(Theta2(:,2:hidden_layer_size + 1).*(Theta2(:,2:hidden_layer_size + 1))))*lambda/(2*m) ;

end        
    
%% ================ Part 2: Back Prop ================


yt = [1:num_labels]==y ;
T1 = Theta1;
T2 = Theta2;
T1(:,1) =0;
T2(:,1) =0;
d3 = a3 - yt ;
d3 = d3' ;
d2 = Theta2'*d3.*(a2'.*(1-a2')) ;
d2 = d2(2:size(d2,1),:);   % doubt..
for i=1:m
    Theta2_grad = Theta2_grad + d3(:,i)*a2(i,:) ;
    Theta1_grad = Theta1_grad + d2(:,i)*X(i,:) ;
end
if lambda==0
    Theta2_grad = Theta2_grad/m ;
    Theta1_grad = Theta1_grad/m ;

else
    Theta2_grad = Theta2_grad/m + lambda*T2/m;
    Theta1_grad = Theta1_grad/m + lambda*T1/m;
    
end
% -------------------------------------------------------------

%=========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
