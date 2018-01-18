function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%   NNCOSTFUNCTION Implements the neural network cost function for a two layer
%   neural network which performs classification
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

%==============================================================
% Part 1, Cost Function

X  = [ones(m,1),X];
K = num_labels;

for ii=1:m
    yy = zeros(K,1);
    target = y(ii);
    yy(target) = 1;
    
    a1 = X(ii,:);
    z2 = Theta1*a1'; a2 = sigmoid(z2); a2=[1;a2];
    
    z3=Theta2*a2; a3 = sigmoid(z3);
    L1 = log(a3);       L2 = log(1-a3);
    T1 = yy.*L1;        T2 = (1-yy).*L2;
    TpT= T1+T2;
    Jtemp =  sum(TpT);
    J = J+Jtemp;
end
J = -1/m*J;

%==============================================================
% Part 1, Regularized Cost Function

alpha = lambda/(2*m);

[t1r,t1c]=size(Theta1); [t2r,t2c]=size(Theta2);
reg1=0; reg2=0;
for jj=1:t1r
    for kk=2:t1c
        reg1 = reg1+Theta1(jj,kk)^2;
    end
end

for jj=1:t2r
    for kk=2:t2c
        reg2 = reg2 + Theta2(jj,kk)^2;
    end
end
J=alpha*(reg1+reg2)+J;

%==========================================
%==== Part 2
Delta1=zeros(t1r,t1c);
Delta2=zeros(K,t1r+1);
for ii=1:m
    yy = zeros(K,1);
    target = y(ii);
    yy(target) = 1;
    a1 = X(ii,:);
    z2 = Theta1*a1'; a2 = sigmoid(z2); a2=[1;a2];
    z3=Theta2*a2; a3 = sigmoid(z3);
    delta3 = a3-yy;
    aa =sigmoidGradient(z2);
    
    delta2 = Theta2(:,2:end)'*delta3.* aa;
    
    Delta1=Delta1+delta2*a1;
    Delta2=Delta2+delta3*a2';
end

Theta1_grad = 1/m*Delta1;
Theta2_grad = 1/m*Delta2;
beta = lambda/m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end)+beta*Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end)+beta*Theta2(:,2:end);


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
