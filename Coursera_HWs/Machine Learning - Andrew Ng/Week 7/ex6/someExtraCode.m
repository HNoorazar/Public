% extra code for part 2 or 3 of HW:
load('ex6data3.mat')
v = [0.01 0.03 0.1 0.3 1 3 10 30]';
parameters = zeros(64,2);
minError = 1000;
BestC = 1000;
BestSigma = 1000;

predResult = zeros(length(v)^2,1);

for ii=1:length(v)
    for jj = 1:length(v)
        C = v(ii);
        sigma = v(jj);
        model = svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        pred = svmPredict(model, Xval);
        error = mean(double(pred ~= yval));
        if error < minError
            minError = error;
            BestC = C;
            BestSigma = sigma;
        end
    end
end
