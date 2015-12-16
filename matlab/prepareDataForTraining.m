
countZerosTest = arrayfun(@(x)sum(Y_test==x), 0);
countOnesTest = arrayfun(@(x)sum(Y_test==x), 1);
onesToAdd = countZerosTest-countOnesTest;

n = round(onesToAdd/countOnesTest);

% delete rows with invalid GroundTruth-Value
[row, ~] = find(Y_train==-1);
Y_train(row,:) = [];
X_train(row,:,:,:) = [];

[row, ~] = find(Y_test==-1);
Y_test(row,:) = [];
X_test(row,:,:,:) = [];


% even out positive and negative outcomes in test data
idx = (Y_test(:)==1);
Y_ones = Y_test(idx);
X_ones = X_test(idx,:,:,:);

for i=1:n
    X_test = [X_test; X_ones];
    Y_test = [Y_test; Y_ones];
end

countZerosTrain = arrayfun(@(x)sum(Y_train==x), 0)
countOnesTrain = arrayfun(@(x)sum(Y_train==x), 1)
countNegativesTrain = arrayfun(@(x)sum(Y_train==x), -1)
countZerosTest = arrayfun(@(x)sum(Y_test==x), 0)
countOnesTest = arrayfun(@(x)sum(Y_test==x), 1)
countNegativesTest = arrayfun(@(x)sum(Y_test==x), -1)

