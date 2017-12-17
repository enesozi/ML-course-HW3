cla;
%%======================================================
%% STEP 1: Training Part
% Load data and sort wrt to class
data = load('points2d.dat');
X = sortrows(data,3);
[N,D] = size(X);
X = X(:,1:(D-1));
color = {'b' 'r' 'g'};
% First, create a [10,000 x 2] matrix 'gridX' of coordinates representing
% the input values over the grid.
gridSize = 100;
u = linspace(-10, 10, gridSize);
[A,B] = meshgrid(u, u);
gridX = [A(:), B(:)];
muCells = cell(3,3);
sigmaCells = cell(3,3);
% For each class calculate em for training part.
for class = 0:2
    first  = class*N/3+1;
    last  = (class+1)*N/3-N/6;
    figure(class+1)
    axis([-10 10 -10 10]);
    M = X(first:last,:);
    % Calculate for each cluster.
    for k = 1:3
     % Get mean and covariance for each cluster and keep it.   
     [mu,sigma,idx,r] = GMM(M,k); 
     muCells{class+1,k} = mu;
     sigmaCells{class+1,k} = sigma;
     subplot(3,1,k);
     hold on;
        for j = 1:k
         %classProbs{class+1,k} = mean(r);
         scatter(M(idx == j, 1), M(idx == j, 2), color{j});
         set(gcf,'color','white') % White background for the figure.
         % Calculate the Gaussian response for every value in the grid.
         z = gaussianProb(gridX, mu(j,:), sigma{j});
         % Reshape the responses back into a 2D grid to be plotted with contour.
         Z = reshape(z, gridSize, gridSize);
         % Plot the contour lines to show the pdf over the data.
         [~, ~] = contour(u, u, Z);
        end
     t = sprintf('Original Data and Expected PDFs class=%d & k = %d',class,k);
     title(t);
    end
end

%% Validation part
km = zeros(3,3);
I = [1001 1500;3001 3500;5001 5500];
% For each class calculate which k is the best.
for t = 1:3
    % Class guessed
    cc = 0;
    % Gaussian number guessed
    kk = 0;
    correct = 0;
    for i = I(t,1):I(t,2)
        % Probability of belonging to some cluster.
        prob = -1;
        for class = 0:2
            for k = 1:3
                s = sigmaCells{class+1,k};
                m = muCells{class+1,k}; 
                % For each cluster of all classes
                % Calculates and compares the prob.
                % If better prob. encountered then class and clustter
                % number is updated.
                for j = 1:k
                    p = gaussianProb(X(i,:), m(j,:), s{j});
                    if p>prob
                        prob = p;
                        cc = class;
                        kk = k;
                    end
                    %fprintf('class %d k %d j %d : %f \n',(class+1),k,j,gaussianND(X(1500,:), m(j,:), s{j}));
                end
            end
        end
        % Correct guess made then k is good to consider.
        if t == cc+1
            km(t,kk) = km(t,kk)+1;
        end
    end
end
[maxVals,maxKs] = max(km,[],2);
kSums = sum(km,2);
fprintf('\n');
for i=1:3
fprintf('Best k for class %d:, %d ratio(%%): %.2f \n',i-1,maxKs(i),100*maxVals(i)/kSums(i));
end
%% Test and error prediction part
% Only one matrix to show confusion amount.
% (1,1) --> Actual class1 and Guessed class1 etc.
confusion = zeros(3,3);
I = [1501 2000;3501 4000;5501 6000];
% For each class calculate which prediction error.
for t = 1:3
    cc = 0;
    correct = 0;
    for i = I(t,1):I(t,2)
        prob = -1;
        for class = 0:2         
                % Cluster 3 seems to best from the validation part.
                s = sigmaCells{class+1,maxKs(class+1)};
                m = muCells{class+1,maxKs(class+1)};
                % For the cluster 3 calculates and compares prob.
                % If prob is better class no guessed is updated.
                for j = 1:maxKs(class+1)
                    p = gaussianProb(X(i,:), m(j,:), s{j});
                    if p>prob
                        prob = p;
                        cc = class;
                    end
                end
         end
        % To fill confusion matrix.
        confusion(t,cc+1) = confusion(t,cc+1)+1; 
    end
end
fprintf('\n');
disp('Prediction Error(%) for each class respectively');
100-diag(confusion)./5
disp('Conusion matrix for all class:');
confusion
%% KNN Part
trainingData = data(1:3000,:);
testData = data(3001:end,:);
Ks = [1 10 40];
confusionKnn = zeros(3,2);
cl = zeros(3000,1);
for k=1:3
    cl = KNN(trainingData,Ks(k),testData);
    s = sum(cl==data(3001:end,3));
    confusionKnn(k,1) = s;
    confusionKnn(k,2) = 3000-s;
end
[x,y] = max(confusionKnn(:,1));
fprintf ('Best k: %d \n',Ks(y));
disp('Prediction Error(%) for each k: ');
confusionKnn(:,2)./30
disp('Confusion matrix for all k: ');
confusionKnn
