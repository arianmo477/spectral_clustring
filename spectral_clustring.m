clear all;
close all;
clc;

fprintf('Welcome to the Spectral Clustering Application!\n');

% Prompt user to select a dataset
datasetChoice = menu('Select Dataset', 'Circle.csv', 'Spiral.csv', 'Sphere.csv');
if datasetChoice == 1
    fileName = 'Circle.csv';
    numDimensions = 2;
    k = input('Enter the value of k for k-nearest neighbors (e.g., 10, 20, 40): ');
elseif datasetChoice == 2
    fileName = 'Spiral.csv';
    numDimensions = 2;
    k = input('Enter the value of k for k-nearest neighbors (e.g., 10, 20, 40): ');
elseif datasetChoice == 3
    fileName = 'Sphere.csv';
    numDimensions = 3;
    k = input('Enter the value of k for k-nearest neighbors (e.g., 10, 20, 40): ');
else
    error('Invalid choice! Please restart and select a valid dataset.');
end

laplacianChoice = menu('Select L or nomalized L', 'L', 'L_norm');

% Load the dataset
data = readmatrix(fileName);
X = data(:, 1:numDimensions); % Extract coordinates
if datasetChoice == 2
    true_clusters = data(:, 3); % Extract true labels for Spiral dataset
end

% Parameters
sigma = 1; 
numEigen = input('Enter the number of eigenvalues to compute (e.g., 3): ');

% Step 1: Construct the similarity graph and adjacency matrix W
fprintf('Constructing k-nearest neighborhood graph with k = %d...\n', k);
W = computeKNNGraph(X, k, sigma);

% Plot the sparsity pattern of the adjacency matrix W
figure;
spy(W); 
title(sprintf('Sparsity Pattern of the Adjacency Matrix (k = %d)', k));
xlabel('Column Index');
ylabel('Row Index');

% Step 2: Compute the degree matrix D and Laplacian matrices
[D, L] = computeLaplacians(W, laplacianChoice);

% Step 3 & 4: Compute the number of connected components & Compute the smallest eigenvalues and eigenvectors
fprintf('Computing the number of connected components...\n');
sigma = 1e-5; % Small positive shift
[eigvecs, eigvals] = eigs(L, numEigen, sigma);
eigvals = diag(eigvals); % Extract eigenvalues as a vector

if laplacianChoice == 1
    tolerance = 1e-6; 
else
    tolerance = 1e-3; 
end
num_connected_components = sum(abs(eigvals) < tolerance);
fprintf('Number of connected components: %d\n', num_connected_components);

% Step 4: Compute the smallest eigenvalues and eigenvectors

% optional part
%{
maxIter = 1000; 
relTol = 1e-6; 
[eigvals, eigvecs] = inversePowerMethod(L_norm, numEigen, maxIter, relTol);
%}


% Use eigenvalues to determine the number of clusters
M = chooseClusters(eigvals, k, laplacianChoice);
fprintf('Number of clusters chosen using eigengap heuristic: %d\n', M);

% Step 5: Extract eigenvectors for clustering
U = eigvecs(:, 1:M);

% Step 6: Perform k-means Clustering
clusters_kmeans = kmeans(U, M, 'Replicates', 10);

% Step 7: Assign original points to clusters
clusters_X = cell(M, 1); % Initialize cell array to hold clusters
for i = 1:M
    clusters_X{i} = X(clusters_kmeans == i, :); % Assign points in cluster i
end

% Display the points in each cluster for verification
for i = 1:M
    fprintf('Cluster %d contains %d points.\n', i, size(clusters_X{i}, 1));
end

% Step 8: Plot the kmeans Clustering results
figure;
if numDimensions == 2
    gscatter(X(:, 1), X(:, 2), clusters_kmeans);
    title(sprintf('%s Dataset - kmeans Clustering (k = %d)', fileName, k));
    xlabel('X1');
    ylabel('X2');
elseif numDimensions == 3
    scatter3(X(:, 1), X(:, 2), X(:, 3), 50, clusters_kmeans, 'filled');
    title(sprintf('%s Dataset - kmeans Clustering (k = %d)', fileName, k));
    xlabel('X1');
    ylabel('X2');
    zlabel('X3');
    grid on;
end

% Step 9: Compute clusters using other methods
% DBSCAN Clustering


minPts = k;    % Minimum number of points to form a dense region
epsilon = findBestEpsilon(U, minPts);
clusters_dbscan = dbscan(U, epsilon, minPts);
figure;
if numDimensions == 2
    gscatter(X(:, 1), X(:, 2), clusters_dbscan);
elseif numDimensions == 3
    scatter3(X(:, 1), X(:, 2), X(:, 3), 50, clusters_dbscan, 'filled');
end
title(sprintf('%s Dataset - DBSCAN Clustering', fileName));
xlabel('X1');
ylabel('X2');
if numDimensions == 3
    zlabel('X3');
end

% Hierarchical Clustering
Z = linkage(U, 'ward');

distances = Z(:, 3);  % Distances are in the third column of the linkage matrix
delta_distances = diff(distances);  % Calculate differences between consecutive distances
[~, elbow_index] = max(delta_distances);  % Find the index of the largest change
cutoff = distances(elbow_index);  % Set cutoff at the elbow

clusters_hier = cluster(Z, 'criterion', 'distance', 'cutoff', cutoff);  
figure;
if numDimensions == 2
    gscatter(X(:, 1), X(:, 2), clusters_hier);
elseif numDimensions == 3
    scatter3(X(:, 1), X(:, 2), X(:, 3), 50, clusters_hier, 'filled');
end
title(sprintf('%s Dataset - Hierarchical Clustering', fileName));
xlabel('X1');
ylabel('X2');
if numDimensions == 3
    zlabel('X3');
end

% Evaluate silhouette score for clustrings
if datasetChoice ~= 2
    silhouette_kmeans = mean(silhouette(X, clusters_kmeans));
    fprintf('Silhouette Score for kmeans Clustering: %.3f\n', silhouette_kmeans);
    
    silhouette_hier = mean(silhouette(X, clusters_hier));
    fprintf('Silhouette Score for Hierarchical Clustering: %.3f\n', silhouette_hier);
    
    silhouette_dbscan = mean(silhouette(X, clusters_dbscan));
    fprintf('Silhouette Score for DBSCAN Clustering: %.3f\n', silhouette_dbscan);
end


% Evaluate performance (if true labels are available for Spiral dataset)
if datasetChoice == 2
    ARI_kmeans = adjustedRandIndex(true_clusters, clusters_kmeans);
    ARI_dbscan = adjustedRandIndex(true_clusters, clusters_dbscan);
    ARI_hier = adjustedRandIndex(true_clusters, clusters_hier);
    fprintf('Adjusted Rand Index (ARI) for kmeans Clustering: %.3f\n', ARI_kmeans);
    fprintf('Adjusted Rand Index (ARI) for dbscan: %.3f\n', ARI_dbscan);
    fprintf('Adjusted Rand Index (ARI) for Hierarchical Clustering: %.3f\n', ARI_hier);
end


% Functions
function W = computeKNNGraph(X, k, sigma)
    N = size(X, 1);
    W = sparse(N, N);
    for i = 1:N
        distances = sum((X - X(i, :)).^2, 2);
        [~, idx] = mink(distances, k+1);
        for j = 2:k+1
            W(i, idx(j)) = exp(-distances(idx(j)) / (2 * sigma^2));
            W(idx(j), i) = W(i, idx(j));
        end
    end
end

function [D, L] = computeLaplacians(W, laplacianChoice)
    D = diag(sum(W, 2));
    
    if laplacianChoice == 2
        D_inv_sqrt = spdiags(1 ./ sqrt(diag(D)), 0, size(D, 1), size(D, 2));
        L = speye(size(D)) - D_inv_sqrt * W * D_inv_sqrt;
    else
        L = D - W;
    end
    
end

function M = chooseClusters(eigvals, k, Lchoice)
    % Sort the eigenvalues in ascending order
    eigvals = sort(eigvals);
    
    
    % Plot the eigenvalues
    figure;
    plot(eigvals, '-o', 'LineWidth', 1.5);
    xlabel('Index');
    ylabel('Eigenvalue');
    if Lchoice == 2
        title(sprintf('Eigenvalues of normalized Laplacian (k = %d)', k));
    else
        title(sprintf('Eigenvalues of Laplacian (k = %d)', k));
    end
    grid on;
    M = input("choose the number of clusters: ");
    
end

function ARI = adjustedRandIndex(trueLabels, predictedLabels)
    contingencyMatrix = confusionmat(trueLabels, predictedLabels);
    sumRows = sum(contingencyMatrix, 2);
    sumCols = sum(contingencyMatrix, 1);
    total = sum(contingencyMatrix(:));
    sumComb = sum(sum(contingencyMatrix .* (contingencyMatrix - 1))) / 2;
    sumRowsComb = sum(sumRows .* (sumRows - 1)) / 2;
    sumColsComb = sum(sumCols .* (sumCols - 1)) / 2;
    expectedIndex = (sumRowsComb * sumColsComb) / (total * (total - 1) / 2);
    maxIndex = (sumRowsComb + sumColsComb) / 2;
    ARI = (sumComb - expectedIndex) / (maxIndex - expectedIndex);
end

function bestEpsilon = findBestEpsilon(U, minPts)
    % U: Data matrix (rows are points, columns are features)
    % minPts: Minimum number of points to form a dense region
    
    % Compute pairwise distances
    [N, ~] = size(U);
    k = minPts - 1; % k for k-th nearest neighbor
    distances = zeros(N, N);
    
    for i = 1:N
        distances(i, :) = sqrt(sum((U - U(i, :)).^2, 2));
    end
    
    % Extract the k-th nearest neighbor distance for each point
    kDistances = zeros(N, 1);
    for i = 1:N
        sortedDistances = sort(distances(i, :), 'ascend');
        kDistances(i) = sortedDistances(k + 1); % k-th nearest neighbor distance
    end
    
    % Sort and plot the k-distance graph
    sortedKDistances = sort(kDistances, 'ascend');

    kDistanceDiff = diff(sortedKDistances);  % Difference between consecutive distances
    rateOfChange = diff(kDistanceDiff);      % Second derivative (rate of change)
    
    % Find the index of the largest rate of change
    [~, elbowIndex] = max(rateOfChange);
    
    % The best epsilon is the k-distance corresponding to the elbow point
    bestEpsilon = sortedKDistances(elbowIndex + 1); % +1 because diff reduces the length by 1
    
    % Plot the k-distance graph with the elbow point
    figure;
    plot(sortedKDistances, '-o', 'LineWidth', 1.5);
    hold on;
    plot(elbowIndex + 1, bestEpsilon, 'ro', 'MarkerFaceColor', 'r');
    xlabel('Point Index (Sorted)');
    ylabel(sprintf('%d-th Nearest Neighbor Distance', k));
    title('k-Distance Graph with Elbow Detection');
    grid on;
    legend('k-Distances', 'Detected Elbow');
end


% optional part
%{
function [eigvals, eigvecs] = inversePowerMethod(A, numEigen, maxIter, relTol)
    
    
    n = size(A, 1);
    eigvals = zeros(numEigen, 1);
    eigvecs = zeros(n, numEigen);
    
    
    shift = 1e-5;
    
    for i = 1:numEigen
        
        v = randn(n, 1);
        v = v / norm(v, 2);
        
        
        lambda = 0; 
        for k = 1:maxIter
            
            u = (A - shift * eye(n)) \ v;
            
            
            u = u / norm(u, 2);
            
            
            lambda_new = u' * A * u;
            
            
            if abs(lambda_new - lambda) < relTol
                break;
            end
            lambda = lambda_new;
            v = u;
        end
        
        
        eigvals(i) = lambda;
        eigvecs(:, i) = u;
        
        
        A = A - lambda * (u * u');
    end
end
%}
