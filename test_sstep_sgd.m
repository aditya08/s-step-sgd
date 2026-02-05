% Ensure libsvm is installed and on the MATLAB path.
% Download libsvm from: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
% After downloading, add the MATLAB interface folder (e.g., libsvm-*/matlab) to your MATLAB path:
%   addpath('path/to/libsvm/matlab')
% Optionally, compile the mex files if needed by running:
%   make
% in the libsvm/matlab directory (on systems with a supported compiler).
% Once added to the path, the function libsvmread will be available.
% Instructions to download the w1a dataset (from the LIBSVM Data repository):
% 1) Visit the LIBSVM datasets page:
%    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
% 2) Search for "w1a" or navigate to the "Binary classification" section where w1a is listed.
% 3) Download the file named "w1a" (it is usually provided as a plain text file).
%    Direct link (may change over time): 
%    https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html
%    or look for a link labelled "w1a" on that page.
% 4) Save the downloaded file into your project directory. In this script we expect the file
%    to be located at ../caml/data/w1a.txt relative to the script location.
%    - Create the directories if they do not exist:
%        mkdir -p ../caml/data
%    - Move or copy the downloaded file into that folder and (optionally) rename it to w1a.txt.
% 5) The file format is the standard LIBSVM sparse format: each line is
%       <label> index1:value1 index2:value2 ...
%    libsvmread handles this format directly, so no further preprocessing is required.
% 6) If you prefer to download programmatically from MATLAB, you can use websave:
%    url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w1a';
%    outpath = fullfile('<path>','w1a.txt');
%    websave(outpath, url);
%    Note: the exact URL may differ; if websave fails, download manually via a browser.
% 7) After placing w1a.txt in <path>, the subsequent libsvmread call will load it.
% Note: update path below
[y, A] = libsvmread('<path>/w1a.txt');
[m, n] = size(A);
A = full(A);
x = zeros(n, 1);
% adjust labels if they are {1,2}^m instead of {-1, 1}^m
% y = (y == 2) - ~(y == 2);
% Training hyperparameters 
options.maxiters = 15360;
options.s = 1;
options.print_interval=512;
options.batch_size = 16;
options.eta = .5;  % learning rate
s_max = 256;
% pad matrix and labels to be a multiple of s_max*batch_size
if mod(m, options.batch_size*s_max) ~= 0
    extra_rows = mod(options.batch_size*s_max - m, options.batch_size*s_max);
    A = [A; zeros(extra_rows, n)];
    y = [y; zeros(extra_rows, 1)];
    [m, n] = size(A);
end

[x_s1, objval, accuracy] = sgd(A, x, y, options);



options.s = 16;
[x_s16, objval_s16, accuracy_s16] = sstep_sgd(A, x, y, options);

options.s = 256;
[x_s256, objval_s256, accuracy_s256] = sstep_sgd(A, x, y, options);

%% Plot convergence curve
clear fig
% save w1a-sstep-comparison.mat
fig = figure();
hold on;
xticks = 0:options.print_interval:options.maxiters;
plot(xticks,objval, LineStyle="-", Marker="o", MarkerSize=16, LineWidth=2)
plot(xticks, objval_s16, LineStyle="-.", Marker="x", MarkerSize=14, LineWidth=2)
plot(xticks, objval_s256, LineStyle=":", Marker="d", MarkerSize=10, LineWidth=2)
legend('SGD', 'SGD, s = 16', 'SGD, s = 256')
xlabel('Iterations (k)')
ylabel('Objective Function')
fontsize(fig, 20, "points")
% exportgraphics(fig, 'w1a-sstep-comparison.pdf','Resolution', 300)
