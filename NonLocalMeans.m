I = imread('cam.jpg');           % Read Image
Noisy = imnoise(I, 'gaussian');  % Add gaussian noise
NoisyDouble = double(Noisy);     % Convert Img into double for better precision

% Select neighborhood window and search window 
neighborhoodWindow = 5;
searchWindow = 11;
NW = floor(neighborhoodWindow/2);  % We need to divide it to find the range of the neighborhood window
SW = floor(searchWindow/2);        % Same as neighborhood window

% Vary the parameters to get different variation of filtering
h = 50;
sigma = 5;

denoisedImage = Non_Local_Means(NoisyDouble,h,sigma, NW,SW);

% Gaussian filteting
GaussianFiltering = imgaussfilt(Noisy, 2);

% Anisotropic Filteting
AnisotropicFiltering = imdiffusefilt(Noisy);

% Local neighborhood filtering
localMean = imboxfilt(Noisy, neighborhoodWindow);

% Uncomment and run to compare Non-local mean filters with different parameters
% Varying sigma and h values to get the optimal values for parameter
 % DN1 = Non_Local_Means(NoisyDouble,10,1, NW,SW);
 % DN2 = Non_Local_Means(NoisyDouble,20,2, NW,SW);
 % DN3 = Non_Local_Means(NoisyDouble,30,3, NW,SW); 
 % DN4 = Non_Local_Means(NoisyDouble,40,4, NW,SW);
 % DN5 = Non_Local_Means(NoisyDouble,50,5, NW,SW);
 % DN6 = Non_Local_Means(NoisyDouble,60,6, NW,SW);
 % figure; 
 % subplot(2,4,1), imshow(I), title('Original Image');
 % subplot(2,4,2), imshow(Noisy), title('Noisy Image');
 % subplot(2,4,3), imshow(DN1), title('Non Local Filter (h = 10, sigma = 1)');
 % subplot(2,4,4), imshow(DN2), title('Non Local Filter (h = 20, sigma = 2)');
 % subplot(2,4,5), imshow(DN3), title('Non Local Filter (h = 30, sigma = 3)');
 % subplot(2,4,6), imshow(DN4), title('Non Local Filter (h = 40, sigma = 4)');
 % subplot(2,4,7), imshow(DN5), title('Non Local Filter (h = 50, sigma = 5)');
 % subplot(2,4,8), imshow(DN6), title('Non Local Filter (h = 60, sigma = 6)');

% Comparing Non Local Means algorithm with other denoising methods
figure;
subplot(2,3,1), imshow(I), title('Original Image');
subplot(2,3,2), imshow(Noisy), title('Noisy Image');
subplot(2,3,3), imshow(GaussianFiltering), title('Gaussian Filtering');
subplot(2,3,4), imshow(AnisotropicFiltering), title('Anisotropic Filtering');
subplot(2,3,5), imshow(localMean), title('Box Filtering (Local Mean)');
subplot(2,3,6), imshow(denoisedImage), title('Non local means(h = 50, sigma = 5)');

% Non Local Means function for denoising
function denoisedImage = Non_Local_Means(Image, h, sigma, NW, SW)
  [M,N] = size(Image);  
  denoisedImage = zeros(M,N);
  NoisyImage = padarray(Image, [NW, NW], 'symmetric'); % Use padding for border pixel neighborhood
  gaussianKernel = getGaussianKernel(NW, sigma);
  
  % Loop through each pixel (row & col) in the image
  % Find indices of the search window [we will compare each pixel(Neighborhood)] 
  % in this window with the row and col pixel(Neighborhood)
  % We calculate the Non local Means for each pixel in the image and replace with original pixel intensity
  for row=1:M
      for col=1:N
          % Reset Non Local & Normalizing Constant for each pixel
          NonLocal = 0;
          Z = 0;
     
          % Adjust row and col for padding
          % For example: Neighboring window = 5, NW = 2
          % We need to add padding so we can take neighborhood of boundary pixels
          adjustRow = row+NW;
          adjustCol = col+NW;
          
          % Get Indices for search window
          windowIndices = getWindow(adjustRow, adjustCol, NW, SW, N, M);
          
          % Looping through the search window
          for searchRow=windowIndices(1):windowIndices(2)
              for searchCol=windowIndices(3):windowIndices(4)
                  % Get neighborhood for selected pixel(row, col) - we use the adjusted row & col due to padding
                  Indices = getNeighborhood(adjustRow, adjustCol, NW);
                  selectedNeighborhood = NoisyImage(Indices(1):Indices(2), Indices(3):Indices(4));
                  
                  % Get neighborhood for next pixel
                  Indices = getNeighborhood(searchRow, searchCol, NW);
                  nextNeighbordhood = NoisyImage(Indices(1):Indices(2), Indices(3):Indices(4));
                  
                  ED = sum(sum(gaussianKernel.*(selectedNeighborhood-nextNeighbordhood)^2));  % Calculating the Euclidian distance and multiply with gaussian kernel
                  weights = exp(-ED/(h^2));                                                   % Calculating the weights
                  NonLocal = NonLocal+(weights*NoisyImage(searchRow,searchCol));              % Calculate the NonLocalMean 
                  Z = Z+weights;                                                              % Calculate Normalizing constant
              end
          end
          denoisedImage(row,col) = NonLocal/Z; % Change pixel value
      end
  end
  denoisedImage = uint8(denoisedImage); % Change pixels it back to int
end

% Get neighborhood of pixel based on the window size
function Indices = getNeighborhood(row, col, window) 
  % To get the indices of the neighborhood we cover the pixel's(row, col) top, bottom, left, right
  % with the size of our window - that's how we get the neighboor for this particular (row,col)
  Indices = zeros(1,4);
  Indices(1) = row-window; % Start row
  Indices(2) = row+window; % End row
  Indices(3) = col-window; % Start col
  Indices(4) = col+window; % End col
end

% Get the indices for the search window: startrow, endrow, startcol & endcol
function windowIndices = getWindow(row, col, NW, SW, N, M) 
  % Place a SW window around our selected image pixel
  % Change the window indices whenever we change row,col 
  % To make sure we get a window size of the size of the selected window
  windowIndices = zeros(1,4);
  windowIndices(1) = max(row-SW, NW+1); % Start row
  windowIndices(2) = min(row+SW, NW+M); % End row
  windowIndices(3) = max(col-SW, NW+1); % Start col
  windowIndices(4) = min(col+SW, NW+N); % End col
end

% Get the Gaussian kernel (gives preference to centre of the kernel)
function gaussianKernel = getGaussianKernel(NW, sigma)
  range = -NW:NW; % Calculate range of values
  [xCoord, yCoord] = meshgrid(range, range); 
  
  gaussianKernel = exp( -(xCoord.^2 + yCoord.^2)/ (2*(sigma)^2)); % Calculating gaussian kernel
  gaussianKernel = gaussianKernel / sum(gaussianKernel(:));       % Normalizing the kernel
end
