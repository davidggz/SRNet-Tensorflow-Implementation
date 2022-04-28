%           EXAMPLE - USING "S-UNIWARD" embedding distortion
%
% -------------------------------------------------------------------------
% Copyright (c) 2013 DDE Lab, Binghamton University, NY.
% All Rights Reserved.
% -------------------------------------------------------------------------
% Permission to use, copy, modify, and distribute this software for
% educational, research and non-profit purposes, without fee, and without a
% written agreement is hereby granted, provided that this copyright notice
% appears in all copies. The program is supplied "as is," without any
% accompanying services from DDE Lab. DDE Lab does not warrant the
% operation of the program will be uninterrupted or error-free. The
% end-user understands that the program was developed for research purposes
% and is advised not to rely exclusively on the program for any reason. In
% no event shall Binghamton University or DDE Lab be liable to any party
% for direct, indirect, special, incidental, or consequential damages,
% including lost profits, arising out of the use of this software. DDE Lab
% disclaims any warranties, and has no obligations to provide maintenance,
% support, updates, enhancements or modifications.
% -------------------------------------------------------------------------
% Author: Vojtech Holub
% -------------------------------------------------------------------------
% Contact: vojtech_holub@yahoo.com
%          fridrich@binghamton.edu
%          http://dde.binghamton.edu
% -------------------------------------------------------------------------
clc; clear all;

% load cover image
coverPath = fullfile('..', 'images_cover', '1.pgm');

% set payload
payload = single(0.4);

fprintf('Embedding using Matlab file');
MEXstart = tic;

%% Run default embedding
stego = S_UNIWARD(coverPath, payload);
        
MEXend = toc(MEXstart);
fprintf(' - DONE');

cover = imread(coverPath);
figure;
subplot(1, 2, 1); imshow(cover); title('cover');
subplot(1, 2, 2); imshow((double(stego) - double(cover) + 1)/2); title('embedding changes: +1 = white, -1 = black');
fprintf('\n\nImage embedded in %.2f seconds, change rate: %.4f', MEXend, sum(cover(:)~=stego(:))/numel(cover));