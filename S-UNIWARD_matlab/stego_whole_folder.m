clc; clear all;

% set payload
PAYLOAD = single(0.4);
COVER_DIR = '../../../adversarial_attacks/white_box_attack/SRNet-trainandval-fromCover-0.4/val/0'; % Folder with the pgm cover images to be transformed
STEGO_DIR = '../../../adversarial_attacks/white_box_attack/SRNet-trainandval-fromCover-0.4/val/1';

% Create the output dir
mkdir(STEGO_DIR)

% load cover image
cover_filenames = dir(fullfile(COVER_DIR, '*.pgm'));

for index = 1:length(cover_filenames)
    cover_full_filename = fullfile(COVER_DIR, cover_filenames(index).name);

    stego = S_UNIWARD(cover_full_filename, PAYLOAD);
    fprintf('Cover filename: %s\n', cover_full_filename)

    stego = cast(stego,'uint8');
    imwrite(stego, fullfile(STEGO_DIR, cover_filenames(index).name))
end