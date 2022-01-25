clear;
clc;

prefix = '/media/user/Data/odrive/grad/traffic_signal/data/UCF_CC_50/';

files = importdata('images.list');
num = size(files, 1);

csv = fopen('metadata.csv', 'w');
fprintf(csv, 'file_name,height,width,channel,count\n');

for i=1:num

    img_file = char(strcat(prefix, 'images/', files(i)));

    base_name = split(files(i), '.');
    parts = split(base_name(1), '_');
%     mat_file = char(strcat(prefix, 'ground_truth/', parts(1), '_', parts(2), '_GT_', parts(3), '_', parts(4), '.mat'));
    mat_file = char(strcat(prefix, 'ground_truth/', base_name(1), '_ann.mat'));

    base_name = split(mat_file, '.');
    csv_file = char(strcat(base_name(1), '.csv'));


    im = imread(img_file);
    [h, w, c] = size(im);


    val = load(mat_file);
%     p = val.image_info{1, 1}.location;
%     s = val.image_info{1, 1}.number;
    p = val.annPoints;
    s = size(p, 1);


    xcoord = p(:, 1);
    ycoord = p(:, 2);

    T = table(xcoord, ycoord);
    writetable(T, csv_file);

    fprintf(csv, '%s,%d,%d,%d,%d\n', char(files(i)), h, w, c, s);
    disp(i/num*100);
end
fclose(csv);