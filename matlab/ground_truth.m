clear;
clc;
close;

dir = '';

name = 'test_img_0100';

image_file = strcat(dir, name, '.jpg');
mat_file = strcat(dir, name, '_ann.mat');

xmin = 1;
ymin = 590;
xmax = 799;
ymax = 768;

im = imread(image_file);
figure;
imshow(im);
hold on;

val = load(mat_file);
p = val.image_info{1, 1}.location;
s = val.image_info{1, 1}.number;

crop = [0, 0];

count = 1;
for i=1:s
    if ((p(i, 1) >= xmin && p(i, 1) <= xmax) && (p(i, 2) >= ymin && p(i, 2) <= ymax))
        crop(count, :) = p(i, :);
        count = count + 1;
    end
end
count = count - 1;

rectangle('Position',[xmin ymin xmax-xmin ymax-ymin], 'EdgeColor', 'red', 'LineWidth', 1.5)
scatter(crop(:, 1), crop(:, 2), 100, 'x', 'red', 'LineWidth', 1.5);
disp(count);