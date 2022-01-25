clear;
clc;
close;

dir = '';

name = 'test_img_0100';

image_file = strcat(dir, name, '.jpg');
mat_file = strcat(dir, name, '_ann.mat');

partial = 0;

im = imread(image_file);

res = size(im);
w = res(2);
h = res(1);

figure;
imshow(im);
hold on;

if (partial == 1)
    val = load(mat_file);    
    
    old_p = val.image_info{1, 1}.location;
    old_s = val.image_info{1, 1}.number;
    
    scatter(old_p(:, 1), old_p(:, 2), 100, 'x', 'red', 'LineWidth', 2);
    
    disp(old_s);
end

[x, y] = getpts;
new_p = [x, y];
new_s = size(x, 1);

if (partial == 1)
    new_p = cat(1, old_p, new_p);
    new_s = old_s + new_s;  
end

crop = [0, 0];

count = 1;
for i=1:new_s
    if ((new_p(i, 1) >= 0 && new_p(i, 1) <= w) && (new_p(i, 2) >= 0 && new_p(i, 2) <= h))
        crop(count, :) = new_p(i, :);
        count = count + 1;
    end
end
count = count - 1;

new_p = crop;
new_s = count;

image_info{1, 1}.location = new_p;
image_info{1, 1}.number = new_s;
save(mat_file, 'image_info');

disp(new_s);

close;