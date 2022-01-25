clear;
clc;
close;

folder = '/media/user/Data/odrive/grad/traffic_signal/data/shanghai/';
% grid = 7;
image_size = 128;

txt = fopen(strcat(folder, 'shanghai_dense_labels.csv'), 'w');
fprintf(txt, 'names,count\n');

images = split(ls(strcat(folder, 'images')));
total = size(images) - 1;
total = total(1, 1);

for i=1:total
    
    name = char(images(i));
    %     name = 'test_A_IMG_92.jpg';
    
    base_name = split(name, '.');
    parts = split(base_name(1), '_');
    mat_file = char(strcat(folder, 'ground_truth/', parts(1), '_', parts(2), '_', 'GT_', parts(3), '_' ,parts(4), '.mat'));
    
    im = imread(strcat(folder, 'images/', name));
    
    [h, w, c] = size(im);
    %     disp([h, w]);
    
    xgrid = floor(w / image_size);
    ygrid = floor(h / image_size);
    
    xstep = ceil(w / xgrid);
    ystep = ceil(h / ygrid);
    
    count = 0;
    for y = 1:ystep:h
        for x = 1:xstep:w
            
            xmin = x;
            ymin = y;
            xmax = xmin + xstep;
            ymax = ymin + ystep;
            %             disp([xmin, ymin, xmax, ymax]);
            count = count + 1;
            %             disp(count);
            %             subplot(grid, grid, count);
            
            cp_im = imcrop(im, [xmin, ymin, xmax-xmin, ymax-ymin]);
            
            cpim_name = char(strcat(base_name(1), '_', int2str(count), '.jpg'));
            
            imwrite(cp_im, strcat(folder, 'patch/', cpim_name));
            
            grid_count = gt_patch(mat_file, xmin, xmax, ymin, ymax);
            
            fprintf(txt, '%s,%d\n', cpim_name, grid_count);
        end
    end
    disp(name);
    fprintf('%d / %d = %f\n', i, total, 100*i/total);
    %     break;
end

fclose(txt);

function grid_count = gt_patch(mat_file, xmin, xmax, ymin, ymax)

val = load(mat_file);
p = val.image_info{1, 1}.location;
s = val.image_info{1, 1}.number;

crop = [0, 0];

grid_count = 1;
for i=1:s
    if ((p(i, 1) >= xmin && p(i, 1) <= xmax) && (p(i, 2) >= ymin && p(i, 2) <= ymax))
        crop(grid_count, :) = p(i, :);
        grid_count = grid_count + 1;
    end
end
grid_count = grid_count - 1;

end

