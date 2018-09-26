dataDir = 'DIV2K_train_HR\';

mkdir('train_DIV2K_input_4x');
mkdir('train_DIV2K_label_4x');

folder_label = fullfile('train_DIV2K_label_4x');
folder_input_4x = fullfile('train_DIV2K_input_4x');

count_input = 0;
count_label = 0;
count = 0;
f_lst = [];
f_lst = [f_lst; dir(fullfile(dataDir, '*.png'))];

for f_iter = 1:numel(f_lst)
%     disp(f_iter);
    f_info = f_lst(f_iter);
    if f_info.name == '.'
        continue;
    end
    f_path = fullfile(dataDir,f_info.name);
    img_raw = imread(f_path);
    img_raw = im2double(img_raw);
    
    img_size = size(img_raw);
    width = img_size(2);
    height = img_size(1);
    
    %img_raw = img_raw(1:height-mod(height,12),1:width-mod(width,12),:);
    %img_size = size(img_raw);
    
	  patch_size = 48;
    stride = 48;
    x_size = (img_size(2)-patch_size)/stride+1;
    y_size = (img_size(1)-patch_size)/stride+1;
    
	
    for x = 0:x_size-1
        for y = 0:y_size-1
            x_coord = x*stride; y_coord = y*stride; 
            patch_name = sprintf('train/%d',count);
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            patch_raw = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0);
            patch = patch_raw;
            patch_name = sprintf('%s/%d',folder_label, count_label)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');
           
            patch_4x = imresize(patch_raw,1/4,'bicubic');
            patch = patch_4x;
            patch_name = sprintf('%s/%d',folder_input_4x, count_input)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');
            
            count_label = count_label + 1;
            count_input = count_input + 1;
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            patch_raw = imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90);
            patch = patch_raw;
            patch_name = sprintf('%s/%d',folder_label, count_label)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');

            patch_4x = imresize(patch_raw,1/4,'bicubic');
            patch = patch_4x;
            patch_name = sprintf('%s/%d',folder_input_4x, count_input)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');
            
            count_label = count_label + 1;
            count_input = count_input + 1;
                  
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            patch_raw = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 0));
            patch = patch_raw;
            patch_name = sprintf('%s/%d',folder_label, count_label)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');

            patch_4x = imresize(patch_raw,1/4,'bicubic');
            patch = patch_4x;
            patch_name = sprintf('%s/%d',folder_input_4x, count_input)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');
            
            count_label = count_label + 1;
            count_input = count_input + 1;

            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%            
            patch_raw = fliplr(imrotate(img_raw(y_coord+1:y_coord+patch_size,x_coord+1:x_coord+patch_size,:), 90));
            patch = patch_raw;
            patch_name = sprintf('%s/%d',folder_label, count_label)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');

            patch_4x = imresize(patch_raw,1/4,'bicubic');
            patch = patch_4x;
            patch_name = sprintf('%s/%d',folder_input_4x, count_input)
            save(sprintf('%s.mat', patch_name), 'patch', '-v6');
            
            count_label = count_label + 1;
            count_input = count_input + 1; 
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
        end
    end
    
    display(count);
    count = count + 1;
    
end