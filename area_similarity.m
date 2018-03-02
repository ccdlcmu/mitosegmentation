% intersection over union accuracy 
clear

% mat file is converted from npy file
mat_file = 'matdata/fcn_mask_test.mat';
load(mat_file,'predict')
mask_file = 'mask';

imgs = dir([mask_file,'/*.tif']);
as_acc = zeros(1,length(imgs));

for i=1:length(imgs)
    im1 = imread([mask_file,'/',imgs(i).name]);
		
    im2 = squeeze(predict(i,:,:));
    im2 = double(im2); im2(im2>=0.5)=1; im2(im2<0.5)=0;
    
    e1 = length(find((im1+im2)==2));
    e2 = length(find((im1+im2)>=1));
    as_acc(i) = e1/e2;
end

as_m = mean(as_acc)
as_sem = std(as_acc)/sqrt(length(imgs))
