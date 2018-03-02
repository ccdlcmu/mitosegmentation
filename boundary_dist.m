% average surface distance, one more metric to measure the segmetation result 
% other than the area similarity 

clear

% file1_name = 'mask';
% file2_name = 'imgs';

imgs = dir([file1_name,'/*.tif']);

ave_surf_dist = zeros(1,length(imgs));

d_thr = Inf; % 3d+1
for i=1:length(imgs)
    im1 = imread([file1_name,'/',imgs(i).name]);
    im2 = imread([file2_name,'/',imgs(i).name]);
    im1 = double(im1); im1(im1>0)=1;
    im2 = double(im2); im2(im2>0)=1;
    b1 = bwboundaries(im1);
    b2 = bwboundaries(im2);
%     imshow(im1)
%     hold on
%     for k=1:length(b1)
%         boundary=b1{k};
% %         plot(boundary(:,2),boundary(:,1))
%         scatter(boundary(:,2),boundary(:,1),1,'*')
%     end
    bp1=[];
    for k=1:length(b1)
        bp1=[bp1;b1{k}];
    end
    bp2=[];
    for k=1:length(b2)
        bp2=[bp2;b2{k}];
    end    
    D = pdist2(bp1,bp2);
    minD1 = min(D,[],1);
    minD2 = min(D,[],2);
     % to delete some very far points
    minD1 = minD1(minD1<d_thr);
    minD2 = minD2(minD2<d_thr);
    
    minD =  sum(minD1)+sum(minD2);
    bNum =  length(minD1)+length(minD2);
    ave_surf_dist(i) = minD/bNum;
end

dist_m = mean(ave_surf_dist)
dist_sem = std(ave_surf_dist)/sqrt(length(imgs))
