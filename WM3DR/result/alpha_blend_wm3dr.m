function alpha_blend_wm3dr(video, starti, framenum)
%video = '21_crop';
%starti = 0;
%framenum = 400;

srcdir = ['../../Data/',video,'/'];
srcdir2 = [video,'/'];
tardir = [video,'/bm/'];
t1=tic;
if ~exist(tardir)
    mkdir(tardir);
end
for i = starti:(starti+framenum-1)
    file = [num2str(i,'%05d'),'.png'];
    file1 = [num2str(i,'%05d'),'.png'];
    if ~exist(fullfile(srcdir,file1))
        file1 = ['frame',num2str(i,'%d'),'.png'];
    end
    if exist(fullfile(tardir,[file(1:end-4),'_wm3dr_bm.png']))
        continue;
    end
    im1 = imread(fullfile(srcdir,file1));
    im2 = imread(fullfile(srcdir2,[file(1:end-4),'_render.png']));
    trans = imread(fullfile(srcdir2,[file(1:end-4),'_mask.png']));
    [~,L] = bwboundaries(trans);
    %imshow(label2rgb(L,@jet,[.5,.5,.5]));
    trans(L>=2) = 255;
    %figure;imshow(trans);
    trans = double(trans)/255;
    im3 = double(im1).*(1-trans) + double(im2).*trans;
    im3 = uint8(im3);
    imwrite(im3,fullfile(tardir,[file(1:end-4),'_wm3dr_bm.png']));
end
toc(t1)