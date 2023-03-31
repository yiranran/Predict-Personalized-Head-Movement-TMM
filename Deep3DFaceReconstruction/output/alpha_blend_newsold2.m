function alpha_blend_newsold2(video, starti, framenum)
%video = 'Learn_English';
%starti = 357; % choose 400, 300 for training render-to-video, 100 for testing
%framenum = 400;

srcdir = ['render/',video,'/'];
srcdir2 = ['render/',video,'/'];
tardir = ['render/',video,'/bm/'];
files = dir(fullfile(srcdir,'*.png'));
t1=tic;
if ~exist(tardir)
    mkdir(tardir);
end
for i = starti:(starti+framenum-1)
    file = ['frame',num2str(i),'_input2.png'];
    im1 = imread(fullfile(srcdir,file));
    [im2,~,trans] = imread(fullfile(srcdir2,['frame',num2str(i),'_render3.png']));
    [B,L] = bwboundaries(trans);
    %imshow(label2rgb(L,@jet,[.5,.5,.5]));
    trans(L>=2) = 255;
    %figure;imshow(trans);
    trans = double(trans)/255;
    im3 = double(im1).*(1-trans) + double(im2).*trans;
    im3 = uint8(im3);
    imwrite(im3,fullfile(tardir,['frame',num2str(i),'_renderold_bm2.png']));
end
toc(t1)%1094.343765 seconds