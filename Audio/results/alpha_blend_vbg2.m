function alpha_blend_vbg2(srcdirbg,srcdir)
	disp(srcdirbg);
	files = dir(fullfile(srcdirbg,'*.png'));
	for i = 1:length(files)
	    file = files(i).name;
		if length(file) >= 11 && strcmp(file(end-10:end),'_blend2.png')
			continue;
		end
		tarname = [file(1:end-4),'_blend2.png'];
		if exist(fullfile(srcdir,tarname))
			continue;
		end
		im1 = imread(fullfile(srcdirbg,file));
		%disp(fullfile(srcdirbg,file));
	    im2 = imread(fullfile(srcdir,file));
        trans = imread(fullfile(srcdir,[file(1:end-4),'_mask.png']));
	    [~,L] = bwboundaries(trans);
	    
        % BEGIN: deprecated by RainEggplant, 220714
        % trans(L>=2) = 255;

	    % trans = double(trans)/255;
        % Replaced by: deprecated by RainEggplant, 220714
        trans(L>=2) = 1;
	    trans = double(trans);
        % END: deprecated by RainEggplant, 220714
        
	    im3 = double(im1).*(1-trans) + double(im2).*trans;
	    im3 = uint8(im3);
	    
	    %disp(tarname);
	    imwrite(im3,fullfile(srcdir,tarname));
	end
end