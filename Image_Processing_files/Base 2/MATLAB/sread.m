% Giving directions 
function I = sread(filename,k,f)

% filename: image name. For example: 'kungfu.jpg'
% k: intensity decision argument. When k=0, input the original image, and when k=1, input the original image with intensity f
% f: parameter for the RGB intensity
	
% -------------------------------------------------
% load the image
Image = imread(filename);


% -------------------------------------------------
% If k=0, then load the original image
if (k==0)
	I = Image;
% If k=1, then load the orginal image using the intensity giving by f
elseif (k==1)
	I = Image(:,:,1)*f(:,1) + Image(:,:,2)*f(:,2)+ Image(:,:,3)*f(:,3);
end

end
