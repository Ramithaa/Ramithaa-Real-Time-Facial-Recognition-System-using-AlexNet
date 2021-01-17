
function I = classifyVideo(bNet)
load bNet
cam = webcam;
count=0;

while (1)
picture = cam.snapshot; 
picture = imresize (picture,[227,227]); 

[label,score] = classify(bNet,picture); 
image(picture); 
title({char(label), num2str(max(score),2)});% Show the label 
count=count+1; % counter to rename the images taken via the webcam
imwrite(picture,sprintf('[label]%d.jpg' ,count));
drawnow; %update all charts

end 

