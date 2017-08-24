%%  Function to extract best boxes from a set of boxes
%  Best boxes are defined as the boxes with the highest IOU overlap ratio
%  with the ground truth boxes
%
%  INPUTS
%  boxes   -  Set of boxes to find best boxes from [x_min,y_min,x_max,y_max]
%             boxes is an array of dimension (number of boxes X 4)
%  gtBoxes -  Ground truth annotations [x_min,y_min,x_max,y_max]
%             gtBoxes is an array of dimension (number of objects X 4)
%
%  OUTPUTS
%  boxId   -  index of best boxes from the set of boxes
%  maxIou  -  IOU of best boxes with ground truths

function [boxId,maxIou] = bestBoxes(boxes,gtBoxes)

boxes = [boxes(:,1),boxes(:,2),boxes(:,3)-boxes(:,1)+1,boxes(:,4)-boxes(:,2)+1];
gtBoxes = [gtBoxes(:,1),gtBoxes(:,2),gtBoxes(:,3)-gtBoxes(:,1)+1,gtBoxes(:,4)-gtBoxes(:,2)+1];

int = rectint(boxes,gtBoxes);
boxArea = boxes(:,3).*boxes(:,4);
gtArea  = gtBoxes(:,3).*gtBoxes(:,4);
union   = bsxfun(@plus,boxArea,gtArea') - int;
iou     = int./union;

[maxIou,boxId] = max(iou,[],1);