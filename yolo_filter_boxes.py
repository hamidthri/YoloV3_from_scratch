import torch
import iou
def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .6):
    scores = box_confidence * box_class_probs
    box_classes = torch.argmax(scores, dim=-1)
    box_class_scores, _ = torch.max(scores, dim=-1)
    
    filtering_mask = box_class_scores >= threshold
    
    scores = box_class_scores[filtering_mask]
    boxes = boxes[filtering_mask]
    classes = box_classes[filtering_mask]
    
    return scores, boxes, classes


def non_max_suppression(scores, boxes, classes, iou_treshold=0.5):
    
    _, indices = torch.sort(scores, descending=True)
    
    boxes = boxes[indices]
    scores = scores[indices]
    classes = classes[indices]
    
    keep_boxes = []
    keep_scores = []
    keep_classes = []
    
    while (len(boxes) > 0):
        keep_boxes.append(boxes[0])
        keep_scores.append(scores[0])
        keep_classes.append(classes[0])
        
        if len(boxes) == 1:
            break
        
        ious = torch.tensor([iou(boxes[0], box) for box in boxes[1:]])
        mask = (ious <= iou_treshold) & (classes[1:] != classes[0])
        
        boxes = boxes[1:][mask]
        scores = scores[1:][mask]
        classes = classes[1:][mask]
        
    return keep_scores, keep_boxes, keep_classes
