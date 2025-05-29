from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch

from PIL import Image
import cv2
import numpy as np

from paddleocr import PaddleOCR
from dataclasses import dataclass

@dataclass
class BoxElement:
    type: str
    bbox: np.array
    interactivity: bool
    content: str = None
    source_bbox: str = None
    source_content: str = None

    def __eq__(self, other):
        return np.all(self.bbox == other.bbox)

def get_paddle_ocr_model():
    paddle_ocr_model = PaddleOCR(
        lang='en',  # other lang also available
        use_angle_cls=False,
        use_gpu=False,  # using cuda will conflict with pytorch in the same process
        show_log=False,
        max_batch_size=1024,
        use_dilation=True,  # improves accuracy
        det_db_score_mode='slow',  # improves accuracy
        rec_batch_num=1024)
    return paddle_ocr_model

def run_ocr_model(ocr_model, image_arr, threshold=0.5):
    if isinstance(ocr_model, PaddleOCR):
        texts, bboxes = [], []
        ocr_outs = ocr_model.ocr(image_arr, cls=False)[0]
        for out in ocr_outs:
            coord, (text, score) = out
            if score < threshold: continue # filter with given threshold

            # convert to xywh format
            (x1, y1), (x2, y2) = coord[0], coord[2]
            xywh = np.array([x1, y1, x2-x1, y2-y1], dtype=np.float32)

            texts.append(text)
            bboxes.append(xywh)

        box_elems = [BoxElement('text', bbox, False, text, 'ocr', 'ocr') 
                     for bbox, text in zip(bboxes, texts)]
        return box_elems
    
    else: 
        raise NotImplementedError()

def get_yolo_model(model_path):
    model = YOLO(model_path, task='detect')
    return model

def run_yolo_model(yolo_model, image, yolo_model_conf):
    if yolo_model_conf is None:
        yolo_model_conf = {}
    result = yolo_model.predict(source=image, **yolo_model_conf)[0]
    
    bboxes = result.boxes.xywh.cpu().numpy()
    bboxes[:, :2] -= bboxes[:, 2:] / 2 # yolo to coco format
    
    box_elems = [BoxElement('icon', bbox, True, None, 'yolo', None) for bbox in bboxes]
    return box_elems

def get_caption_model(model_path, device='cuda'):
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True).to(device)

    return dict(model=model, processor=processor)

def run_caption_model(caption_model, image_arr, bboxes, batch_size=128):
    croped_pil_image = []
    for x, y, w, h in bboxes:
        left, top, right, bottom = map(round, [x, y, (x+w), (y+h)])

        cropped_image = image_arr[top:bottom, left:right, :]
        cropped_image = cv2.resize(cropped_image, (64, 64))
        croped_pil_image.append(Image.fromarray(cropped_image))

    model, processor = caption_model['model'], caption_model['processor']
    prompt = "<CAPTION>"

    captions = []
    with torch.no_grad():
        for i in range(0, len(croped_pil_image), batch_size):
            batch = croped_pil_image[i:i+batch_size]
            
            x = processor(images=batch, text=[prompt]*len(batch), return_tensors="pt", do_resize=False)
            input_ids = x['input_ids'].to(dtype=torch.int32, device='cuda')
            pixel_values = x['pixel_values'].to(dtype=torch.float16, device='cuda')
            
            output_ids = model.generate(input_ids=input_ids, 
                                           pixel_values=pixel_values,
                                           max_new_tokens=20, num_beams=1, do_sample=False)

            output_text = processor.batch_decode(output_ids, skip_special_tokens=True)
            output_text = [i.strip() for i in output_text]
            captions += output_text
    
    return captions

def _compute_iou_and_area(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = w1 * h1
    box2_area = w2 * h2
    
    union_area = box1_area + box2_area - inter_area
    iou = inter_area / union_area if union_area > 0 else 0
    return iou, box1_area, box2_area

def _is_inside(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    return x2 <= x1 and y2 <= y1 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2

def remove_overlap(ocr_elems, yolo_elems, iou_threshold=0.9):
    filtered_elems = []
    filtered_elems.extend(ocr_elems) # 1. keep all the ocr results
    
    # 2. filter overlapping yolo bboxes, and keep the smaller one
    filtered_yolo_elems = []
    for i in range(len(yolo_elems)):
        box1 = yolo_elems[i].bbox
        flag = True # flag to keep this yolo elem

        for j in range(i+1, len(yolo_elems)):
            box2 = yolo_elems[j].bbox
            
            iou, box1_area, box2_area = _compute_iou_and_area(box1, box2)
            if iou > iou_threshold and box1_area > box2_area:
                flag = False
                break
        
        if flag:
            filtered_yolo_elems.append(yolo_elems[i])
        
        
    for yolo_elem in filtered_yolo_elems:
        yolo_bbox = yolo_elem.bbox
        
        flag = True # flag to add this yolo bbox to filtered boxes or not
        ocr_contents = []
        
        for ocr_elem in ocr_elems:
            ocr_bbox = ocr_elem.bbox
            # if ocr bbox is inside icon bbox, take the content and delete that ocr bbox
            # else if icon bbox is in ocr bbox, drop the icon, set box_added to True

            if _is_inside(ocr_bbox, yolo_bbox): # ocr inside icon
                ocr_contents.append(ocr_elem.content)
                if ocr_elem in filtered_elems:
                    filtered_elems.remove(ocr_elem)
            
            elif _is_inside(yolo_bbox, ocr_bbox): # this yolo bbox is not required, since this icon is already included by ocr bbox
                flag = False
                break
        
        if flag:
            new_elem = BoxElement('icon', yolo_bbox, True, None, 'yolo', 'yolo')
            if ocr_contents:
                content = ' '.join(ocr_contents)
                new_elem.content = content
                new_elem.source_bbox = 'yolo'
                new_elem.source_content = 'ocr'
            
            filtered_elems.append(new_elem)
    
    return filtered_elems
