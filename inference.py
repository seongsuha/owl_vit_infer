import os
import cv2
import json
import requests
import argparse
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

import torch, gc
gc.collect()
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
    
score_threshold = 0.1 # TODO


def plot_outputs(image, queries, scores, boxes, labels, color=(0,255,0)):
    image = np.asarray(image.copy())
    for score, box, label in zip(scores, boxes, labels):
        if score < score_threshold:
            continue
        box_int = list(map(int, box))
        x1y1 = (box_int[0], box_int[1])
        x2y2 = (box_int[2], box_int[3])
        image = cv2.rectangle(image, x1y1, x2y2, color, 2)
        image = cv2.putText(image, f"{queries[label]}:{score:1.2f}",
                            (box_int[0]+3, box_int[1]+25), 1, 1.5, color, 1, cv2.LINE_AA)
    image = Image.fromarray(image)
    return image    


def inference_grit(args, model, processor):
    img_base_path = f"{args.grit_path}/images"
    sample_json_path = f"{args.grit_path}/samples/ablation/localization.json"
    with open(sample_json_path, 'r') as f:
        item_infos = json.load(f)
        item_infos = [x for x in item_infos if 'coco' in x['image_id']]
    
    with torch.no_grad():
        output_json = []
        for i, item_info in enumerate(item_infos):
            if i%100==0:
                print(f"{i}/{len(item_infos)}th item processing...")
                
            try:
                img_path = f"{img_base_path}/{item_info['image_id']}"
                image = Image.open(img_path).convert("RGB")
                text = [item_info['task_query']]                
            except Exception as e:
                temp = {
                    'example_id': item_info['example_id'],
                    'confidence': float(0.0),
                    'bboxes': [[0,0,50,50]]   
                }
                output_json.append(temp)
                continue
                
            inputs = processor(images=image, text=text, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            target_sizes = torch.Tensor([image.size[::-1]]).to(device)
            results = processor.post_process_object_detection(outputs=outputs,
                                                              target_sizes=target_sizes,
                                                              threshold=score_threshold)[0]
            boxes = results["boxes"].cpu().detach().numpy()
            scores = results["scores"].cpu().detach().numpy()
            output_json.append({
                'example_id': item_info['example_id'],
                'confidence': float(scores.mean()) if len(scores)!=0 else 0,
                'bboxes': [list(map(int, box)) for box in boxes.tolist()]   
            })
            
    output_dir = f"{args.output_base}/{args.split}"
    os.makedirs(output_dir, exist_ok=True)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = int(trainable_params / 1e6) # number of params in million
    params_json = {"params_in_millions":num_params}    
    with open(f"{output_dir}/params.json", 'w') as f:
        json.dump(params_json, f)
    
    with open(f"{output_dir}/localization.json", 'w') as f:
        json.dump(output_json, f)
        

def inference_images(args, model, processor):
    image = Image.open(args.image_path).convert('RGB')
    text = [args.query]
    print(f"Image Loaded")
    
    with torch.no_grad():
        inputs = processor(images=image, text=text, return_tensors="pt").to(device)
        outputs = model(**inputs)
        
        target_sizes = torch.Tensor([image.size[::-1]]).to(device)
        results = processor.post_process_object_detection(outputs=outputs,
                                                          target_sizes=target_sizes,
                                                          threshold=score_threshold)[0]
        boxes, scores, labels = results["boxes"], results["scores"], results["labels"]
        boxes = boxes.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        
    file_name = os.path.split(args.image_path)[-1]
    file_name = os.path.join(args.output_base, f"{file_name[:file_name.rfind('.')]}_output.jpg")
    output_image = plot_outputs(image, text, scores, boxes, labels)
    output_image.save(file_name, "JPEG")
    print(f"Output saved at {file_name}")
    

def parse_args():
    parser = argparse.ArgumentParser(description='OWL_ViT Inference')
    parser.add_argument("--image_path",
                        type=str,
                        help='image path to infer')
    parser.add_argument("--query",
                        type=str,
                        help="text query to localize")
    parser.add_argument("--output_base",
                        default="./outputs",
                        type=str,
                        help='output base directory')
    
    parser.add_argument("--grit",
                        action='store_true')
    parser.add_argument("--grit_path",
                        default='/data2/projects/seongsu/seongsu/twelve_labs/grit_official/data/GRIT',
                        type=str,
                        help='path to GRIT')
    parser.add_argument("--split",
                        default="ablation",
                        type=str,
                        help='which split to work on')

    args = parser.parse_args()
    return args


def main(args):
    processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
    model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(device)
    if args.grit:
        inference_grit(args, model, processor)
    else:
        inference_images(args, model, processor)
    print("Inference Done.")    


if __name__ == "__main__":
    args = parse_args()
    main(args)