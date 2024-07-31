import argparse
import cv2
import os
from ultralytics import YOLO

def run_inference(weights, source, imgsz, conf, project, name, save_txt, save_conf):
    # Load the YOLO model
    model = YOLO(weights)

    # Create project directory if it doesn't exist
    os.makedirs(project, exist_ok=True)

    # Process the source
    results = model.predict(source, imgsz=imgsz, conf=conf)

    # Save results
    for result in results:
        # Get the original image
        img = result.orig_img

        # Draw bounding boxes and labels
        for det in result.boxes:
            # Convert tensors to numpy arrays and then to lists
            xyxy = det.xyxy.cpu().numpy()
            conf = det.conf.cpu().numpy()
            cls = det.cls.cpu().numpy()

            for i in range(len(xyxy)):
                x1, y1, x2, y2 = map(int, xyxy[i])
                conf_val = conf[i]
                cls_val = int(cls[i])
                label = f'{model.names[cls_val]} {conf_val:.2f}'
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the processed image
        filename = os.path.join(project, f'{name}.jpg')
        cv2.imwrite(filename, img)

        # Optionally save text results
        if save_txt:
            txt_path = os.path.join(project, f'{name}.txt')
            with open(txt_path, 'w') as f:
                for i in range(len(xyxy)):
                    x1, y1, x2, y2 = map(float, xyxy[i])
                    conf_val = conf[i]
                    cls_val = int(cls[i])
                    line = f'{cls_val} {conf_val} {x1} {y1} {x2} {y2}\n'
                    f.write(line)

        # Optionally save confidences
        if save_conf:
            conf_path = os.path.join(project, f'{name}_confidences.txt')
            with open(conf_path, 'w') as f:
                for conf_val in conf:
                    f.write(f'{conf_val}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="Path to model weights")
    parser.add_argument("--source", type=str, required=True, help="Path to image or video")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for inference")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for predictions")
    parser.add_argument("--project", type=str, default="runs/detect", help="Project directory for saving results")
    parser.add_argument("--name", type=str, default="exp", help="Name for the results")
    parser.add_argument("--save-txt", action="store_true", help="Save results in text format")
    parser.add_argument("--save-conf", action="store_true", help="Save confidences in text format")
    args = parser.parse_args()

    run_inference(args.weights, args.source, args.imgsz, args.conf, args.project, args.name, args.save_txt, args.save_conf)
