from ultralytics import YOLO

def main():
    data_yaml_path = 'D:/9nueveno/inteligencia artificial/laultima/YOLO/datasets2/data.yaml'
    
    import torch
    if torch.cuda.is_available():
        torch.device('cpu')
   
    model = YOLO('yolov8s.pt')

    model.train(
        data=data_yaml_path,
        epochs=30,
        batch=16,
        imgsz=640,
        lr0=5e-4,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        save_period=1,
        project='runs_final/train',
        name='yolov8s_run_final',
        exist_ok=True,
        device='cpu'  # Asegurarse de usar la CPU
    )

if __name__ == '__main__':
    main()
