import argparse
import os
import cv2
from ultralytics import YOLO

def run_inference(weights, source, imgsz, conf, project, name, save_txt, save_conf):
    # Cargar el modelo
    model = YOLO(weights)
    
    # Crear el directorio de salida si no existe
    save_dir = os.path.join(project, name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Abrir el video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error al abrir el video: {source}")
        return
    
    # Obtener información del video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # o 'XVID' para AVI
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Crear el VideoWriter para guardar el video procesado
    out_video_path = os.path.join(save_dir, f"{name}_processed.mp4")
    out = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))
    print(f"Video procesado guardado en: {out_video_path}")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar el fotograma
        results = model.predict(source=frame, imgsz=imgsz, conf=conf)
        
        # Mostrar resultados en el fotograma
        for result in results:
            # Aquí asumimos que 'result.plot()' devuelve una imagen con anotaciones
            annotated_frame = result.plot()
            out.write(annotated_frame)
        
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"Proceso completado. {frame_count} fotogramas procesados.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='Path to weights file')
    parser.add_argument('--source', type=str, required=True, help='Path to video file')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold for detections')
    parser.add_argument('--project', type=str, default='runs/detect', help='Project directory')
    parser.add_argument('--name', type=str, default='custom_detect', help='Name of the run')
    parser.add_argument('--save-txt', action='store_true', help='Save results to text files')
    parser.add_argument('--save-conf', action='store_true', help='Save confidences in text files')
    
    args = parser.parse_args()
    run_inference(args.weights, args.source, args.imgsz, args.conf, args.project, args.name, args.save_txt, args.save_conf)
