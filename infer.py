import torch
import os
import cv2
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from super_gradients.training import models

# Configuración del dispositivo
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Cargar el modelo
model = models.get(
    model_name='yolo_nas_s',
    checkpoint_path='D:/9nueveno/inteligencia artificial/laultima/checkpoints/yolo_nas_s/ckpt_best.pth',
    num_classes=7  # Cambia el número de clases según tu dataset
).to(device)

# Directorios
ROOT_TEST = 'dataset/test/images/'
all_images = os.listdir(ROOT_TEST)

# Directorio de resultados
os.makedirs('inference_results/images/', exist_ok=True)

# Realizar inferencia
for image in tqdm(all_images, total=len(all_images)):
    image_path = os.path.join(ROOT_TEST, image)
    out = model.predict(image_path)
    out.save('inference_results/images')
    os.rename(
        'inference_results/images/pred_0.jpg', 
        os.path.join('inference_results/images/', image)
    )

# Visualizar resultados
def plot_box(image, bboxes, labels):
    # Código para visualizar las cajas delimitadoras (similar al que proporcionaste)
    # ...
    return image

def plot(image_path, label_path, num_samples):
    # Código para visualizar imágenes con las cajas delimitadoras
    # ...
    plt.show()

plot(
    image_path='inference_results/images/', 
    label_path='dataset/test/labels/',
    num_samples=10,
)
