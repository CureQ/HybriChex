# app/model/classifier_utils.py

import torch
from .model import model, device
import cv2

class Classifier:
    def __init__(self):
        self.model = model.to(device)
        self.device = device
        self.model.eval()

        self.class_names = [
           'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
          'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
          'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
        ]

        self.thresholds = [
            0.380, 0.307, 0.332, 0.351, 0.386, 0.300, 0.285,
            0.224, 0.414, 0.311, 0.316, 0.297, 0.336, 0.311
        ]


    def classify(self, image_tensor):
        with torch.no_grad():
            image_tensor = image_tensor.to(self.device)
            outputs = self.model(image_tensor)
            # --- Fix: outputs kan een list zijn (bij multi-head) ---
            if isinstance(outputs, list):
                # outputs: list of [B, 2] per class, pak index 1 (positief) per class
                outputs = torch.stack([o[:,1] if o.shape[-1]==2 else o.squeeze(-1) for o in outputs], dim=1)
            # outputs: [B, num_classes]
            probs = torch.sigmoid(outputs).cpu().numpy()[0]

        # Classificaties boven drempel
        results = []
        for i, prob in enumerate(probs):
            if prob >= self.thresholds[i]:
                results.append((self.class_names[i], float(prob)))

        # Top 5 voorspellingen (altijd hoogste 5, ongeacht drempel)
        prob_list = [(self.class_names[i], float(prob)) for i, prob in enumerate(probs)]
        top5 = sorted(prob_list, key=lambda x: x[1], reverse=True)[:5]

        return {
            'classifications': results,
            'top5': top5
        }

    def generate_gradcam(self, image_tensor, class_idx):
        import torch
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        # Wrapper zodat GradCAM altijd een [B, num_classes] tensor krijgt
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                out = self.model(x)
                if isinstance(out, list):
                    out = torch.stack([o[:,1] if o.shape[-1]==2 else o.squeeze(-1) for o in out], dim=1)
                return out

        cam_model = ModelWrapper(self.model)
        # Gebruik expliciet de laatste conv-layer van de CNN-backbone
        last_layer = self.model.cnn_backbone.features[-1]

        cam = GradCAM(model=cam_model, target_layers=[last_layer])
        grayscale_cam = cam(input_tensor=image_tensor, targets=[ClassifierOutputTarget(class_idx)])
        return grayscale_cam[0]

    def gradcam_overlay(self, original_img, grayscale_cam, use_rgb=True):
        """
        Overlay GradCAM heatmap met pytorch-grad-cam's show_cam_on_image.
        - original_img: numpy array (H, W, 3) of (H, W), mag 0-255 of 0-1 zijn, mag greyscale zijn
        - grayscale_cam: numpy array (H, W), waarden 0-1 of 0-255
        Returns: overlay image (uint8, RGB als use_rgb=True)
        """
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import numpy as np

        # Forceer originele afbeelding naar float32 en 0-1, en altijd 3 kanalen (RGB)
        if original_img.dtype != np.float32:
            original_img = original_img.astype(np.float32)
        if original_img.max() > 1.0:
            original_img = original_img / 255.0
        if len(original_img.shape) == 2:
            # Greyscale naar RGB
            original_img = np.stack([original_img]*3, axis=-1)
        elif original_img.shape[2] == 1:
            # (H, W, 1) naar (H, W, 3)
            original_img = np.repeat(original_img, 3, axis=2)

        # Forceer gradcam naar float32 en 0-1
        grayscale_cam = np.nan_to_num(grayscale_cam)
        if grayscale_cam.max() > 1.0:
            grayscale_cam = grayscale_cam / 255.0
        grayscale_cam = np.clip(grayscale_cam, 0, 1)

        cam_image = show_cam_on_image(
            original_img,
            grayscale_cam,
            use_rgb=True,
            colormap=cv2.COLORMAP_JET  # <-- Gebruik JET voor blauw-laag, rood-hoog
        )


        
        return cam_image

    # def gradcam_overlay_bgr(self, original_img, grayscale_cam):
    #     """
    #     Overlay GradCAM heatmap en geef altijd BGR terug (voor cv2.imshow of Flask).
    #     """
    #     cam_image = self.gradcam_overlay(original_img, grayscale_cam, use_rgb=True)
    #     import cv2
    #     cam_image_bgr = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    #     return cam_image_bgr




