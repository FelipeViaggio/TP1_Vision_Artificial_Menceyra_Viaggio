import cv2
from matplotlib import pyplot as plt

# --- PREPROCESAMIENTO ---
def imread_rgb(path):
    """
    Lee una imagen y la devuelve en formato RGB (OpenCV lee en BGR).
    
    Args:
        path (str or Path): Ruta a la imagen.
    
    Returns:
        img_rgb (np.ndarray): Imagen en formato RGB.
    """
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    assert img_bgr is not None, f"No se pudo leer la imagen {path}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    return img_rgb

def resize_width(img, width):
    """
    Redimensiona una imagen manteniendo la relación de aspecto.
    
    Args:
        img (np.ndarray): Imagen a redimensionar.
        width (int): Nuevo ancho.
    
    Returns:
        img_resized (np.ndarray): Imagen redimensionada.
    """
    if width is None or img.shape[1] <= width:
        return img
    r = width / img.shape[1]

    return cv2.resize(img, (int(img.shape[1] * r), int(img.shape[0] * r)), interpolation=cv2.INTER_AREA)

# --- VISUALIZACIÓN ---
def visualize_keypoints(images, idx, img_gray, keypoint, ANCHOR_IDX=1):
    """
    Visualiza los keypoints detectados en una imagen.
    
    Args:
        images (list of str): Lista de nombres de las imágenes.
        idx (int): Índice de la imagen actual.
        img_gray (np.ndarray): Imagen en escala de grises.
        keypoint (list of cv2.KeyPoint): Keypoints detectados.
        ANCHOR_IDX (int): Índice de la imagen ancla para el título.
    
    Returns:
        None"""
    img_kp = cv2.drawKeypoints(
        img_gray, keypoint, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    plt.figure(figsize=(12, 8))
    plt.imshow(img_kp)
    plt.title(f"{images[idx]}\nKeypoints: {len(keypoint)}" + ("\nAncla" if idx == ANCHOR_IDX else ""))
    plt.axis("off")
    plt.show()

def visualize_comparison(images, idx, img_gray, keypoints, keypoints_anms, ANCHOR_IDX=1):
    """
    Visualiza la comparación entre keypoints originales y keypoints seleccionados por ANMS.
    
    Args:
        images (list of str): Lista de nombres de las imágenes.
        idx (int): Índice de la imagen actual.
        img_gray (np.ndarray): Imagen en escala de grises.
        keypoints (list of cv2.KeyPoint): Keypoints originales detectados.
        keypoints_anms (list of cv2.KeyPoint): Keypoints seleccionados por ANMS.
        ANCHOR_IDX (int): Índice de la imagen ancla para el título.
    
    Returns:
        None
    """
    img_kp = cv2.drawKeypoints(
        img_gray, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    img_kp_anms = cv2.drawKeypoints(
        img_gray, keypoints_anms, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(img_kp)
    plt.title(f"{images[idx]}\nKeypoints: {len(keypoints)}" + ("\nAncla" if idx == ANCHOR_IDX else ""))
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(img_kp_anms)
    plt.title(f"{images[idx]} + ANMS\nKeypoints: {len(keypoints_anms)}" + ("\nAncla" if idx == ANCHOR_IDX else ""))
    plt.axis("off")

    plt.show()