import cv2
from matplotlib import pyplot as plt
import numpy as np
from random import sample

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

def draw_matches_gray(grayA, grayB, kpsA, kpsB, matches, max_draw=80, title=""):
    """
    Visualiza matches sobre fondo gris, submuestreando para claridad.
    - grayA, grayB: imágenes en escala de grises (uint8).
    - kpsA, kpsB: listas de cv2.KeyPoint.
    - matches: list[cv2.DMatch].
    """
    if len(matches) > max_draw:
        idx = sorted(sample(range(len(matches)), max_draw))
        matches = [matches[i] for i in idx]

    A_bgr = cv2.cvtColor(grayA, cv2.COLOR_GRAY2BGR)
    B_bgr = cv2.cvtColor(grayB, cv2.COLOR_GRAY2BGR)

    vis = cv2.drawMatches(
        A_bgr, kpsA, B_bgr, kpsB, matches, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(14, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def draw_matches_subset_gray(grayA, grayB, kpsA, kpsB, matches, mask_bool, title="", max_draw=120):
    """
    Visualiza un subconjunto de matches (p. ej., inliers RANSAC) sobre fondo gris.
    
    Args:
        grayA (np.ndarray): Imagen A en escala de grises (uint8).
        grayB (np.ndarray): Imagen B en escala de grises (uint8).
        kpsA (list of cv2.KeyPoint): Keypoints de A.
        kpsB (list of cv2.KeyPoint): Keypoints de B.
        matches (list of cv2.DMatch): Lista completa de matches A<->B.
        mask_bool (array-like of bool): Máscara booleana para elegir qué matches dibujar.
        title (str): Título de la figura.
        max_draw (int): Máximo de matches a dibujar.
    
    Returns:
        None
    """
    # Filtrar por máscara (inliers/outliers)
    idxs = [i for i, keep in enumerate(mask_bool) if keep]
    if len(idxs) == 0:
        print("Nada para dibujar:", title)
        return
    if len(idxs) > max_draw:
        idxs = idxs[:max_draw]
    matches_sub = [matches[i] for i in idxs]

    A_bgr = cv2.cvtColor(grayA, cv2.COLOR_GRAY2BGR)
    B_bgr = cv2.cvtColor(grayB, cv2.COLOR_GRAY2BGR)
    vis = cv2.drawMatches(
        A_bgr, kpsA, B_bgr, kpsB, matches_sub, None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(14, 6))
    plt.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def plot_inliers_side_by_side(imgA, imgB, ptsA, ptsB, inliers_mask, max_draw=300):
    """
    Muestra A | B con líneas verdes solo entre inliers (estilo consigna).

    Args:
        imgA (np.ndarray): Imagen ancla A (RGB o gris).
        imgB (np.ndarray): Imagen vecina B (RGB o gris).
        ptsA (np.ndarray): Puntos en A (N,2).
        ptsB (np.ndarray): Puntos en B (N,2).
        inliers_mask (array-like bool): Máscara de inliers (N,).
        max_draw (int): Máximo de inliers a dibujar.

    Returns:
        None
    """
    A = imgA if imgA.ndim == 3 else cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)
    B = imgB if imgB.ndim == 3 else cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)

    hA, wA = A.shape[:2]
    hB, wB = B.shape[:2]
    H = max(hA, hB)

    canvas = np.zeros((H, wA + wB, 3), dtype=A.dtype)
    canvas[:hA, :wA] = A
    canvas[:hB, wA:wA + wB] = B

    idx = np.where(np.asarray(inliers_mask).astype(bool))[0]
    if len(idx) > max_draw:
        # muestreo uniforme simple para claridad
        idx = np.linspace(0, len(idx) - 1, max_draw).astype(int)

    for i in idx:
        xA, yA = int(round(ptsA[i, 0])), int(round(ptsA[i, 1]))
        xB, yB = int(round(ptsB[i, 0])) + wA, int(round(ptsB[i, 1]))
        cv2.circle(canvas, (xA, yA), 2, (0, 255, 0), -1)
        cv2.circle(canvas, (xB, yB), 2, (0, 255, 0), -1)
        cv2.line(canvas, (xA, yA), (xB, yB), (0, 255, 0), 1)

    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    plt.title("Correspondencias inliers (RANSAC) — A | B")
    plt.axis("off")
    plt.show()


def draw_matches_inliers_only(imgA, imgB, kpsA, kpsB, matches, inliers_mask, max_draw=300):
    """
    Dibuja SOLO inliers con cv2.drawMatches (verde para matches, rojo para puntos sueltos).

    Args:
        imgA (np.ndarray): Imagen A (RGB o gris).
        imgB (np.ndarray): Imagen B (RGB o gris).
        kpsA (list of cv2.KeyPoint): Keypoints de A.
        kpsB (list of cv2.KeyPoint): Keypoints de B.
        matches (list): Lista completa de matches (cv2.DMatch o pares (i,j)).
        inliers_mask (array-like bool): Máscara de inliers (N,).
        max_draw (int): Máximo de inliers a dibujar.

    Returns:
        None
    """
    A = imgA if imgA.ndim == 3 else cv2.cvtColor(imgA, cv2.COLOR_GRAY2BGR)
    B = imgB if imgB.ndim == 3 else cv2.cvtColor(imgB, cv2.COLOR_GRAY2BGR)

    keep = []
    for m, keep_flag in zip(matches, np.asarray(inliers_mask).astype(bool)):
        if not keep_flag:
            continue
        if hasattr(m, "queryIdx"):
            keep.append(m)
        else:
            # por si viniera como par (idxA, idxB)
            keep.append(cv2.DMatch(_queryIdx=int(m[0]), _trainIdx=int(m[1]), _imgIdx=0, _distance=0))

    if len(keep) > max_draw:
        step = len(keep) / float(max_draw)
        keep = [keep[int(i * step)] for i in range(max_draw)]

    out = cv2.drawMatches(
        A, kpsA, B, kpsB, keep, None,
        matchColor=(0, 255, 0), singlePointColor=(255, 0, 0),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    plt.figure(figsize=(12, 6))
    plt.imshow(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    plt.title("Inliers (verde) — drawMatches")
    plt.axis("off")
    plt.show()


def quick_overlay(imgA, imgB, H):
    """
    Preview rápido: B warpeada sobre el tamaño de A + polígono de esquinas transformadas.

    Args:
        imgA (np.ndarray): Imagen ancla A (RGB).
        imgB (np.ndarray): Imagen B (RGB).
        H (np.ndarray): Homografía tal que A <- B.

    Returns:
        None
    """
    Hh, Ww = imgA.shape[:2]
    warpB_on_A = cv2.warpPerspective(imgB, H, (Ww, Hh))

    # Mezcla simple en la región válida
    overlay = imgA.copy()
    if imgA.ndim == 3:
        mask = (warpB_on_A.sum(axis=2) > 0)
    else:
        mask = (warpB_on_A > 0)
    alpha = 0.6
    overlay[mask] = (alpha * warpB_on_A[mask] + (1 - alpha) * overlay[mask]).astype(overlay.dtype)

    # Polígono de esquinas de B transformadas a A
    hB, wB = imgB.shape[:2]
    CB = np.array([[0, 0], [wB - 1, 0], [wB - 1, hB - 1], [0, hB - 1]], np.float32).reshape(-1, 1, 2)
    CB_to_A = cv2.perspectiveTransform(CB, H).reshape(-1, 2)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imgA)
    plt.title("Ancla A")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    xs = list(CB_to_A[:, 0]) + [CB_to_A[0, 0]]
    ys = list(CB_to_A[:, 1]) + [CB_to_A[0, 1]]
    plt.plot(xs, ys, 'y-', lw=2)
    plt.title("B warpeada sobre A (preview)")
    plt.axis("off")
    plt.show()

def show_canvas_preview(canvasA, canvasB, title="Preview en canvas óptimo (sin blending)"):
    """
    Muestra una vista previa simple en el canvas: B sobrescribe A donde existe.

    Args:
        canvasA (np.ndarray): Imagen A ya colocada en el canvas.
        canvasB (np.ndarray): Imagen B warpeada y colocada en el canvas.
        title (str): Título de la figura.

    Returns:
        None
    """
    preview = canvasA.copy()
    if preview.ndim == 3:
        maskB = (canvasB.sum(axis=2) > 0)
    else:
        maskB = (canvasB > 0)
    preview[maskB] = canvasB[maskB]

    plt.figure(figsize=(14, 6))
    plt.imshow(preview)
    plt.title(title)
    plt.axis("off")
    plt.show()

def draw_canvas_polygons(preview_img, CA_T, CB_T, title="Esquinas en canvas (verificación de bbox)"):
    """
    Dibuja los polígonos de las esquinas de A y de H·B sobre una imagen del canvas.

    Args:
        preview_img (np.ndarray): Imagen del canvas (por ejemplo, la de preview).
        CA_T (np.ndarray): Esquinas de A transformadas por T (4,2).
        CB_T (np.ndarray): Esquinas de B->A transformadas por T (4,2).
        title (str): Título de la figura.

    Returns:
        None
    """
    def _plot_poly(ax, P, color, label):
        x = list(P[:, 0]) + [P[0, 0]]
        y = list(P[:, 1]) + [P[0, 1]]
        ax.plot(x, y, color, lw=2, label=label)

    plt.figure(figsize=(14, 6))
    plt.imshow(preview_img)
    _plot_poly(plt.gca(), CA_T, 'y-', 'Esquinas A')
    _plot_poly(plt.gca(), CB_T, 'c-', 'Esquinas H·B')
    plt.legend(loc='lower right')
    plt.title(title)
    plt.axis("off")
    plt.show()

def show_no_blend(canvasA, canvasB, title="A y B en canvas (sin blending)"):
    """
    Muestra una vista previa simple: B sobrescribe A donde existe.
    
    Args:
        canvasA (np.ndarray): Imagen A ya colocada en el canvas.
        canvasB (np.ndarray): Imagen B warpeada al canvas.
        title (str): Título de la figura.
    
    Returns:
        np.ndarray: Imagen combinada sin blending (útil para comparar).
    """
    no_blend = canvasA.copy()
    if no_blend.ndim == 3:
        maskB = (canvasB.sum(axis=2) > 0)
    else:
        maskB = (canvasB > 0)
    no_blend[maskB] = canvasB[maskB]

    plt.figure(figsize=(8, 6))
    plt.imshow(no_blend)
    plt.title(title)
    plt.axis("off")
    plt.show()
    return no_blend

def show_blending_result(no_blend_img, blended_img):
    """
    Visualiza lado a lado el resultado sin blending y con blending.
    
    Args:
        no_blend_img (np.ndarray): Imagen combinada sin blending.
        blended_img (np.ndarray): Imagen final con blending por distancia.
    
    Returns:
        None
    """
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1); plt.title("Sin blending"); plt.imshow(no_blend_img); plt.axis("off")
    plt.subplot(1, 2, 2); plt.title("Blending por distancia"); plt.imshow(blended_img); plt.axis("off")
    plt.show()

def show_three_images_no_blend(canA, canB, canC, title="Canvas sin blending (A + B + C)"):
    """
    Visualiza el canvas combinando A/B/C sin blending (B y C sobrescriben A donde existan).

    Args:
        canA (np.ndarray): A en canvas.
        canB (np.ndarray): B en canvas.
        canC (np.ndarray): C en canvas.
        title (str): Título.

    Returns:
        np.ndarray: Imagen combinada sin blending.
    """
    out = canA.copy()
    mB = (canB.sum(axis=2) > 0); out[mB] = canB[mB]
    mC = (canC.sum(axis=2) > 0); out[mC] = canC[mC]

    plt.figure(figsize=(10, 6))
    plt.imshow(out)
    plt.title(title)
    plt.axis('off')
    plt.show()
    return out

def show_three_blend_results(no_blend_img, blend_img, blend_cropped):
    """
    Muestra lado a lado: sin blending, con blending, y recorte final.

    Args:
        no_blend_img (np.ndarray): Canvas sin blending.
        blend_img (np.ndarray): Resultado con blending por distancia.
        blend_cropped (np.ndarray): Resultado recortado al contenido.

    Returns:
        None
    """
    plt.figure(figsize=(18, 6))
    plt.subplot(1,3,1); plt.title("Sin blending"); plt.imshow(no_blend_img); plt.axis('off')
    plt.subplot(1,3,2); plt.title("Con blending"); plt.imshow(blend_img); plt.axis('off')
    plt.subplot(1,3,3); plt.title("Recortada"); plt.imshow(blend_cropped); plt.axis('off')
    plt.show()