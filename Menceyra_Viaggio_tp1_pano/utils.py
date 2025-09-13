import numpy as np
import matplotlib.pyplot as plt
import cv2

def anms_select(keypoints, descriptors, N=800, c_robust=1.1):
    """
    Adaptive Non-Maximal Suppression (ANMS) para seleccionar keypoints
    distribuidos uniformemente en la imagen.

    Args:
        keypoints (list of cv2.KeyPoint): Lista de keypoints detectados.
        descriptors (np.ndarray): Descriptores asociados a los keypoints.
        N (int): Número de keypoints a seleccionar.
        c_robust (float): Constante para robustez en la selección.
    
    Returns:
        selected_keypoints (list of cv2.KeyPoint): Keypoints seleccionados.
        selected_descriptors (np.ndarray): Descriptores asociados a los keypoints seleccionados.
    """
    n = len(keypoints)
    if n == 0:
        return [], None
    if n <= N:
        return keypoints, descriptors

    # Extraer coordenadas y respuestas
    coords = np.array([kp.pt for kp in keypoints])
    responses = np.array([kp.response for kp in keypoints])

    # Inicializar radios de supresión
    ratios = np.full(n, np.inf)

    for i in range(n):
        r_i = np.inf
        for j in range(n):
            if responses[j] > c_robust * responses[i]:
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < r_i:
                    r_i = dist
        ratios[i] = r_i

    # Seleccionar los N keypoints con los mayores radios
    N_effective = min(N, n)
    selected_indices = np.argsort(-ratios)[:N_effective]
    selected_keypoints = [keypoints[i] for i in selected_indices]
    selected_descriptors = descriptors[selected_indices] if descriptors is not None else None

    return selected_keypoints, selected_descriptors

def pick_points(img, n=4, win_name="Seleccionar puntos", radius=5):
    """
    Selecciona n puntos (x,y) sobre 'img'.

    Args:
        img (np.ndarray): Imagen sobre la cual seleccionar puntos.
        n (int): Número de puntos a seleccionar.
        win_name (str): Nombre de la ventana de OpenCV.
        radius (int): Radio del círculo que marca los puntos seleccionados.

    Returns:
        np.ndarray: Array de forma (n, 2) con las coordenadas (x, y) de los puntos seleccionados.
    """
    # Preparar imagen para mostrar (BGR para OpenCV)
    if img.ndim == 2:
        disp = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        disp = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    pts = []

    def mouse_cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < n:
            pts.append((x, y))
            cv2.circle(disp, (x, y), radius, (0, 255, 0), -1)

    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_cb)

    while True:
        # Refrescar ventana 
        cv2.imshow(win_name, disp)
        k = cv2.waitKey(20) & 0xFF

        # Teclas para salir
        if k in (13, 27, ord('q')):  
            break

        # Salir si ya hay n puntos
        if len(pts) >= n:
            break

        # Salir si el usuario cierra la ventana
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    # soltar callback
    cv2.setMouseCallback(win_name, lambda *args: None)
    cv2.destroyWindow(win_name)
    for _ in range(5):
        cv2.waitKey(1)

    return np.array(pts, dtype=int)

def dlt(ori, dst):
    """
    Estima la homografía H (3x3) que mapea puntos ori (origen) a dst (destino)
    usando el algoritmo Direct Linear Transform (DLT).
    
    Args:
        ori (np.ndarray): Puntos de origen de forma (4, 2).
        dst (np.ndarray): Puntos de destino de forma (4, 2).
        
    Returns:
        np.ndarray: Matriz de homografía H de forma (3, 3).
    """
    # Armado de la matriz A y el vector b
    A = []
    b = []
    for i in range(4):
        x, y = ori[i]
        x_prima, y_prima = dst[i]
        A.append([-x, -y, -1, 0, 0, 0, x * x_prima, y * x_prima])
        A.append([0, 0, 0, -x, -y, -1, x * y_prima, y * y_prima])
        b.append(x_prima)
        b.append(y_prima)

    A = np.array(A)
    b = np.array(b)

    # Resolución del sistema de ecuaciones para encontrar los parámetros de H
    H = -np.linalg.solve(A, b)

    # Agregado del elemento h_33
    H = np.hstack([H, [1]])

    # Reoganización de H para formar la matrix en 3x3
    H = H.reshape(3, 3)

    return H

def show_points(img, pts, title):
    """
    Muestra puntos (x,y) sobre la imagen img.
    
    Args:
        img (np.ndarray): Imagen sobre la cual mostrar los puntos.
        pts (np.ndarray): Array de forma (N, 2) con las coordenadas (x, y) de los puntos.
        title (str): Título de la gráfica.
        
    Returns:
        None
    """
    plt.figure(figsize=(5,5))
    if img.ndim == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)
    for i, (x,y) in enumerate(pts, start=1):
        plt.scatter([x],[y], s=30)
        plt.text(x+5, y-5, str(i))
    plt.title(title); plt.axis('off'); plt.show()

def lowe_ratio_filter(knn_matches, ratio: float = 0.75):
    """
    Aplica el test de razón de Lowe sobre knnMatches (k=2).

    Args:
        knn_matches (list of list of cv2.DMatch): Resultados de knnMatch con k=2.
        ratio (float): Umbral de la razón para filtrar matches

    Returns:
        list of cv2.DMatch: Matches que pasan el test de Lowe.
    """
    good = []
    for mn in knn_matches:
        if len(mn) < 2:
            continue
        m, n = mn
        if m.distance < ratio * n.distance:
            good.append(m)
    return good

def cross_check_filter(matches_ab, matches_ba):
    """
    Filtra matches para quedarse solo con los que son recíprocos A<->B.
    
    Args:
        matches_ab (list of cv2.DMatch): Matches de A a B.
        matches_ba (list of cv2.DMatch): Matches de B a A.

    Returns:
        list of cv2.DMatch: Matches que son recíprocos.

    """
    ab = {(m.queryIdx, m.trainIdx) for m in matches_ab}
    ba = {(m.trainIdx, m.queryIdx) for m in matches_ba}
    inter = ab & ba
    # Reconstruye DMatch "limpio"
    return [cv2.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0.0) for (i, j) in inter]

def extract_matched_points(kpsA, kpsB, matches):
    """
    Dado un conjunto de matches entre kpsA y kpsB, extrae los puntos 2D.

    Args:
        kpsA (list of cv2.KeyPoint): Keypoints de la imagen A.
        kpsB (list of cv2.KeyPoint): Keypoints de la imagen B.
        matches (list of cv2.DMatch): Matches entre kpsA y kpsB.

    Returns:
        ptsA (np.ndarray): Puntos 2D en A de forma (N, 2).
        ptsB (np.ndarray): Puntos 2D en B de forma (N, 2).
    """
    if not matches:
        return np.empty((0,2), np.float32), np.empty((0,2), np.float32)
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
    return ptsA, ptsB

def match_descriptors(descA, descB, method = "bf", use_lowe= True, ratio = 0.75, do_crosscheck= False):
    """
    Empareja descriptores entre A y B con knn (k=2) + test de Lowe + cross-check opcional.

    Args:
        descA (np.ndarray): Descriptores de la imagen A.
        descB (np.ndarray): Descriptores de la imagen B.
        method (str): "bf" para Brute-Force, "flann" para FLANN.
        use_lowe (bool): Si True, aplica el test de razón de Lowe.
        ratio (float): Umbral de la razón para el test de Lowe.
        do_crosscheck (bool): Si True, aplica cross-check entre A->B y B->A.

    Returns:
        good (list of cv2.DMatch): Lista de matches filtrados.
        dbg (dict): Diccionario con estadísticas de matching.
    """
    if method.lower() == "bf":
        matcher_ab = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matcher_ba = matcher_ab
    elif method.lower() == "flann":
        index_params = dict(algorithm=1, trees=5)  
        search_params = dict(checks=50)
        matcher_ab = cv2.FlannBasedMatcher(index_params, search_params)
        matcher_ba = matcher_ab
    else:
        raise ValueError("method must be 'bf' or 'flann'")

    knnAB = matcher_ab.knnMatch(descA, descB, k=2)
    knnBA = matcher_ba.knnMatch(descB, descA, k=2)

    goodAB = lowe_ratio_filter(knnAB, ratio=ratio) if use_lowe else [m[0] for m in knnAB if m]
    goodBA = lowe_ratio_filter(knnBA, ratio=ratio) if use_lowe else [m[0] for m in knnBA if m]

    if do_crosscheck:
        good = cross_check_filter(goodAB, goodBA)
    else:
        good = goodAB

    dbg = {
        "knnAB": len(knnAB), "knnBA": len(knnBA),
        "ratioAB": len(goodAB), "ratioBA": len(goodBA),
        "final": len(good)
    }
    return good, dbg

def _proj(H, P):
    """
    Proyecta puntos P con homografía H.

    Args:
        H (np.ndarray): Matriz de homografía de forma (3, 3).
        P (np.ndarray): Puntos 2D de forma (N, 2).

    Returns:
        np.ndarray: Puntos proyectados de forma (N, 2).
    """
    # Proyecta P (N,2) con H
    P1 = np.hstack([P, np.ones((P.shape[0], 1))])
    Q  = (H @ P1.T).T
    return Q[:, :2] / Q[:, 2:3]

def _sym_reproj_error(H, A, B):
    """
    Error de reproyección simétrico entre A y B con H: A <- B.

    Args:
        H (np.ndarray): Matriz de homografía de forma (3, 3).
        A (np.ndarray): Puntos 2D en A de forma (N, 2).
        B (np.ndarray): Puntos 2D en B de forma (N, 2).

    Returns:
        np.ndarray: Error de reproyección simétrico para cada punto, forma (N,).
    """
    # Si H es singular/condición mala, devuelve inf para forzar descarte
    if not np.all(np.isfinite(H)):
        return np.full(A.shape[0], np.inf)
    try:
        if np.linalg.cond(H) > 1e12:
            return np.full(A.shape[0], np.inf)
    except np.linalg.LinAlgError:
        return np.full(A.shape[0], np.inf)

    # Forward B->A
    A_hat = _proj(H, B)
    e_fwd = np.linalg.norm(A_hat - A, axis=1)

    # Backward A->B
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(A.shape[0], np.inf)
    B_hat = _proj(Hinv, A)
    e_bwd = np.linalg.norm(B_hat - B, axis=1)

    return e_fwd + e_bwd

def _degenerate(pts):
    """
    Chequea si un conjunto de puntos es degenerado (colineal o casi).

    Args:
        pts (np.ndarray): Puntos 2D de forma (N, 2).

    Returns:
        bool: True si los puntos son degenerados, False en caso contrario.
    """
    # Evitar 4 puntos casi colineales 
    if pts.shape[0] < 3:
        return True
    x, y = pts[:, 0], pts[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area < 1e-3

def ransac_homography(ptsA, ptsB, thresh=3.0, max_trials=2000, confidence=0.995, random_state=42, refine="opencv"):
    """
    Estima homografía robusta con RANSAC y DLT.

    Args:
        ptsA (np.ndarray): Puntos 2D en A de forma (N, 2).
        ptsB (np.ndarray): Puntos 2D en B de forma (N, 2).
        thresh (float): Umbral de reproyección para considerar inliers.
        max_trials (int): Número máximo de iteraciones RANSAC.
        confidence (float): Confianza deseada para ajustar el número de iteraciones.
        random_state (int): Semilla para el generador de números aleatorios.
        refine (str): Método de refinamiento final: "dlt" o "opencv".

    Returns:
        H (np.ndarray): Matriz de homografía estimada de forma (3, 3).
        inliers (np.ndarray): Máscara booleana de inliers de forma (N,).
    """
    A = np.asarray(ptsA)
    B = np.asarray(ptsB)
    assert A.shape == B.shape and A.shape[0] >= 4 and A.shape[1] == 2

    N = A.shape[0]
    rng = np.random.default_rng(random_state)

    best_H = None
    best_inliers = None
    best_n = 0

    s = 4  # Tamaño de muestra mínima
    T = int(max_trials)
    trials_done = 0

    while trials_done < T:
        trials_done += 1
        # Muestra aleatoria sin reemplazo
        idx = rng.choice(N, size=s, replace=False)
        if _degenerate(A[idx]) or _degenerate(B[idx]):
            continue

        # Modelo: H = A <- B (origen B; destino A)
        try:
            H = dlt(B[idx], A[idx])  
        except Exception:
            continue
        # Evaluar modelo
        err = _sym_reproj_error(H, A, B)
        if not np.all(np.isfinite(err)):
            continue
        # Inliers
        inliers = err < thresh
        ninl = int(inliers.sum())
        # Actualizar mejor modelo
        if ninl > best_n:
            best_n = ninl
            best_inliers = inliers
            best_H = H

            w = ninl / N
            w = min(max(w, 1e-6), 1 - 1e-6)
            need = np.log(1 - confidence) / np.log(1 - w**s)
            T = int(min(T, max(100, np.ceil(need))))
    
    if best_inliers is None or best_n < 4:
        raise RuntimeError("RANSAC no encontró modelo.")

    # Refinar con TODOS los inliers 
    A_in = A[best_inliers]
    B_in = B[best_inliers]

    if refine == "dlt":
        H_ref = dlt(A_in, B_in)
    else:
        H_ref, _ = cv2.findHomography(B_in, A_in, method=0)

    return H_ref, best_inliers

def _corners(img):
    """
    Devuelve las 4 esquinas de una imagen.

    Args:
        img (np.ndarray): Imagen de entrada.

    Returns:
        np.ndarray: Array de forma (4, 1, 2) con las coordenadas de las esquinas.
    """
    h, w = img.shape[:2]
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

def _bbox_from_points(P):
    """
    Dado un conjunto de puntos P (N,1,2) o (N,2), devuelve el bounding box.

    Args:
        P (np.ndarray): Puntos 2D de forma (N, 1, 2) o (N, 2).

    Returns:
        (xmin, ymin, xmax, ymax) (tuple of int): Coordenadas del bounding box.
    """
    P2 = P.reshape(-1, 2)
    xmin, ymin = np.floor(P2.min(axis=0))
    xmax, ymax = np.ceil(P2.max(axis=0))
    return int(xmin), int(ymin), int(xmax), int(ymax)

def _build_translation(tx, ty):
    """
    Construye matriz de traslación 3x3.

    Args:
        tx (float): Traslación en x.
        ty (float): Traslación en y.

    Returns:
        np.ndarray: Matriz de traslación de forma (3, 3).
    """
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0,  1 ]], dtype=np.float64)

def compute_optimal_canvas(imgA, imgB, H_A_from_B):
    """
    Calcula el canvas óptimo que contiene A y H_A_from_B·B (todas en coords de A).

    Args:
        imgA (np.ndarray): Imagen A.
        imgB (np.ndarray): Imagen B.
        H_A_from_B (np.ndarray): Homografía que mapea B a A de forma (3, 3).

    Returns:
        T (np.ndarray): Matriz de traslación para el canvas de forma (3, 3).
        size (tuple of int): Tamaño del canvas (Wc, Hc).
        CA (np.ndarray): Esquinas de A en coords de A, forma (4, 1, 2).
        CB_A (np.ndarray): Esquinas de B en coords de A, forma (4, 1, 2).
        bbox (tuple of int): Bounding box (xmin, ymin, xmax, ymax) en coords de A.
    """
    CA = _corners(imgA)                                    # esquinas de A en coords A
    CB = _corners(imgB)                                    # esquinas de B
    CB_A = cv2.perspectiveTransform(CB, H_A_from_B)        # B -> A

    # Juntar y sacar bbox
    allP = np.vstack([CA, CB_A])                          
    xmin, ymin, xmax, ymax = _bbox_from_points(allP)

    # Traslación para evitar coords negativas
    tx = -xmin if xmin < 0 else 0
    ty = -ymin if ymin < 0 else 0
    T = _build_translation(tx, ty)

    # Tamaño final del canvas
    Wc = int(xmax + tx)
    Hc = int(ymax + ty)
    return T, (Wc, Hc), CA, CB_A, (xmin, ymin, xmax, ymax)

def place_A_on_canvas(imgA, T, size):
    """
    Warpea A con traslación T: coords de canvas.

    Args:
        imgA (np.ndarray): Imagen A.
        T (np.ndarray): Matriz de traslación de forma (3, 3).
        size (tuple of int): Tamaño del canvas (Wc, Hc).

    Returns:
        np.ndarray: Imagen A warpeada en el canvas.
    """
    Wc, Hc = size
    return cv2.warpPerspective(imgA, T, (Wc, Hc))

def warp_B_to_canvas(imgB, H_A_from_B, T, size):
    """
    Warpea B con H_A_from_B y luego T: coords de canvas.
    
    Args:
        imgB (np.ndarray): Imagen B.
        H_A_from_B (np.ndarray): Homografía que mapea B a A de forma (3, 3).
        T (np.ndarray): Matriz de traslación de forma (3, 3).
        size (tuple of int): Tamaño del canvas (Wc, Hc).

    Returns:
        warped_imgB (np.ndarray): Imagen B warpeada en el canvas.
        H_adj (np.ndarray): Homografía ajustada T @ H_A_from_B de forma (3, 3).
    """
    Wc, Hc = size
    H_adj = T @ H_A_from_B                  
    return cv2.warpPerspective(imgB, H_adj, (Wc, Hc)), H_adj

def compute_weights(mask_uint8, blur_ksize=0, eps=1e-6):
    """
    Calcula pesos por distancia al borde de la región válida (mask > 0).

    Args:
        mask_uint8 (np.ndarray): Máscara binaria de validez (uint8).
        blur_ksize (int): Tamaño del kernel de GaussianBlur (debe ser impar).
        eps (float): Pequeña constante para evitar división por cero.

    Returns:
        np.ndarray: Pesos normalizados de forma (H, W) en float32.
    """
    m = (mask_uint8 > 0).astype(np.uint8)
    # Distancia al borde (dentro de la región válida)
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    if blur_ksize and blur_ksize > 1:
        dist = cv2.GaussianBlur(dist, (blur_ksize, blur_ksize), 0)
    mx = float(dist.max())
    if mx < eps:
        return dist.astype(np.float32) 
    return (dist / (mx + eps)).astype(np.float32)

def weighted_blend(canvas_imgs):
    """
    Hace blending canal a canal con pesos de distanceTransform.

    Args:
        canvas_imgs (list of np.ndarray): Lista de imágenes en el canvas (Hc, Wc, 3) o (Hc, Wc).

    Returns:
        np.ndarray: Imagen resultante del blending de forma (Hc, Wc, 3).
    """
    Hc, Wc = canvas_imgs[0].shape[:2]
    # Acumuladores
    acc_num = np.zeros((Hc, Wc, 3), dtype=np.float32)
    acc_den = np.zeros((Hc, Wc), dtype=np.float32)
    for img in canvas_imgs:
        # Máscara binaria
        if img.ndim == 2:
            m = (img > 0).astype(np.uint8)
            img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            m = (img.sum(axis=2) > 0).astype(np.uint8)
            img3 = img
        # Pesos
        w = compute_weights(m, blur_ksize=0)  
        acc_num += (img3.astype(np.float32) * w[..., None])
        acc_den += w
    # Evitar división por cero
    acc_den = np.clip(acc_den, 1e-6, None)
    out = (acc_num / acc_den[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)

def auto_crop_nonzero(img):
    """
    Recorta automáticamente al bounding box no vacío de una imagen en canvas.

    Args:
        img (np.ndarray): Imagen en canvas (Hc, Wc, 3) o (Hc, Wc).

    Returns:
        np.ndarray: Imagen recortada al bounding box no vacío.
    """
    if img.ndim == 2:
        m = img > 0
    else:
        m = img.sum(axis=2) > 0
    ys, xs = np.where(m)
    if len(ys) == 0:
        return img
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return img[y0:y1, x0:x1]

def compute_optimal_canvas_3(imgA, imgB, imgC, H_AB, H_AC):
    """
    Calcula el canvas óptimo que contiene A, H_AB·B y H_AC·C (todas en coords de A).

    Args:
        imgA (np.ndarray): Imagen A.
        imgB (np.ndarray): Imagen B.
        imgC (np.ndarray): Imagen C.
        H_AB (np.ndarray): Homografía que mapea B a A de forma (3, 3).
        H_AC (np.ndarray): Homografía que mapea C a A de forma (3, 3).

    Returns:
        T (np.ndarray): Matriz de traslación para el canvas de forma (3, 3).
        size (tuple of int): Tamaño del canvas (Wc, Hc).
    """
    # Esquinas de A, B y C en coords de A
    CA   = _corners(imgA)
    CB_A = cv2.perspectiveTransform(_corners(imgB), H_AB)
    CC_A = cv2.perspectiveTransform(_corners(imgC), H_AC)
    # Juntar y sacar bbox
    allP = np.vstack([CA, CB_A, CC_A])
    xmin, ymin, xmax, ymax = _bbox_from_points(allP)
    # Traslación para evitar coords negativas
    tx = -xmin if xmin < 0 else 0
    ty = -ymin if ymin < 0 else 0
    T  = _build_translation(tx, ty)

    Wc, Hc = int(xmax + tx), int(ymax + ty)
    return T, (Wc, Hc)

def place_on_canvas(img, H, size):
    """
    Warpea img con H: coords de canvas.

    Args:
        img (np.ndarray): Imagen de entrada.
        H (np.ndarray): Homografía de forma (3, 3).
        size (tuple of int): Tamaño del canvas (Wc, Hc).

    Returns:
        np.ndarray: Imagen warpeada en el canvas.
    """
    Wc, Hc = size
    return cv2.warpPerspective(img, H, (Wc, Hc))

def pano_blend_3(imgA, imgB, imgC, H_AB, H_AC):
    """
    Pipeline compacto para 3 imágenes: calcula canvas, warps y blending por distancia.

    Args:
        imgA (np.ndarray): Imagen A.
        imgB (np.ndarray): Imagen B.
        imgC (np.ndarray): Imagen C.
        H_AB (np.ndarray): Homografía que mapea B a A de forma (3, 3).
        H_AC (np.ndarray): Homografía que mapea C a A de forma (3, 3).

    Returns:
        dict: Diccionario con resultados intermedios y finales.
    """
    T, size = compute_optimal_canvas_3(imgA, imgB, imgC, H_AB, H_AC)
    H_A   = T
    H_Bad = T @ H_AB
    H_Cad = T @ H_AC

    canA = place_on_canvas(imgA, H_A,   size)
    canB = place_on_canvas(imgB, H_Bad, size)
    canC = place_on_canvas(imgC, H_Cad, size)

    no_blend = canA.copy()
    mB = (canB.sum(axis=2) > 0); no_blend[mB] = canB[mB]
    mC = (canC.sum(axis=2) > 0); no_blend[mC] = canC[mC]

    pano = weighted_blend([canA, canB, canC])
    pano_crop = auto_crop_nonzero(pano)

    return {
        'T': T, 'size': size,
        'H_A': H_A, 'H_Bad': H_Bad, 'H_Cad': H_Cad,
        'canA': canA, 'canB': canB, 'canC': canC,
        'no_blend': no_blend,
        'blend': pano,
        'blend_cropped': pano_crop
    }