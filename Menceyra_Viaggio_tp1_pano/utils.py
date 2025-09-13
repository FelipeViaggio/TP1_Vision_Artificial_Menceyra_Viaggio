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
    Devuelve np.array shape (k,2).
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

        # teclas para salir
        if k in (13, 27, ord('q')):  
            break

        # salir si ya juntamos n puntos
        if len(pts) >= n:
            break

        # salir si el usuario cierra la ventana
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    # soltar callback
    cv2.setMouseCallback(win_name, lambda *args: None)
    cv2.destroyWindow(win_name)
    for _ in range(5):
        cv2.waitKey(1)

    return np.array(pts, dtype=int)

def dlt(ori, dst):

    # Construir matriz A y vector b
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

    # resolvemos el sistema de ecuaciones A * h = b
    # el sistema es de 8x8, por lo que podemos resolverlo si A es inversible

    # resuelve el sistema de ecuaciones para encontrar los parámetros de H
    H = -np.linalg.solve(A, b)

    # agrega el elemento h_33
    H = np.hstack([H, [1]])

    # reorganiza H para formar la matrix en 3x3 to form the 3x3 homography matrix
    H = H.reshape(3, 3)

    return H

def show_points(img, pts, title):
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
    Mantiene m si distance(m) < ratio * distance(n).
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
    """
    ab = {(m.queryIdx, m.trainIdx) for m in matches_ab}
    ba = {(m.trainIdx, m.queryIdx) for m in matches_ba}
    inter = ab & ba
    # reconstruyo DMatch "limpio"
    return [cv2.DMatch(_queryIdx=i, _trainIdx=j, _imgIdx=0, _distance=0.0) for (i, j) in inter]

def extract_matched_points(kpsA, kpsB, matches):
    """
    Dado un conjunto de matches entre kpsA y kpsB, extrae los puntos 2D
    """
    if not matches:
        return np.empty((0,2), np.float32), np.empty((0,2), np.float32)
    ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
    ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
    return ptsA, ptsB

def match_descriptors(descA, descB, method = "bf", use_lowe= True, ratio = 0.75, do_crosscheck= False):
    """
    Empareja descriptores entre A y B con knn (k=2) + test de Lowe + cross-check opcional.
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
    """
    # Proyecta P (N,2) con H
    P1 = np.hstack([P, np.ones((P.shape[0], 1))])
    Q  = (H @ P1.T).T
    return Q[:, :2] / Q[:, 2:3]

def _sym_reproj_error(H, A, B):
    """
    Error de reproyección simétrico entre A y B con H: A <- B.
    """
    # Si H es singular/condición mala, devolvemos inf para forzar descarte
    if not np.all(np.isfinite(H)):
        return np.full(A.shape[0], np.inf)
    try:
        if np.linalg.cond(H) > 1e12:
            return np.full(A.shape[0], np.inf)
    except np.linalg.LinAlgError:
        return np.full(A.shape[0], np.inf)

    # forward B->A
    A_hat = _proj(H, B)
    e_fwd = np.linalg.norm(A_hat - A, axis=1)

    # backward A->B
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
    """
    A = np.asarray(ptsA)
    B = np.asarray(ptsB)
    assert A.shape == B.shape and A.shape[0] >= 4 and A.shape[1] == 2

    N = A.shape[0]
    rng = np.random.default_rng(random_state)

    best_H = None
    best_inliers = None
    best_n = 0

    s = 4  # tamaño de muestra mínima
    T = int(max_trials)
    trials_done = 0
    # ciclo principal de RANSAC
    while trials_done < T:
        trials_done += 1
        # muestra aleatoria sin reemplazo
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
    
    # 
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
    Devuelve las 4 esquinas de una imagen
    """
    h, w = img.shape[:2]
    return np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32).reshape(-1, 1, 2)

def _bbox_from_points(P):
    """
    Dado un conjunto de puntos P (N,1,2) o (N,2), devuelve el bounding box
    """
    P2 = P.reshape(-1, 2)
    xmin, ymin = np.floor(P2.min(axis=0))
    xmax, ymax = np.ceil(P2.max(axis=0))
    return int(xmin), int(ymin), int(xmax), int(ymax)

def _build_translation(tx, ty):
    """
    Construye matriz de traslación 3x3.
    """
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0,  1 ]], dtype=np.float64)

def compute_optimal_canvas(imgA, imgB, H_A_from_B):
    """
    Calcula el canvas óptimo que contiene A y H_A_from_B·B (todas en coords de A)
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
    """
    Wc, Hc = size
    return cv2.warpPerspective(imgA, T, (Wc, Hc))

def warp_B_to_canvas(imgB, H_A_from_B, T, size):
    """
    Warpea B con H_A_from_B y luego T: coords de canvas.
    """
    Wc, Hc = size
    H_adj = T @ H_A_from_B                  
    return cv2.warpPerspective(imgB, H_adj, (Wc, Hc)), H_adj

def compute_weights(mask_uint8, blur_ksize=0, eps=1e-6):
    """
    Calcula pesos por distancia al borde de la región válida (mask > 0).
    """
    m = (mask_uint8 > 0).astype(np.uint8)
    # distancia al borde (dentro de la región válida)
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
    """
    Hc, Wc = canvas_imgs[0].shape[:2]
    # Acumuladores
    acc_num = np.zeros((Hc, Wc, 3), dtype=np.float32)
    acc_den = np.zeros((Hc, Wc), dtype=np.float32)
    for img in canvas_imgs:
        # máscara binaria de validez
        if img.ndim == 2:
            m = (img > 0).astype(np.uint8)
            img3 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            m = (img.sum(axis=2) > 0).astype(np.uint8)
            img3 = img
        # pesos
        w = compute_weights(m, blur_ksize=0)  
        acc_num += (img3.astype(np.float32) * w[..., None])
        acc_den += w
    # evitar división por cero
    acc_den = np.clip(acc_den, 1e-6, None)
    out = (acc_num / acc_den[..., None])
    return np.clip(out, 0, 255).astype(np.uint8)

def auto_crop_nonzero(img):
    """
    Recorta automáticamente al bounding box no vacío de una imagen en canvas.
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
    """
    Wc, Hc = size
    return cv2.warpPerspective(img, H, (Wc, Hc))

def pano_blend_3(imgA, imgB, imgC, H_AB, H_AC, blur_ksize=11):
    """
    Pipeline compacto para 3 imágenes: calcula canvas, warps y blending por distancia.
    """
    # Necesita: weighted_blend y auto_crop_nonzero definidos (3.7)
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

    # weighted_blend y auto_crop_nonzero ya existen en utils (3.7)
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