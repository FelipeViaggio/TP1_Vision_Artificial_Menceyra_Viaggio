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
    coords = np.array([kp.pt for kp in keypoints], dtype=np.float32)
    responses = np.array([kp.response for kp in keypoints], dtype=np.float32)

    # Inicializar radios de supresión
    ratios = np.full(n, np.inf, dtype=np.float32)

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

def pick_points_cv2(img, n=4, win_name="Seleccionar puntos", radius=5):
    """
    Selecciona n puntos (x,y) sobre 'img'. Cierra con:
    - juntar n puntos
    - Enter / ESC / 'q'
    - cerrar la ventana (botón rojo)
    Devuelve np.array shape (k,2), dtype=int.
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
        # Refrescar ventana y drenar eventos
        cv2.imshow(win_name, disp)
        k = cv2.waitKey(20) & 0xFF

        # teclas para salir
        if k in (13, 27, ord('q')):   # Enter, ESC o 'q'
            break

        # salir si ya juntamos n puntos
        if len(pts) >= n:
            break

        # salir si el usuario cierra la ventana
        # (en macOS/Qt puede devolver < 1 cuando se cierra)
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    # soltar callback y cerrar robustamente
    cv2.setMouseCallback(win_name, lambda *args: None)
    cv2.destroyWindow(win_name)
    # En macOS conviene drenar algunos eventos extra
    for _ in range(5):
        cv2.waitKey(1)

    return np.array(pts, dtype=int)

def dlt(ori, dst):

    # Construct matrix A and vector b
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

def pick_points(img, n=4, title="Hacé click en los puntos (izq→der). Cerrar con Enter si te pasás."):
    """
    Muestra la imagen y devuelve n coordenadas (x, y) en píxeles
    según donde se hizo click. Retorna np.ndarray de shape (k, 2),
    con k = n (o menor si cerrás antes con Enter).

    Parámetros
    ----------
    img : np.ndarray
        Imagen (RGB o escala de grises). Si la tenés en BGR, convertí antes.
    n : int
        Cantidad de puntos a seleccionar.
    title : str
        Título que se muestra arriba de la imagen.

    Retorna
    -------
    pts : np.ndarray de shape (k, 2), dtype=int
        Coordenadas (x, y) enteras dentro de la imagen.
    """
    # Mostrar imagen
    plt.figure()
    if img.ndim == 2:
        plt.imshow(img, cmap='gray', origin='upper')
    else:
        # Si viene en BGR, convertir a RGB ANTES de llamar a esta función
        plt.imshow(img, origin='upper')
    plt.title(title)
    plt.axis('off')

    # ginput devuelve lista de (x, y) en coords de ejes
    clicks = plt.ginput(n=n, timeout=0, show_clicks=True)  # espera hasta n clicks
    plt.close()

    if not clicks:
        return np.empty((0, 2), dtype=int)

    # Redondear a entero y clipear por si algún click queda al borde
    H, W = img.shape[:2]
    pts = np.rint(np.array(clicks)).astype(int)
    pts[:, 0] = np.clip(pts[:, 0], 0, W - 1)  # x
    pts[:, 1] = np.clip(pts[:, 1], 0, H - 1)  # y
    return pts

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