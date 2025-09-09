import numpy as np
import matplotlib.pyplot as plt

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