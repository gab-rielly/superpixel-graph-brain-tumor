import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import networkx as nx

from skimage.segmentation import slic, mark_boundaries
from skimage.measure import regionprops


# --------------------------------------------------
# RAG
# --------------------------------------------------
def construir_rag(segmentos):

    G = nx.Graph()

    labels = np.unique(segmentos)

    for label in labels:
        G.add_node(int(label))

    linhas, colunas = segmentos.shape

    for i in range(linhas - 1):
        for j in range(colunas - 1):

            atual = segmentos[i, j]

            direita = segmentos[i, j + 1]
            baixo = segmentos[i + 1, j]

            if atual != direita:
                G.add_edge(int(atual), int(direita))

            if atual != baixo:
                G.add_edge(int(atual), int(baixo))

            # diagonais
            if i < linhas - 1 and j < colunas - 1:
                diag1 = segmentos[i + 1, j + 1]
                if atual != diag1:
                    G.add_edge(int(atual), int(diag1))

            if i < linhas - 1 and j > 0:
                diag2 = segmentos[i + 1, j - 1]
                if atual != diag2:
                    G.add_edge(int(atual), int(diag2))

    return G


# --------------------------------------------------
# MAIN PIPELINE
# --------------------------------------------------
def executar():

    pasta = r"C:\Users\gabyr\Documents\PDI\BraTS2020_TrainingData\MICCAI_BraTS2020_TrainingData"

    pacientes = sorted(glob.glob(os.path.join(pasta, "BraTS20_*")))

    pasta_paciente = pacientes[0]

    arquivos = os.listdir(pasta_paciente)

    # -------------------------
    # MRI (FLAIR)
    # -------------------------
    flair_file = [f for f in arquivos if "flair" in f.lower()][0]
    flair_path = os.path.join(pasta_paciente, flair_file)

    img = nib.load(flair_path).get_fdata()
    slice_img = np.rot90(img[:, :, 75])

    img_norm = (slice_img - np.min(slice_img)) / (np.max(slice_img) - np.min(slice_img) + 1e-8)

    # -------------------------
    # MASK (GROUND TRUTH)
    # -------------------------
    mask_file = [f for f in arquivos if "seg" in f.lower()][0]
    mask_path = os.path.join(pasta_paciente, mask_file)

    mask = nib.load(mask_path).get_fdata()
    mask_slice = np.rot90(mask[:, :, 75])

    # -------------------------
    # SLIC
    # -------------------------
    print("Executando SLIC...")

    segmentos = slic(
        img_norm,
        n_segments=600,
        compactness=0.1,
        start_label=1,
        channel_axis=None
    )

    print("Construindo RAG...")

    G = construir_rag(segmentos)

    # -------------------------
    # MAPEAR SUPERPIXEL -> TUMOR
    # -------------------------
    print("Rotulando superpixels...")

    props = regionprops(segmentos)

    labels_superpixel = {}

    pos = {}

    for regiao in props:

        coords = regiao.coords

        valores = mask_slice[coords[:, 0], coords[:, 1]]

        tumor_ratio = np.mean(valores > 0)

        label = regiao.label

        if tumor_ratio > 0.5:
            labels_superpixel[label] = "tumor"
        else:
            labels_superpixel[label] = "normal"

        # posição para plot
        y, x = regiao.centroid
        pos[label] = (x, y)

    # -------------------------
    # ESTATÍSTICAS
    # -------------------------
    grau_tumor = []
    grau_normal = []

    for node in G.nodes():
        grau = G.degree(node)

        if labels_superpixel[node] == "tumor":
            grau_tumor.append(grau)
        else:
            grau_normal.append(grau)

    print("\n===== RESULTADOS =====")

    print("Nós:", G.number_of_nodes())
    print("Arestas:", G.number_of_edges())

    print("Grau médio tumor:", np.mean(grau_tumor))
    print("Grau médio normal:", np.mean(grau_normal))

    print("Grafo conexo:", nx.is_connected(G))

    # -------------------------
    # VISUALIZAÇÃO 1: superpixels
    # -------------------------
    plt.figure(figsize=(6, 6))
    plt.imshow(mark_boundaries(img_norm, segmentos))
    plt.title("Superpixels (SLIC)")
    plt.axis("off")
    plt.show()

    # -------------------------
    # VISUALIZAÇÃO 2: RAG colorido
    # -------------------------
    plt.figure(figsize=(8, 8))
    plt.imshow(img_norm, cmap="gray")

    for u, v in G.edges():
        x1, y1 = pos[u]
        x2, y2 = pos[v]

        plt.plot([x1, x2], [y1, y2], color="gray", alpha=0.3, linewidth=0.5)

    for node in G.nodes():

        x, y = pos[node]

        if labels_superpixel[node] == "tumor":
            plt.scatter(x, y, color="red", s=15)
        else:
            plt.scatter(x, y, color="blue", s=15)

    plt.title("RAG: Tumor (vermelho) vs Normal (azul)")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    executar()