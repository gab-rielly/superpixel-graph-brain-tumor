# Modelagem de Imagens Médicas por Grafos de Adjacência de Regiões (RAG) baseados em Superpixels

Este projeto propõe uma representação de imagens de ressonância magnética cerebral por meio de Grafos de Adjacência de Regiões (Region Adjacency Graphs - RAG) construídos a partir de superpixels gerados pelo algoritmo Simple Linear Iterative Clustering (SLIC).

A abordagem utiliza imagens do conjunto de dados [Dataset BraTS 2020 (Kaggle)](https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) , nas quais cada superpixel é representado como um vértice do grafo, enquanto as relações de vizinhança entre regiões adjacentes são modeladas por arestas. Essa representação permite transformar a imagem do domínio euclidiano em uma estrutura relacional baseada em grafos, preservando informações espaciais relevantes e reduzindo a complexidade da representação original.
