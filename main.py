import cv2
import face_recognition
import os
import pickle
import time
from sklearn.tree import DecisionTreeClassifier
import numpy as np

def capturar_imagens_e_treinar(nome_pessoa, caminho_pasta='base_de_dados'):

    if not os.path.exists(caminho_pasta):
        os.makedirs(caminho_pasta)

    video_capture = cv2.VideoCapture(0)

    imagens = []
    nomes = []

    print(f"Iniciando a captura de imagens para {nome_pessoa}")
    start_time = time.time()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rostos = face_recognition.face_locations(frame)
        codificacoes = face_recognition.face_encodings(frame, rostos)

        for face_encoding in codificacoes:
            imagens.append(face_encoding)
            nomes.append(nome_pessoa)

        for (top, right, bottom, left) in rostos:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv2.imshow('Captura de Imagens', frame)

        if time.time() - start_time >= 10:
            print("10 segundos se passaram. Fechando a janela...")
            break

    video_capture.release()
    cv2.destroyAllWindows()

    with open(f'{caminho_pasta}/{nome_pessoa}.pkl', 'wb') as f:
        pickle.dump((imagens, nomes), f)

    print(f"Imagens de {nome_pessoa} salvas com sucesso!")

def treinar_arvore_decisao(caminho_pasta='base_de_dados'):
    banco_dados = []
    nomes = []

    for arquivo in os.listdir(caminho_pasta):
        if arquivo.endswith('.pkl'):
            with open(os.path.join(caminho_pasta, arquivo), 'rb') as f:
                imagens, nomes_pessoa = pickle.load(f)
                banco_dados.extend(imagens)
                nomes.extend(nomes_pessoa)

    if not banco_dados:
        print("nada")
        return None

    clf = DecisionTreeClassifier()
    clf.fit(banco_dados, nomes)
    print("Modelo treinado com sucesso!")
    return clf


def identificar_rosto(clf):
    if clf is None:
        print("Classificador não treinado!")
        return

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rostos = face_recognition.face_locations(frame)
        codificacoes = face_recognition.face_encodings(frame, rostos)

        for face_encoding, (top, right, bottom, left) in zip(codificacoes, rostos):
            name = "Desconhecido"

            try:
                # Predizer o nome usando a árvore de decisão
                name = clf.predict([face_encoding])[0]
            except:
                pass

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Identificação de Rosto', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    print("Escolha uma opção:")
    print("1. Capturar imagens e treinar modelo")
    print("2. Identificar rosto")
    opcao = input("Digite a opção desejada (1 ou 2): ")

    if opcao == '1':
        nome_pessoa = input("Digite o nome da pessoa para capturar as imagens: ")
        capturar_imagens_e_treinar(nome_pessoa)
    elif opcao == '2':
        clf = treinar_arvore_decisao()
        identificar_rosto(clf)
    else:
        print("Opção inválida!")
