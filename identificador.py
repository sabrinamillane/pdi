import cv2
import face_recognition
import os
import pickle
import time

def capturar_imagens(nome_pessoa, caminho_pasta='base_de_dados'):

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

if __name__ == '__main__':
    nome_pessoa = input("Digite o nome da pessoa para capturar as imagens: ")
    capturar_imagens(nome_pessoa)
