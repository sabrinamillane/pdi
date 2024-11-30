import cv2
import face_recognition
import pickle
import os


def identificar_rosto(caminho_pasta='base_de_dados'):
    banco_dados = {}
    for arquivo in os.listdir(caminho_pasta):
        if arquivo.endswith('.pkl'):
            nome_pessoa = arquivo.split('.')[0]
            with open(os.path.join(caminho_pasta, arquivo), 'rb') as f:
                imagens, nomes = pickle.load(f)
                banco_dados[nome_pessoa] = imagens

    if not banco_dados:
        print("Erro: Nenhum banco de dados encontrado!")
        return

    print("Banco de dados carregado. Iniciando identificação...")

    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        rostos = face_recognition.face_locations(frame)
        codificacoes = face_recognition.face_encodings(frame, rostos)

        for face_encoding, (top, right, bottom, left) in zip(codificacoes, rostos):
            name = "Desconhecido"

            for pessoa, codificacoes_salvas in banco_dados.items():
                matches = face_recognition.compare_faces(codificacoes_salvas, face_encoding)
                if True in matches:
                    name = pessoa
                    break

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Identificação de Rosto', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    identificar_rosto()
