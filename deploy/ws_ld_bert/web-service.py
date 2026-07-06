from flask import Flask, request, jsonify
from waitress import serve
from pathlib import Path
import sys

sys.path.append("..")
from web_service.src import NerTokes, Ner, NerPipe
import os

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

# относительное расположение модели
model_path = 'models/ner_dpbert_3epoch_95f1ORG_99acc.bin'
BASE_DIR = Path().cwd()
# объединяем путь к корневой директории с относительным расположением модели
model_checkpoint = BASE_DIR.joinpath(model_path)

label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-INNKPP', 'I-INNKPP', 'B-RSKS', 'I-RSKS',
              'B-STAT', 'I-STAT']
nlp_ner = NerPipe(model_checkpoint, label_list, '##', 0.8)


@app.route('/ldner', methods=['POST'])
def post():
    """    {
        "txt": текст
        "tag": "ORG",
    }.

    Ответ: {entities: []}
    """
    if request.content_type == 'application/json':
        # if True:
        try:
            r_json = request.json

            txt = r_json.get("txt")

            resp_sec = nlp_ner.get_spans(txt)
            # nlp_ner.get_span_coordinates()
            res = []
            for s in nlp_ner.spans_coord:
                teg = s[2]
                if teg == 'INNKPP':
                    teg = 'INN'
                res.append([txt[s[0]:s[1]], teg])

            print("result: ", resp_sec)
        except:
            jsonify({'entities': []}), 400
        return jsonify({'entities': res}), 200


@app.route('/ldner_positions', methods=['GET', 'POST'])
def post_positions():
    """    {
        "txt": текст
        "tag": "ORG",
    }.

    Ответ: {entities: []}
    """
    if request.content_type == 'application/json':
        # if True:
        try:
            r_json = request.json

            txt = r_json.get("txt")

            resp_sec = nlp_ner.get_spans(txt)
            # nlp_ner.get_span_coordinates()
            res = []
            for s in nlp_ner.spans_coord:
                teg = s[2]
                if teg == 'INNKPP':
                    teg = 'INN'
                res.append([txt[s[0]:s[1]], [s[0], s[1]], teg])

            print("result: ", resp_sec)
        except:
            jsonify({'entities': []}), 400
        return jsonify({'entities': res}), 200


def get_host_ip():
    """Функция получения ip адреса машины хоста."""
    import socket
    try:
        ip = socket.gethostbyname(socket.gethostname() + ".local")
    except:
        ip = socket.gethostbyname(socket.gethostname())
    return ip


if __name__ == '__main__':
    docker_run = True

    if docker_run:
        ip = get_host_ip()
        port = os.environ["NER_PORT"]
        serve(app, port=port, host=ip)
    else:
        serve(app, port=8080, host='0.0.0.0')
