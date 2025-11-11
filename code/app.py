from flask import Flask, request, jsonify, render_template
from modelLoad import modelLoad, preparePre, file_name
from func.utils import *
from func.main_func import process_data, res
from flask_cors import CORS
import os
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'templates'))
CORS(app)

@app.route("/")
def index():
    # index.html 파일 렌더링
    return render_template("index.html")

@app.route("/api/getResult", methods=["POST"])
def process():
    data = request.get_json()
    user_input = data.get("input", "")
    
    if not ready:
        return jsonify({"result": user_input})
    
    # 여기에 원하는 처리 로직 넣기 (AI, 수식, 텍스트 처리 등)
    s= user_input

    new_vec= process_data([s], getVec_steps)
    new_vec = toArray(new_vec)

    predictions = loadedModel.predict(new_vec)
    label =  res[int(predictions[0])]
    print(f'"{s}" → {label}')

    return jsonify({"result": label})

ready= False

fileName= file_name()
path = os.path.join(real_path, "models", fileName)
loadedModel= modelLoad(path=path)
getVec_steps= preparePre()

ready= True

app.run(host="0.0.0.0", port=5000)
