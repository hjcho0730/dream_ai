from flask import Flask, request, jsonify, render_template
from modelLoad import modelLoad, preparePre, file_name
from func.utils import *
from func.main_func import process_data, res
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route("/")
def index():
    # index.html 파일 렌더링
    return render_template("index.html")

@app.route("/api/getResult", methods=["POST"])
def process():
    data = request.get_json()
    user_input = data.get("input", "")
    
    # 여기에 원하는 처리 로직 넣기 (AI, 수식, 텍스트 처리 등)
    s= user_input

    new_vec= process_data([s], getVec_steps)
    new_vec = toArray(new_vec)

    predictions = loadedModel.predict(new_vec)
    label =  res[int(predictions[0])]
    print(f'"{s}" → {label}')

    return jsonify({"result": label})

if __name__ == "__main__":
    fileName= file_name()
    path = os.path.join(real_path, "models", fileName)
    loadedModel= modelLoad(path=path)
    getVec_steps= preparePre()

    app.run(debug=True)
