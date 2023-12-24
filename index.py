import io
import base64
from PIL import Image
from flask import Flask, request, jsonify
import model

app = Flask(__name__)

akshar = ''
@app.route('/api/post_string', methods=['POST'])
def post_string():
    try:
        data = request.get_json()  
        string_data = data.get('image')
        image_bytes = base64.b64decode(string_data)
        image = Image.open(io.BytesIO(image_bytes))
        global akshar 
        akshar = model.predict(image)
        return jsonify({'message': 'Image successfully processed.'})

    except Exception as e:
        return jsonify({'error': str(e)})
    
@app.route('/api/data', methods=['GET'])
def get_data():
    global akshar
    data = {
        'prediction': akshar  
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run()
    
