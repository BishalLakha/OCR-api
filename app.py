from flask import Flask
from flask_restful import Api
from resources.ocr import OCR

app = Flask(__name__)

api = Api(app)

api.add_resource(OCR,'/image/<string:image>')

if __name__ == "__main__":
    app.run(port=5000,debug=True)