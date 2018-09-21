from flask_restful import Resource,request
from models.ocr import OCRModel
import os

ocr = OCRModel()


class OCR(Resource):
    path = "/home/bishal/Projects/WebProject/OCR_API/images"
    def get(self,image):
        image_path = os.path.join(OCR.path,image)
        if os.path.exists(image_path):
            if ocr.get_text_from_image(image_path):
                return ocr.json()
            else:
                return {"message":"Can't find any optical character"}, 404
        else:
            return {"message":"Can't find the image named {}".format(image)}, 404

    def post(self,image):
        try:
            file = request.files['image']

            f = os.path.join(OCR.path, file.filename)
            file.save(f)

            return {'message':"Image uploaded successfully"}
        except:
            return {"message": "Can't upload image"}, 500

