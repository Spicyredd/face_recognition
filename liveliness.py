from gradio_client import Client
import gradio_client




class liveness_detector:
    def __init__(self, image_dir):


        self.__image_dir = image_dir  # image directory 

        self.client = Client("https://faceonlive-face-liveness-detection-sdk.hf.space/")  # API for liveness score
        self.result = self.client.predict(
		gradio_client.file(image_dir),	# filepath  in 'parameter_4' Image component
		api_name="/face_liveness"
        

)   # result from the api
        
        
        self.status = self.result['status']
        self.liveness_score = self.result['data']['liveness_score']
        self.THRESHOLD = 0.75



    def get_image(self):
        return self.__image_dir
    

    
    def check_liveliness(self):
        if self.liveness_score > self.THRESHOLD:
            print("Image is live.")

        else:
            print("Image is not live.")



session = liveness_detector("training_image.png")
session.check_liveliness()
