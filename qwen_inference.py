import requests
from PIL import Image
import time
import io

class ImageProcessor:
    def __init__(self):
        self.api_url = 'http://192.168.29.134:4000/process/' # Local Server URL
        # self.api_url = 'http://164.52.202.253:4000/process/'

    # def process_image_from_array(self, image_array, new_size=(800, 600), text_input='', frame_name="default_image"):
    def process_image_from_array(self, image_array, new_size=(720, 1200), text_input='', frame_name="default_image"):

        # Convert numpy array to PIL Image
        img = Image.fromarray(image_array)
        
        # Resize the image
        img_resized = img.resize(new_size)
        
        # Convert the resized image to bytes
        img_byte_arr = io.BytesIO()
        img_resized.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        # Prepare the files and data for the POST request
        files = {'file': (f'{frame_name}.jpg', img_byte_arr, 'image/jpeg')}
        data = {'text': text_input}
        
        api_url_with_name = f"{self.api_url}{frame_name}"
        # Send the POST request
        st = time.time()
        response = requests.post(api_url_with_name, files=files, data=data)
        et = time.time()
        print("cloud to py:", et - st)

        # Check if the request was successful
        if response.status_code == 200:
            # Return the metadata from the response JSON
            print("from inf endpoint from cloud: ",response.json())
            return response.json().get('metadata', None)
            # return response.json().get('qwen_ocr_result',None).get('metadata', None)
        else:
            # If the request failed, return None or raise an exception
            print(f"Error: Status code {response.status_code}")
            return None
