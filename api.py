from fastapi import FastAPI, WebSocket, BackgroundTasks
from fastapi.responses import StreamingResponse, HTMLResponse
import threading
import time
import cv2
import uvicorn
import asyncio
import httpx  # Import httpx for sending HTTP requests
# import socketio
# from fastapi_socketio import SocketManager
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from inf import CameraController  # Import your CameraController class
from qwen_inference import ImageProcessor
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()

# Initialize Socket.IO and mount it to the FastAPI app
# sio = socketio.AsyncServer(
#     async_mode='asgi',
#     cors_allowed_origins=['http://localhost:3000'],
#     )
# socket_manager = SocketManager(app=app, socketio_path='/ws/socket.io')
# app.mount('/ws', socket_manager)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to the specific origins you want to allow (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

qwen = ImageProcessor()
# Initialize CameraController object
camera = CameraController()
camera.initialize()
camera.set_exposure_time(47490)
camera.set_gain_value(11.0)

# Flag to check for new metadata
new_metadata_available = False
metadata_lock = threading.Lock()  # Lock for thread-safe operations
latest_metadata = None  # Variable to hold the latest metadata

# Create a background thread to continuously update frames
streaming_thread = threading.Thread(target=camera.stream, args=(30, 1), daemon=True)
streaming_thread.start()

executor = ThreadPoolExecutor()

@app.get("/video")
async def video_feed():
    """Stream video frames in real-time using MJPEG format."""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")


async def generate_frames():
    """Generator function to capture frames and encode them to JPEG format."""
    while True:
        if camera.raw_image is not None:
            with camera.lock:
                # Convert the latest frame to BGR format (OpenCV)
                image_bgr = cv2.cvtColor(camera.raw_image[438:1638, 944:2144], cv2.COLOR_RGB2BGR)
                # Resize or perform any preprocessing as needed
                _, jpeg_frame = cv2.imencode('.jpg', image_bgr)
                frame = jpeg_frame.tobytes()

            # Yield frame in the format expected by MJPEG stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

        # Small delay to prevent CPU overload
        await asyncio.sleep(0.05)


@app.post("/capture-frame/")
async def capture_frame(background_tasks: BackgroundTasks):
    """API endpoint to capture a frame and process metadata."""
    global new_metadata_available, latest_metadata
    with camera.lock:
        if camera.raw_image is not None:
            # Crop the image to the ROI
            cropped_image = camera.raw_image[438:1638, 944:2144]

            # Get current datetime and format it for the filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            frame_name = f"CAP_{timestamp}.jpg"

            # Schedule the image processing in the background to prevent blocking the main thread
            background_tasks.add_task(process_image_task, cropped_image, frame_name=frame_name)
            print(frame_name)
            return {"status": "success", "message": "Image captured and being processed."}
        else:
            return {"status": "failure", "message": "No image data available."}
            
            # # Use qwen to process the cropped image
            # metadata = qwen.process_image_from_array(cropped_image, frame_name=frame_name)
            
            # # Add metadata to the dataframe
            # camera.add_to_dataframe(metadata)
            
            # # Save the dataframe to CSV
            # if camera.df is not None and not camera.df.empty:
            #     csv_save_path = "/home/neojetson/Projects/Main_Pipeline/results.csv"
            #     camera.df.to_csv(csv_save_path, index=False)
            #     print(f"CSV file saved successfully at: {csv_save_path}")
            # else:
            #     return {"status": "failure", "message": "DataFrame is empty or not defined. Cannot save CSV."}
            
            # Return the metadata as a response
            # print("sending metadata to front end:", metadata)
        #     return {"status": "success", "metadata": metadata}
        # else:
        #     return {"status": "failure", "message": "No image data available."}

# @sio.on("metadata")
# async def send_metadata_to_socket(metadata):
#     """Send metadata to all connected clients via Socket.IO."""
#     print("Sending metadata to frontend:", metadata)
#     await sio.emit('metadata', {'data': metadata})


def process_image_task(cropped_image, frame_name):
    """Function to process image and save metadata. This runs in a separate thread."""
    global new_metadata_available, latest_metadata
    # Use qwen to process the cropped image
    metadata = qwen.process_image_from_array(cropped_image, frame_name=frame_name)

    # Acquire lock and update the metadata
    with metadata_lock:
        latest_metadata = metadata
        new_metadata_available = True  # Set flag to indicate new metadata is available

    # Automatically send the metadata to the Node.js server
    st = time.time()
    asyncio.run(send_metadata_to_node(metadata))
    et = time.time()
    print("py to node time:", et - st)

async def send_metadata_to_node(metadata):
    """Send the latest metadata to the Node.js server."""
    node_server_url = "http://localhost:5000/api/metadata"  # Update with your Node.js server URL and endpoint
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(node_server_url, json={"metadata": metadata})
            if response.status_code == 200:
                print("Metadata successfully sent to Node.js server")
            else:
                print(f"Failed to send metadata to Node.js server: {response.status_code}")
        except Exception as e:
            print(f"Error sending metadata to Node.js server: {str(e)}")

    
    # Add metadata to the dataframe
    # camera.add_to_dataframe(metadata)
    
    # # Save the dataframe to CSV
    # if camera.df is not None and not camera.df.empty:
    #     csv_save_path = "/home/neojetson/Projects/Main_Pipeline/results.csv"
    #     camera.df.to_csv(csv_save_path, index=False)
    #     print(f"CSV file saved successfully at: {csv_save_path}")
    # else:
    #     print("DataFrame is empty or not defined. Cannot save CSV.")
    
    # Use asyncio to run the send_metadata_to_socket function in the main event loop
    # asyncio.run(send_metadata_to_socket(metadata))
    # Instead of using asyncio.run, we'll use the event loop associated with the main thread
    # loop = asyncio.get_event_loop()
    # loop.call_soon_threadsafe(asyncio.create_task, send_metadata_to_socket(metadata))

    # return metadata


# Handle connection and disconnection events
# @sio.event
# async def connect(sid, environ):
#     print("Client connected:", sid)
#     await sio.emit('connect', {'data': 'Connected successfully!'}, to=sid)


# @sio.event
# async def disconnect(sid):
#     print("Client disconnected:", sid)
        

if __name__ == "__main__":
    # Run FastAPI server with Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)