import gradio as gr
import cv2
import tempfile
import traceback
import numpy as np
from ultralytics import YOLOv10

# Load model once at start
try:
    model = YOLOv10("weights/best.pt")
    print("✅ YOLOv10 model loaded successfully.")
except Exception as e:
    print("❌ Failed to load YOLOv10 model:", e)
    model = None


def yolov10_inference(image, video, model_id, image_size, conf_threshold):
    try:
        if model is None:
            raise RuntimeError("Model not loaded.")

        if image is not None:
            print("📸 Image uploaded.")
            if not isinstance(image, np.ndarray):
                image = np.array(image)

            results = model.predict(source=image, imgsz=image_size, conf=conf_threshold)
            print("🧠 Model returned:", results)

            if not results or results[0] is None:
                raise ValueError("Model returned no result.")

            annotated_image = results[0].plot()
            print("🖼️ Annotated image shape:", annotated_image.shape)

            # Convert to RGB + uint8 just to be safe
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB).astype(np.uint8)

            return annotated_image, None

        elif video is not None:
            print("🎥 Video uploaded.")
            video_path = tempfile.mktemp(suffix=".webm")
            with open(video_path, "wb") as f:
                f.write(video.read())

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            output_video_path = tempfile.mktemp(suffix=".webm")
            out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'vp80'), fps, (frame_width, frame_height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                results = model.predict(source=frame, imgsz=image_size, conf=conf_threshold)
                annotated_frame = results[0].plot()
                out.write(annotated_frame)

            cap.release()
            out.release()

            return None, output_video_path

        else:
            raise ValueError("No input provided.")

    except Exception as e:
        print("🔥 Inference error:", e)
        traceback.print_exc()
        return None, None


def yolov10_inference_for_examples(image, model_path, image_size, conf_threshold):
    annotated_image, _ = yolov10_inference(image, None, model_path, image_size, conf_threshold)
    return annotated_image


def app():
    with gr.Blocks():
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="pil", label="Image", visible=True)
                video = gr.Video(label="Video", visible=False)
                input_type = gr.Radio(
                    choices=["Image", "Video"],
                    value="Image",
                    label="Input Type",
                )
                model_id = gr.Label(value="Using weights/best.pt", label="Model (fixed)")
                image_size = gr.Slider(
                    label="Image Size",
                    minimum=320,
                    maximum=1280,
                    step=32,
                    value=640,
                )
                conf_threshold = gr.Slider(
                    label="Confidence Threshold",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=0.25,
                )
                yolov10_infer = gr.Button(value="Detect Objects")

            with gr.Column():
                output_image = gr.Image(type="numpy", label="Annotated Image", visible=True)
                output_video = gr.Video(label="Annotated Video", visible=False)

        def update_visibility(input_type):
            image = gr.update(visible=(input_type == "Image"))
            video = gr.update(visible=(input_type == "Video"))
            output_image = gr.update(visible=(input_type == "Image"))
            output_video = gr.update(visible=(input_type == "Video"))
            return image, video, output_image, output_video

        input_type.change(
            fn=update_visibility,
            inputs=[input_type],
            outputs=[image, video, output_image, output_video],
        )

        def run_inference(image, video, model_id, image_size, conf_threshold, input_type):
            return yolov10_inference(
                image if input_type == "Image" else None,
                video if input_type == "Video" else None,
                model_id,
                image_size,
                conf_threshold,
            )

        yolov10_infer.click(
            fn=run_inference,
            inputs=[image, video, model_id, image_size, conf_threshold, input_type],
            outputs=[output_image, output_video],
        )

        gr.Examples(
            examples=[
                ["ultralytics/assets/bus.jpg", "fixed", 640, 0.25],
                ["ultralytics/assets/zidane.jpg", "fixed", 640, 0.25],
            ],
            fn=yolov10_inference_for_examples,
            inputs=[image, model_id, image_size, conf_threshold],
            outputs=[output_image],
            cache_examples="lazy",
        )


gradio_app = gr.Blocks()
with gradio_app:
    gr.HTML(
        """
        <h1 style='text-align: center'>
        YOLOv10: Real-Time End-to-End Object Detection
        </h1>
        """
    )
    gr.HTML(
        """
        <h3 style='text-align: center'>
        <a href='https://arxiv.org/abs/2405.14458' target='_blank'>arXiv</a> | 
        <a href='https://github.com/THU-MIG/yolov10' target='_blank'>github</a>
        </h3>
        """
    )
    with gr.Row():
        with gr.Column():
            app()

if __name__ == '__main__':
    gradio_app.launch()