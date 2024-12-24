import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
import matplotlib.pyplot as plt

global polygons  # If polygons is defined globally

def generate_frames(video_path):
    generator = sv.get_video_frames_generator(video_path)
    iterator = iter(generator)
    frame = next(iterator)
    print('type of frame',type(frame))
    print('Asking user to draw polygon')
    print('changing the BGR format to RGB') 
    print('frame shape before coverting',frame.shape)
    # If frame is not in BGR format (e.g., if it's RGBA), convert it
    if frame.shape[-1] == 4:  # If RGBA
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    print('frame converted to BGR')
    print('frame shape after converting',frame.shape)
    return frame
def draw_polygon(image):
  global polygons
  fig, ax = plt.subplots(figsize=(16, 16))
  ax.imshow(image)
  plt.title("Draw a polygon by clicking on the image. Double-click to finish.")

  coords = []

  def onclick(event):
    if event.button == 1 and event.inaxes:  # Left-click within the image
      coords.append((int(event.xdata), int(event.ydata)))
      ax.plot(event.xdata, event.ydata, "ro")  # Mark the point
      if len(coords) > 1:
          x_vals = [x[0] for x in coords]
          y_vals = [y[1] for y in coords]
          ax.plot(x_vals,y_vals)
      fig.canvas.draw()
    elif event.button==3:
      plt.close()

  def finish_drawing(event):
      if event.dblclick:
          plt.close()

  fig.canvas.mpl_connect('button_press_event', onclick)
  fig.canvas.mpl_connect('button_press_event', finish_drawing)
  plt.show()

  return np.array(coords) if coords else np.array([])
def get_polygon_mask(frame, polygon_vertices):
  
    print("Ensuring correct polygon shape and data type...")
    if len(polygon_vertices.shape) != 2 or polygon_vertices.shape[1] != 2:  # Ensure shape is (N, 2)
        print("Reshaping the polygon...")
        polygon_vertices = polygon_vertices.reshape(-1, 2)
        print("Polygon reshaped.")

    polygon_vertices = polygon_vertices.astype(np.int32)
    print("Polygon converted to integer type.")

    # Create a mask for the polygon
    print("Creating mask for the polygon...")
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_vertices], 255)
    print("Mask created.")

    # Extract the region of interest (ROI) using the mask
    print("Extracting the region of interest (ROI)...")
    roi = cv2.bitwise_and(frame, frame, mask=mask)
    print("Region of interest extracted.")

    return mask, roi

def detect_objects_in_roi(roi, model_path, img_size=2304):
    model = YOLO(model_path)
    print("model loaded")
    print("Performing object detection on the ROI...")
    results = model(roi, imgsz=img_size, verbose=False)[0]
    print("Object detection completed. Processing results...")

    # Convert results to detections
    detections = sv.Detections.from_ultralytics(results)
    print("Converted results to detections.")

    return results,detections
def annotate_detections(frame, detections, polygon_vertices, labels):

    print("Annotating the detections on the frame...")

    print("Creating the zone")
    zone = sv.PolygonZone(polygon=polygon_vertices)#, frame_resolution_wh=(frame.shape[1], frame.shape[0]))
    print("Zone created")
    box_annotator =sv.BoxAnnotator(thickness=1)
    print("Box annotator created")
    label_annotator= sv.LabelAnnotator()
    print("Label annotator created")
    zone_annotator = sv.PolygonZoneAnnotator( zone=zone, color=sv.Color.RED, thickness=2, text_thickness=2, text_scale=3)
    print("Zone annotator created")

    # Annotate the frame
    mask = zone.trigger(detections=detections)
    print("Mask created")
    detections_filtered = detections[mask]
    print("Detections filtered")
    frame = box_annotator.annotate(scene=frame, detections=detections_filtered)#,skip_label=True)
    print("Frame annotated")
    frame= label_annotator.annotate(scene=frame, detections=detections, labels=labels)
    print("Frame annotated")
    frame = zone_annotator.annotate(scene=frame)
    print("Zone annotated")

    print("Detection completed, frame annotated.")  
    return frame
def process_detections_user_marked(video_path,model_path):
       # with polygons:
    print("Going to get the frame")
    frame = generate_frames(video_path)
    frame = frame.astype(np.uint8)
    polygons_array = draw_polygon(frame)
    print("Got the polygon")
    print('polygons are',polygons_array)
    print("Going to converting the polygon")
     # Ensure correct shape and data type
    if len(polygons_array.shape) != 2 or polygons_array.shape[1] != 2:  # If shape is (N, 2)
           print("Reshaping the polygon")
           polygons_array = polygons_array.reshape(-1, 2)
           print("Reshaped the polygon")
    polygons_array = polygons_array.astype(np.int32)
    print("Converted the polygon to integer type")
    # Perform object detection      
    mask, roi = get_polygon_mask(frame, polygons_array)
    print("Starting the detection")
    results,detections = detect_objects_in_roi(roi, model_path)  
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.class_id == 0]  # Filter for 'person' class
    print(f"Number of detections: {len(detections)}")

    # Create labels for the detections
    labels = [
        f"{results.names[class_id]}: {confidence:.2f}"
        for class_id, confidence in zip(detections.class_id, detections.confidence)
    ]

    # Annotate the ROI with bounding boxes
   
    # Draw the polygons on the original frame
    for polygon in polygons_array:
        polygon = polygon.reshape((-1, 1, 2))
        cv2.polylines(frame, [polygon], isClosed=False, color=(0, 0, 255), thickness=2)

    #  Annotate the frame
    frame = annotate_detections(frame, detections, polygons_array, labels)

    print("Detection completed, frame annotated.")

    sv.plot_image(frame,(16,16))
    return frame, detections

# annotated_frame, detections = process_detections_user_marked(model_path="yolov8m.pt",video_path="Sample_Video1.mp4")



def main(video_path, model_path, output_path):
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read video")
        return

    # User draws polygon on the first frame
    polygon_vertices = draw_polygon(first_frame)

    # Get video properties for saving
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create mask and perform detection
            mask, roi = get_polygon_mask(frame, polygon_vertices)
            results, detections = detect_objects_in_roi(roi, model_path)
            print('Back to main')
            detections = detections[detections.class_id == 0]  # Filter for 'person' class
            print(f"Number of detections: {len(detections)}")
            
            labels = [
                f"{results.names[class_id]}: {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]

            # Annotate frame with detections
            annotated_frame = annotate_detections(frame, detections, polygon_vertices, labels)
            
            # Write the frame instead of showing it
            
            out.write(annotated_frame)

    finally:
        cap.release()
        out.release()


if __name__ == "__main__":
    print("Starting the main function")
    main("Sample_Video2.mp4","yolov8m.pt","Brownmunday.mp4")

