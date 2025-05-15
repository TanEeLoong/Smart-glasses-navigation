import torch
import cv2
import numpy as np
from ultralytics import YOLO
import pyttsx3
import time
import threading

# ==============================================
# Initialization Parameters
# ==============================================
# ESP32-CAM stream URL
ESP32_CAM_URL = "http://172.20.10.3:81/stream"
# Navigation parameters
det_range = 3.0  # maximum distance for object detection
min_range = 0.6  # Meters for object grouping
in_one = 20      # Ignore gaps <40% if object within 1m
in_two = 10      # Ignore gaps <20% if object within 2m
std_threshold = 5 # Default threshold

# Audio control
last_audio_time = 0
current_audio_text = ""
cooldown = 5  # seconds between audio cues
audio_thread = None

# ==============================================
# Audio Output Setup
# ==============================================
try:
    engine = pyttsx3.init()
    engine.setProperty('rate', 150)  # Speed of speech
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[0].id)  # Select default voice
    print("TTS engine initialized successfully")
except Exception as e:
    print(f"Error initializing TTS engine: {e}")
    engine = None

def speak(text):
    """Threaded speech function"""
    global audio_thread
    try:
        if engine:
            engine.say(text)
            engine.runAndWait()
    except Exception as e:
        print(f"Speech error: {e}")
    finally:
        audio_thread = Nonelast_audio_time = 0

# ==============================================
# Model Loading
# ==============================================
# Load depth estimation model
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.small_transform

# Load YOLO segmentation model
yolo_model = YOLO("yolov8n-seg.pt")

# ==============================================
# Video Capture Setup
# ==============================================
cap = cv2.VideoCapture(ESP32_CAM_URL)
if not cap.isOpened():
    raise IOError("Cannot open webcam")
cv2.namedWindow("Navigation System", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("Navigation System", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# ==============================================
# Navigation Functions
# ==============================================
def get_direction(frame_width, position, ref_center=None, ref_width=None):
    """
    Determine direction based on horizontal position, with optional reference point
    
    Args:
        frame_width: Width of the frame
        position: Center position of current space
        ref_center: Center of reference space (None for first object)
        ref_width: Width of reference space (None for first object)
    """
    # For first object, use frame divisions
    if ref_center is None:
        left_bound = frame_width / 3
        right_bound = 2 * frame_width / 3
        if position < left_bound: return "go left"
        elif position > right_bound: return "go right"
        else: return "go straight"
    
    # For subsequent objects, use reference point
    else:
        # Calculate relative position (negative means left, positive means right)
        relative_pos = position - ref_center
        
        # Use reference width to determine threshold
        threshold = ref_width / 4  # 25% of reference width
        
        if relative_pos < -threshold: return "go left"
        elif relative_pos > threshold: return "go right"
        else: return "go straight"

def generate_audio_instructions(path, objects):
    """Create spoken navigation instructions"""
    if not objects:
        return "No obstacles detected. Go straight"
    
    if not path or not path[0]["path"]:
        return "Path blocked. Please wait"
    
    instructions = []
    for i, segment in enumerate(path[0]["path"]):
        if i >= len(objects): break
        direction = segment["direction"].capitalize()
        distance = f"{objects[i]['distance']:.1f}"
        if i == 0:
            instructions.append(f"{direction} in {distance} meters")
        else:
            instructions.append(f"then {direction} in {distance} meters")
    
    return ", ".join(instructions) if instructions else "Continue straight"

# ==============================================
# Frame Processing Functions
# ==============================================
def process_frame(frame):
    """Main processing pipeline for each frame"""
    height, width = frame.shape[:2]
    
    # Visual dividers
    cv2.line(frame, (width//3, 0), (width//3, height), (255,255,255), 1)
    cv2.line(frame, (2*width//3, 0), (2*width//3, height), (255,255,255), 1)
    
    # Depth estimation
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_batch = transform(rgb).to(device)
    with torch.no_grad():
        depth = midas(input_batch)
    depth = torch.nn.functional.interpolate(
        depth.unsqueeze(1),
        size=(height, width),
        mode="bicubic",
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # Object detection
    results = yolo_model(frame)[0]
    objects = []
    for i, mask in enumerate(results.masks.data if results.masks else []):
        cls_id = int(results.boxes.cls[i])
        label = results.names[cls_id]
        box = results.boxes.xyxy[i].cpu().numpy().astype(int)
        
        # Calculate distance
        seg_mask = cv2.resize(mask.cpu().numpy().astype(np.uint8), 
                             (width, height), 
                             interpolation=cv2.INTER_NEAREST).astype(bool)
        median_depth = np.median(depth[seg_mask]) if np.any(seg_mask) else 0
        dist = max(0.1, (-0.0025 * median_depth) + 6.02)  # Clamp to min 0.1m
        
        if dist <= det_range:  # Only process nearby objects
            objects.append({
                "label": label,
                "distance": dist,
                "box": box,
                "center": (box[0] + box[2]) / 2
            })
    
    # Sort and group objects
    objects.sort(key=lambda x: x["distance"])
    groups = []
    current_group = []
    
    for obj in objects:
        if not current_group or (obj["distance"] - current_group[0]["distance"]) <= min_range:
            current_group.append(obj)
        else:
            groups.append(current_group)
            current_group = [obj]
    if current_group:
        groups.append(current_group)
    
    # Find free spaces
    free_spaces = []
    prev_spaces = None
    
    for i, group in enumerate(groups):
        coverage = np.zeros(width, dtype=np.uint8)
        for obj in group:
            x1, _, x2, _ = obj["box"]
            coverage[x1:x2+1] = 1
        
        # Find contiguous free spaces
        spaces = []
        start = None
        for x in range(width):
            if coverage[x] == 0:
                if start is None: start = x
            elif start is not None:
                spaces.append((start, x-1))
                start = None
        if start is not None:
            spaces.append((start, width-1))
        
        # Apply thresholds
        threshold = std_threshold
        if i == 0 and group:
            dist = group[0]["distance"]
            if dist <= 1.0: threshold = in_one
            elif dist <= 2.0: threshold = in_two
        
        valid_spaces = []
        for start, end in spaces:
            width_px = end - start + 1
            width_pct = (width_px / width) * 100
            if width_pct >= threshold:
                center = (start + end) / 2
                # Direction will be determined later during path finding
                valid_spaces.append({
                    "start": start, "end": end, 
                    "width": width_px, "pct": width_pct,
                    "center": center,  # Store center for later direction calculation
                    "direction": None  # Will be filled in during path finding
                })
        
        # Only keep spaces that overlap with previous group's spaces
        if i > 0 and prev_spaces:
            valid_spaces = [s for s in valid_spaces if any(
                max(s["start"], p["start"]) <= min(s["end"], p["end"])
                for p in prev_spaces
            )]
        
        if group:
            free_spaces.append({
                "distance": group[0]["distance"],
                "objects": group,
                "spaces": valid_spaces
            })
            prev_spaces = valid_spaces

    # Path finding with relative directions
    paths = []
    
    def find_paths(group_idx=0, current_path=None, current_range=None, prev_center=None, prev_width=None):
        global max_path_length  # Track the maximum path length found so far
    
        if current_path is None:
            current_path = []
            max_path_length = 0  # Reset max path length when starting a new search
    
        if group_idx >= len(free_spaces):
            if current_path and len(current_path) >= max_path_length:
                if len(current_path) > max_path_length:
                    # Clear previous paths if we found a longer one
                    paths.clear()
                    max_path_length = len(current_path)
                paths.append(current_path)
            return
    
        if group_idx == 0:
            for i, space in enumerate(free_spaces[0]["spaces"]):
                center = space["center"]
                direction = get_direction(width, center)
            
                space_with_dir = space.copy()
                space_with_dir["direction"] = direction
                free_spaces[0]["spaces"][i] = space_with_dir
            
                find_paths(1, [(0, i)], (space["start"], space["end"]), center, space["width"])
        elif current_range:
            found = False
            for i, space in enumerate(free_spaces[group_idx]["spaces"]):
                overlap_start = max(current_range[0], space["start"])
                overlap_end = min(current_range[1], space["end"])
            
                if overlap_start <= overlap_end:
                    found = True
                    center = space["center"]
                    direction = get_direction(width, center, prev_center, prev_width)
                
                    space_with_dir = space.copy()
                    space_with_dir["direction"] = direction
                    free_spaces[group_idx]["spaces"][i] = space_with_dir
                
                    find_paths(group_idx+1, current_path + [(group_idx, i)], 
                              (overlap_start, overlap_end), center, space["width"])
        
            # Only add paths that reach the current maximum length
            if not found and current_path and len(current_path) >= max_path_length:
                if len(current_path) > max_path_length:
                    paths.clear()
                    max_path_length = len(current_path)
                paths.append(current_path)
    
    find_paths()
    
    # Process paths
    processed_paths = []
    for path in paths:
        segments = []
        for group_idx, space_idx in path:
            group = free_spaces[group_idx]
            space = group["spaces"][space_idx]
            segments.append({
                "distance": group["distance"],
                "start": space["start"],
                "end": space["end"],
                "direction": space["direction"]
            })
        
        # Calculate final path width
        if segments:
            start = max(s["start"] for s in segments)
            end = min(s["end"] for s in segments)
            width = end - start + 1 if end > start else 0
            processed_paths.append({
                "path": segments,
                "width": width
            })
    
    # Sort paths by length then width
    processed_paths.sort(key=lambda x: (len(x["path"]), x["width"]), reverse=True)
    
    return objects, free_spaces, processed_paths

# ==============================================
# Main Loop
# ==============================================
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 960))
        height, width = frame.shape[:2]

        # Process frame
        objects, free_spaces, paths = process_frame(frame.copy())
        
        # Convert frame to grayscale for display (better performance)
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)  # Convert back to 3 channels for colored text
        
        # Draw bounding boxes with distance information
        for obj in objects:
            x1, y1, x2, y2 = obj["box"]
            distance = obj["distance"]
            label = obj["label"]
            
            # Draw bounding box
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 255), 2)
            
            # Create label with distance
            text = f"{label}: {distance:.1f}m"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            
            # Draw background for text
            cv2.rectangle(display, (x1, y1 - text_size[1] - 10), (x1 + text_size[0] + 10, y1), (0, 255, 255), -1)
            
            # Draw text
            cv2.putText(display, text, (x1 + 5, y1 - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Display path info
        y = 30
        if paths:
            instruction = " → ".join(
                f"{s['direction'].capitalize()} ({s['distance']:.1f}m)" 
                for s in paths[0]["path"][:3]
            )
            cv2.putText(display, f"Path: {instruction}", (10, y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            y += 30
        
        # Audio feedback
        current_time = time.time()
        if (not audio_thread or not audio_thread.is_alive()) and \
           (current_time - last_audio_time) >= cooldown:
            
            audio_text = generate_audio_instructions(paths, objects)
            if audio_text != current_audio_text:
                current_audio_text = audio_text
                last_audio_time = current_time
                audio_thread = threading.Thread(target=speak, args=(audio_text,))
                audio_thread.start()
                print(f"Speaking: {audio_text}")
        
        cv2.imshow("Navigation System", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    if engine:
        engine.stop()