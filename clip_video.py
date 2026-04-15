import cv2

def clip_video(input_path, output_path, num_frames=900):
    """
    Clip the first N frames from a video and save to a new file.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to save the clipped video
        num_frames: Number of frames to clip (default: 900)
    """
    # Open the input video
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Read and write frames
    frame_count = 0
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    print(f"Clipped {frame_count} frames and saved to {output_path}")

if __name__ == "__main__":
    input_video = "/Users/caglayantuna/larva_codes/my_codes/data/black_larva/35_cbr_mel_A1_04-06-2024-06042024031526-0000.avi"
    output_video = "/Users/caglayantuna/larva_codes/my_codes/data/black_larva/35_cbr_mel_A1_04-06-2024-06042024031526-0000_clipped.avi"    
    clip_video(input_video, output_video, num_frames=900)