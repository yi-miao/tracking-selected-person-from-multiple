from moviepy.video.io.VideoFileClip import VideoFileClip

# Load the video
video = VideoFileClip("data/vtest1.mp4")

# Resize while maintaining aspect ratio (e.g., setting width)
resized_video = video.resized(width=640)  # Height is adjusted automatically
# resized_video = video.resized(0.5)  # Reduces size by 50%

# Save the resized video
resized_video.write_videofile("output.mp4", codec="libx264", audio_codec="aac")