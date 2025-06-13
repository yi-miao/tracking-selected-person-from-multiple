from moviepy.video.io.VideoFileClip import VideoFileClip

video = VideoFileClip("data/lily.mp4")
# cut_video = video.subclipped(3, 8)
cut_video = video.subclipped("0:24:12", "1:07:40")
cut_video.write_videofile("vout.mp4", codec="libx264")