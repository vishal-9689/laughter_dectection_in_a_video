import streamlit as st
import os
from moviepy.editor import VideoFileClip
from laughter_detector import detect_laughter
import datetime

UPLOAD_DIR = "uploads"
AUDIO_PATH = "temp_audio.wav"
os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("😂 Laughter Detection in Video")
st.write("Upload a video, and this app will show you when laughter occurs!")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file is not None:
    video_path = os.path.join(UPLOAD_DIR, uploaded_file.name)

    # Save video
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Video uploaded!")

    # Preview video
    st.video(video_path)

    try:
        st.info("🔊 Extracting audio...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(AUDIO_PATH, codec='pcm_s16le', verbose=False, logger=None)
        st.success("✅ Audio extracted!")

        # Detect laughter
        st.info("🔍 Detecting laughter...")
        results = detect_laughter(AUDIO_PATH)

        if results:
            st.success(f"🎉 Detected {len(results)} laughter segment(s):")
            for idx, (start, end) in enumerate(results, 1):
                start_str = str(datetime.timedelta(seconds=int(start)))
                end_str = str(datetime.timedelta(seconds=int(end)))
                st.markdown(f"**{idx}.** From `{start_str}` to `{end_str}`")
        else:
            st.warning("😐 No laughter detected in the video.")

    except Exception as e:
        st.error(f"❌ Error: {e}")
