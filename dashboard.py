import streamlit as st
import pandas as pd
from PIL import Image
import io
import base64
from database.database import fetch_all_data, fetch_all_data_with_timestamps
import time
from datetime import datetime, timedelta

st.title("DontFall Dashboard")

# Create placeholders for the tables
combined_placeholder = st.empty()
accelerometer_placeholder = st.empty()
camera_placeholder = st.empty()
microphone_placeholder = st.empty()

def image_to_base64(image_data):
    buffered = io.BytesIO(image_data)
    img = Image.open(buffered)
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# CSS to control the width of the tables
st.markdown(
    """
    <style>
    .dataframe {
        width: 90% !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize a persistent list to store combined entries
all_combined_data = []

while True:
    # Fetch all data with timestamps from the database
    accelerometer_data = fetch_all_data_with_timestamps('accelerometer')
    camera_data = fetch_all_data_with_timestamps('camera')
    microphone_data = fetch_all_data_with_timestamps('microphone')

    combined_data = []

        # Iterate through the camera table
    for cam_entry in camera_data:
        cam_id, cam_content, cam_timestamp = cam_entry
        cam_time = datetime.strptime(cam_timestamp, '%Y-%m-%d %H:%M:%S')

        # Find the latest accelerometer entry within the time window
        acc_entry = next((acc for acc in accelerometer_data if cam_time - timedelta(seconds=20) <= datetime.strptime(acc[2], '%Y-%m-%d %H:%M:%S') <= cam_time + timedelta(seconds=5)), None)
        acc_content = acc_entry[1] if acc_entry else None

        # Find the latest microphone entry within the time window
        mic_entry = next((mic for mic in microphone_data if cam_time - timedelta(seconds=20) <= datetime.strptime(mic[2], '%Y-%m-%d %H:%M:%S') <= cam_time + timedelta(seconds=5)), None)
        mic_content = mic_entry[1] if mic_entry else None

        # Convert the camera content to base64
        cam_content_base64 = f'<img src="data:image/jpeg;base64,{image_to_base64(cam_content)}" width="100"/>' if cam_content else None

        # Only append the combined data if the camera image exists
        if cam_content_base64:
            combined_entry = (acc_content, mic_content, cam_content_base64, cam_timestamp)
            if combined_entry not in all_combined_data:  # Avoid duplicates
                all_combined_data.append(combined_entry)

    # Sort the combined data by timestamp in descending order
    all_combined_data.sort(key=lambda x: datetime.strptime(x[3], '%Y-%m-%d %H:%M:%S'), reverse=True)

    # Create a DataFrame for the combined data
    df_combined = pd.DataFrame(all_combined_data, columns=["Accelerometer Data", "Microphone Data", "Camera Data", "Timestamp"])
    combined_placeholder.markdown(df_combined.to_html(escape=False, index=False), unsafe_allow_html=True)

    # Display Accelerometer Data
    with accelerometer_placeholder.container():
        st.subheader("Accelerometer Data")
        if accelerometer_data:
            df_accelerometer = pd.DataFrame(accelerometer_data, columns=["ID", "Content", "Timestamp"])
            st.markdown(df_accelerometer.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write("No accelerometer data received yet.")

    # Display Camera Data
    with camera_placeholder.container():
        st.subheader("Camera Data")
        if camera_data:
            # Convert image data to base64 and embed in HTML
            for i, row in enumerate(camera_data):
                image_data = row[1]
                base64_image = image_to_base64(image_data)
                camera_data[i] = (row[0], f'<img src="data:image/jpeg;base64,{base64_image}" width="100"/>', row[2])
            df_camera = pd.DataFrame(camera_data, columns=["ID", "Content", "Timestamp"])
            st.markdown(df_camera.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write("No camera data received yet.")

    # Display Microphone Data
    with microphone_placeholder.container():
        st.subheader("Microphone Data")
        if microphone_data:
            df_microphone = pd.DataFrame(microphone_data, columns=["ID", "Content", "Timestamp"])
            st.markdown(df_microphone.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            st.write("No microphone data received yet.")

    # Refresh every 2 seconds
    time.sleep(2)

