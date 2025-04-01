# import streamlit as st
# import pandas as pd
# from PIL import Image
# import io
# import base64
# from database.database import fetch_all_data, fetch_all_data_with_timestamps
# import time
# from datetime import datetime, timedelta

# st.title("DontFall Dashboard")

# # Create placeholders for the tables
# combined_placeholder = st.empty()
# accelerometer_placeholder = st.empty()
# camera_placeholder = st.empty()
# microphone_placeholder = st.empty()

# def image_to_base64(image_data):
#     buffered = io.BytesIO(image_data)
#     img = Image.open(buffered)
#     buffered = io.BytesIO()
#     img.save(buffered, format="JPEG")
#     return base64.b64encode(buffered.getvalue()).decode()

# # CSS to control the width of the tables
# st.markdown(
#     """
#     <style>
#     .dataframe {
#         width: 90% !important;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# while True:
#     # Fetch all data with timestamps from the database
#     accelerometer_data = fetch_all_data_with_timestamps('accelerometer')
#     camera_data = fetch_all_data_with_timestamps('camera')
#     microphone_data = fetch_all_data_with_timestamps('microphone')

#     combined_data = []

#     for acc_entry in accelerometer_data:
#         acc_id, acc_content, acc_timestamp = acc_entry
#         acc_time = datetime.strptime(acc_timestamp, '%Y-%m-%d %H:%M:%S')

#         # Find the latest camera entry within 2 seconds of the accelerometer timestamp
#         cam_entry = next((cam for cam in camera_data if abs(datetime.strptime(cam[2], '%Y-%m-%d %H:%M:%S') - acc_time) <= timedelta(seconds=2)), None)
#         cam_content = f'<img src="data:image/jpeg;base64,{image_to_base64(cam_entry[1])}" width="100"/>' if cam_entry else None

#         # Find the latest microphone entry within 2 seconds of the accelerometer timestamp
#         mic_entry = next((mic for mic in microphone_data if abs(datetime.strptime(mic[2], '%Y-%m-%d %H:%M:%S') - acc_time) <= timedelta(seconds=2)), None)
#         mic_content = mic_entry[1] if mic_entry else None

#         combined_data.append((acc_id, acc_content, mic_content, cam_content, acc_timestamp))

#     df_combined = pd.DataFrame(combined_data, columns=["ID", "Accelerometer Data", "Microphone Data", "Camera Data", "Timestamp"])
#     combined_placeholder.markdown(df_combined.to_html(escape=False, index=False), unsafe_allow_html=True)

#     # Display Accelerometer Data
#     with accelerometer_placeholder.container():
#         st.subheader("Accelerometer Data")
#         if accelerometer_data:
#             df_accelerometer = pd.DataFrame(accelerometer_data, columns=["ID", "Content", "Timestamp"])
#             st.markdown(df_accelerometer.to_html(escape=False, index=False), unsafe_allow_html=True)
#         else:
#             st.write("No accelerometer data received yet.")

#     # Display Camera Data
#     with camera_placeholder.container():
#         st.subheader("Camera Data")
#         if camera_data:
#             # Convert image data to base64 and embed in HTML
#             for i, row in enumerate(camera_data):
#                 image_data = row[1]
#                 base64_image = image_to_base64(image_data)
#                 camera_data[i] = (row[0], f'<img src="data:image/jpeg;base64,{base64_image}" width="100"/>', row[2])
#             df_camera = pd.DataFrame(camera_data, columns=["ID", "Content", "Timestamp"])
#             st.markdown(df_camera.to_html(escape=False, index=False), unsafe_allow_html=True)
#         else:
#             st.write("No camera data received yet.")

#     # Display Microphone Data
#     with microphone_placeholder.container():
#         st.subheader("Microphone Data")
#         if microphone_data:
#             df_microphone = pd.DataFrame(microphone_data, columns=["ID", "Content", "Timestamp"])
#             st.markdown(df_microphone.to_html(escape=False, index=False), unsafe_allow_html=True)
#         else:
#             st.write("No microphone data received yet.")

#     # Refresh every 2 seconds
#     time.sleep(2)

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

    # Determine the primary table to iterate over
    if len(microphone_data) > len(accelerometer_data):
        primary_data = microphone_data
        primary_label = "Microphone"
    else:
        primary_data = accelerometer_data
        primary_label = "Accelerometer"

    # Iterate through the primary table
    for primary_entry in primary_data:
        primary_id, primary_content, primary_timestamp = primary_entry
        primary_time = datetime.strptime(primary_timestamp, '%Y-%m-%d %H:%M:%S')

        # Find the latest camera entry within 2 seconds of the primary timestamp
        cam_entry = next((cam for cam in camera_data if abs(datetime.strptime(cam[2], '%Y-%m-%d %H:%M:%S') - primary_time) <= timedelta(seconds=2)), None)
        cam_content = f'<img src="data:image/jpeg;base64,{image_to_base64(cam_entry[1])}" width="100"/>' if cam_entry else None

        # Find the latest entry from the secondary table (opposite of the primary table) within 2 seconds
        if primary_label == "Accelerometer":
            mic_entry = next((mic for mic in microphone_data if abs(datetime.strptime(mic[2], '%Y-%m-%d %H:%M:%S') - primary_time) <= timedelta(seconds=2)), None)
            mic_content = mic_entry[1] if mic_entry else None
            acc_content = primary_content
        else:
            acc_entry = next((acc for acc in accelerometer_data if abs(datetime.strptime(acc[2], '%Y-%m-%d %H:%M:%S') - primary_time) <= timedelta(seconds=2)), None)
            acc_content = acc_entry[1] if acc_entry else None
            mic_content = primary_content

        # Append the combined data
        combined_data.append((primary_id, acc_content, mic_content, cam_content, primary_timestamp))

    # Add new combined entries to the persistent list, avoiding duplicates
    for entry in combined_data:
        if entry not in all_combined_data:
            all_combined_data.append(entry)

    # Sort the combined data by timestamp in descending order
    all_combined_data.sort(key=lambda x: datetime.strptime(x[4], '%Y-%m-%d %H:%M:%S'), reverse=True)

    # Create a DataFrame for the combined data
    df_combined = pd.DataFrame(all_combined_data, columns=["ID", "Accelerometer Data", "Microphone Data", "Camera Data", "Timestamp"])
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