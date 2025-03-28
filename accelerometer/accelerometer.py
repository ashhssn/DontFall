from m5stack import *
from m5ui import *
from uiflow import *
import imu
import time
import math
import socket
import gc

# Initialize screen for M5StickC Plus
setScreenColor(0x000000)  # Set background color to black

# Create labels for display
title_label = M5TextBox(5, 10, "Starting...", lcd.FONT_Default, 0xFFFFFF, rotate=0)
imu_status_label = M5TextBox(5, 30, "", lcd.FONT_Default, 0xFFFFFF, rotate=0)
info_label1 = M5TextBox(5, 50, "", lcd.FONT_Default, 0xFFFFFF, rotate=0)
info_label2 = M5TextBox(5, 70, "", lcd.FONT_Default, 0xFFFFFF, rotate=0)
info_label3 = M5TextBox(5, 90, "", lcd.FONT_Default, 0xFFFFFF, rotate=0)
info_label4 = M5TextBox(5, 110, "", lcd.FONT_Default, 0xFFFFFF, rotate=0)

# Initialize IMU with error handling
try:
    imu_sensor = imu.IMU()
    has_imu = True
    imu_status_label.setText("IMU OK")
except Exception as e:
    has_imu = False
    imu_status_label.setText("IMU Error: " + str(e)[:20])

# Fall detection parameters
FALL_THRESHOLD = 2.5
MIN_FALL_DURATION = 3

# Fall detection variables
in_potential_fall = False
fall_start_time = 0
fall_sample_count = 0
last_alert_time = 0
alert_cooldown = 5000

# Neural network buffers and features
window_size = 10
acc_x_buffer = []
acc_y_buffer = []
acc_z_buffer = []
mag_buffer = []

# Button state tracking
button_a_triggered = False
button_b_triggered = False
sending_alert = False

# Data streaming variables
send_data_interval = 1000  # 1 second between data updates
last_data_send_time = 0

# Server configuration
PRIMARY_HOST = '172.20.10.4'
PRIMARY_PORT = 5001
SECONDARY_HOST = '172.20.10.9'  # Set the IP of your second host here
SECONDARY_PORT = 5001  # Set the port of your second host here

# The send_data_to_server function - shorter timeout to reduce blocking
def send_data_to_server(data_to_send, host=PRIMARY_HOST, port=PRIMARY_PORT):
    try:
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(1)  # Shorter timeout
        client_socket.connect((host, port))
        
        client_socket.send(str(data_to_send).encode('utf-8'))
        return True
    except Exception as e:
        info_label4.setText("Error: " + str(e)[:15])
        return False
    finally:
        try:
            client_socket.close()
        except:
            pass

# Function to send fall alert to both hosts
def send_fall_alert(data_to_send):
    # Send to primary host
    primary_result = send_data_to_server(data_to_send, PRIMARY_HOST, PRIMARY_PORT)
    
    # Send to secondary host
    secondary_result = send_data_to_server(data_to_send, SECONDARY_HOST, SECONDARY_PORT)
    
    # Return True if at least one send was successful
    return primary_result or secondary_result

# SimpleNN class
class SimpleNN:
    def __init__(self):
        # Pre-trained weights from SisFall dataset
        self.weights1 = [
            [-0.11481944471597672, -0.23911553621292114, -0.07101264595985413, -0.12964028120040894],
            [-0.5508483052253723, -0.7982009053230286, 0.22059978544712067, 0.2361690104007721],
            [-0.5382730960845947, -0.6271637678146362, 0.24294281005859375, 0.08300460875034332],
            [0.5596915483474731, 0.07642094790935516, -0.1889108121395111, -0.16385996341705322],
            [0.5678039193153381, -0.4196859300136566, -0.34804582595825195, -0.3356752097606659],
            [-0.045902069658041, 0.1593722403049469, 0.26741647720336914, 0.13585197925567627],
            [-0.20872995257377625, -0.42193400859832764, -0.04040568321943283, 0.1242443099617958],
        ]
        
        self.biases1 = [0.12505561113357544, -0.5355587601661682, 0.13158567249774933, 0.06646212190389633]
        self.weights2 = [1.1003971099853516, 1.147609829902649, -0.5385225415229797, -0.47447100281715393]
        self.bias2 = -0.2199278473854065
        
        self.feature_means = [0.8394059649118581, 0.07740922573501811, 0.9722356209214023, 0.7254946356513032, 0.05301188692630059, 2.1642732394463104, 0.9722356209214023]
        self.feature_stds = [0.34000559811340875, 0.16344792006733955, 0.5260441394394988, 0.3056911188290175, 0.11193020832621202, 5.828714848381604, 0.5260441394394988]
            
    def forward(self, inputs):
        try:
            # Normalize inputs
            normalized = self._normalize_inputs(inputs)
            
            # Hidden layer
            hidden = []
            for i in range(4):
                sum_val = self.biases1[i]
                for j in range(7):
                    sum_val += normalized[j] * self.weights1[j][i]
                
                # ReLU activation
                hidden.append(max(0, sum_val))
            
            # Output layer
            output_sum = self.bias2
            for i in range(4):
                output_sum += hidden[i] * self.weights2[i]
            
            # Sigmoid activation
            return 1.0 / (1.0 + math.exp(-max(-10, min(10, output_sum))))
        except Exception as e:
            return 0.0
    
    def _normalize_inputs(self, inputs):
        # Standardize inputs using the same method as training
        normalized = []
        for i in range(7):
            normalized.append((inputs[i] - self.feature_means[i]) / self.feature_stds[i])
        return normalized

# Initialize neural network
fall_nn = SimpleNN()
info_label1.setText("NN Initialized")

# Extract features function
def extract_features():
    try:
        if len(mag_buffer) < window_size:
            return None
        
        # Calculate statistical features
        avg_mag = sum(mag_buffer) / window_size
        
        # Standard deviation
        mean = avg_mag
        variance = sum((x - mean) ** 2 for x in mag_buffer) / window_size
        std_mag = variance ** 0.5
        
        # Min and max magnitude
        max_mag = max(mag_buffer)
        min_mag = min(mag_buffer)
        
        # Jerk calculation
        jerk_values = [abs(mag_buffer[i] - mag_buffer[i-1]) for i in range(1, window_size)]
        max_jerk = max(jerk_values) if jerk_values else 0
        
        # Free-fall detection
        free_fall_threshold = 0.3  # Same as training
        free_fall_duration = sum(1 for m in mag_buffer if m < free_fall_threshold)
        
        # Impact peak
        impact_peak = max(mag_buffer)
        
        # Return features in same order as training
        return [avg_mag, std_mag, max_mag, min_mag, max_jerk, free_fall_duration, impact_peak]
    except Exception as e:
        return None

# Main loop function
def main_loop_task():
    global in_potential_fall, fall_start_time, fall_sample_count, last_alert_time
    global acc_x_buffer, acc_y_buffer, acc_z_buffer, mag_buffer
    global button_a_triggered, button_b_triggered, sending_alert
    global last_data_send_time
    
    update_counter = 0
    last_a_state = False
    last_b_state = False
    title_label.setText("Starting main loop")
    
    while True:
        try:
            # Check buttons with polling - use isPressed() to get the button state
            curr_a_state = btnA.isPressed()
            curr_b_state = btnB.isPressed()
            
            # Button A - detect press event
            if curr_a_state and not last_a_state:
                button_a_triggered = True
            
            # Button B - detect press event
            if curr_b_state and not last_b_state:
                button_b_triggered = True
            
            # Update last button states
            last_a_state = curr_a_state
            last_b_state = curr_b_state
            
            # Handle Button A action (separate from detection)
            if button_a_triggered and not sending_alert:
                button_a_triggered = False
                sending_alert = True
                
                title_label.setText("Manual Trigger")
                imu_status_label.setText("")
                info_label1.setText("Sending alert...")
                info_label2.setText("")
                info_label3.setText("")
                
                # Send manual trigger alert to both hosts
                result = send_fall_alert("FALL_ALERT:MANUAL_TRIGGER")
                if result:
                    info_label2.setText("Alert sent!")
                else:
                    info_label2.setText("Send failed")
                
                sending_alert = False
            
            # Handle Button B action
            if button_b_triggered:
                button_b_triggered = False
                title_label.setText("Restarting...")
                import machine
                machine.reset()
            
            # Process IMU data if not sending alert
            if has_imu and not sending_alert:
                # Get accelerometer data
                x, y, z = imu_sensor.acceleration
                magnitude = (x**2 + y**2 + z**2)**0.5
                current_time = time.ticks_ms()
                
                # Add to sliding window buffers
                acc_x_buffer.append(x)
                acc_y_buffer.append(y)
                acc_z_buffer.append(z)
                mag_buffer.append(magnitude)
                
                # Keep buffer at window size
                while len(acc_x_buffer) > window_size:
                    acc_x_buffer.pop(0)
                    acc_y_buffer.pop(0)
                    acc_z_buffer.pop(0)
                    mag_buffer.pop(0)
                
                # Fall detection logic
                fall_confidence = 0
                features = None

                if len(mag_buffer) == window_size:
                    features = extract_features()
                    if features:
                        # Run neural network
                        fall_confidence = fall_nn.forward(features)
                        
                        # Send data to PRIMARY server periodically (regular data updates only go to primary)
                        if time.ticks_diff(current_time, last_data_send_time) > send_data_interval:
                            # Create a data string with accelerometer values and confidence
                            data_to_send = "x={} y={} z={} confidence={}".format(
                                round(x, 2),
                                round(y, 2),
                                round(z, 2),
                                round(fall_confidence, 2)
                            )
                            
                            # Try sending in a non-blocking way
                            try:
                                client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                                client_socket.settimeout(0.5)  # Short timeout
                                client_socket.connect((PRIMARY_HOST, PRIMARY_PORT))
                                client_socket.send(data_to_send.encode('utf-8'))
                                client_socket.close()
                            except:
                                # Silently fail if can't connect - don't disrupt the UI
                                pass
                            
                            # Update the last send time
                            last_data_send_time = current_time
                        
                        # Fall detection logic - simplified to just detect the fall
                        # Only trigger if cooldown period has passed
                        if time.ticks_diff(current_time, last_alert_time) > alert_cooldown:
                            # Check for fall with neural network
                            if fall_confidence > 0.8:  # Neural network threshold
                                # Show fall detection info
                                title_label.setText("Fall Detected!")
                                info_label1.setText("NN Conf: " + str(round(fall_confidence, 2)))
                                info_label2.setText("Alerting...")
                                info_label3.setText("")
                                info_label4.setText("")

                                # Send alert to both hosts
                                sending_alert = True
                                data_to_send = "FALL_DETECTED: Accelerometer Data:  x={} y={} z={} confidence={}".format(
                                    round(x, 2),
                                    round(y, 2),
                                    round(z, 2),
                                    round(fall_confidence, 2)
                                )
                                result = send_fall_alert(data_to_send)
                                if result:
                                    info_label3.setText("Alert sent!")
                                else:
                                    info_label3.setText("Send failed")
                                sending_alert = False
                                
                                # Update the last alert time
                                last_alert_time = current_time
                
                # Display updates
                if update_counter % 15 == 0:
                    if features and len(mag_buffer) == window_size:
                        title_label.setText("Neural Network")
                        info_label1.setText("X: " + str(round(x, 2)))
                        info_label2.setText("Y: " + str(round(y, 2)))
                        info_label3.setText("Z: " + str(round(z, 2)))
                        info_label4.setText("Conf: " + str(round(fall_confidence, 2)))
                    else:
                        title_label.setText("Accelerometer:")
                        info_label1.setText("X: " + str(round(x, 2)))
                        info_label2.setText("Y: " + str(round(y, 2)))
                        info_label3.setText("Z: " + str(round(z, 2)))
                        info_label4.setText("")
                
                update_counter += 1
                
                # Periodic garbage collection
                if update_counter >= 100:
                    update_counter = 0
                    gc.collect()
            
            # Shorter sleep time for more responsive button handling
            time.sleep_ms(20)
                
        except Exception as e:
            # Error recovery
            title_label.setText("Error:")
            info_label1.setText(str(e)[:20])
            time.sleep_ms(500)
            gc.collect()

# Start the main loop as a task
import _thread
_thread.start_new_thread(main_loop_task, ())