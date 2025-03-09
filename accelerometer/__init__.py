from m5stack import *
from m5ui import *
import time
import gc
import math

# Initialize with error handling
lcd.clear()
lcd.font(lcd.FONT_Default)
lcd.text(5, 10, "Starting...")

# Initialize IMU with error handling
try:
    import imu
    imu0 = imu.IMU()
    has_imu = True
    lcd.text(5, 30, "IMU OK")
except Exception as e:
    has_imu = False
    lcd.text(5, 30, "IMU Error: " + str(e)[:20])
time.sleep(1)

# Fall detection parameters
FALL_THRESHOLD = 2.5
IMPACT_THRESHOLD = 3.0
FREE_FALL_THRESHOLD = 0.7
MIN_FALL_DURATION = 3

# Fall detection variables
in_potential_fall = False
fall_start_time = 0
fall_sample_count = 0
last_alert_time = 0
alert_cooldown = 5000

# Neural network buffers and features
window_size = 10  # Reduced size for better performance
acc_x_buffer = []
acc_y_buffer = []
acc_z_buffer = []
mag_buffer = []

# Simple Neural Network for Fall Detection
class SimpleNN:
    def __init__(self):
        # Pre-trained weights from SisFall dataset
        self.weights1 = [
            [-0.11481944471597672, -0.23911553621292114, -0.07101264595985413, -0.12964028120040894],  # Feature 0
            [-0.5508483052253723, -0.7982009053230286, 0.22059978544712067, 0.2361690104007721],  # Feature 1
            [-0.5382730960845947, -0.6271637678146362, 0.24294281005859375, 0.08300460875034332],  # Feature 2
            [0.5596915483474731, 0.07642094790935516, -0.1889108121395111, -0.16385996341705322],  # Feature 3
            [0.5678039193153381, -0.4196859300136566, -0.34804582595825195, -0.3356752097606659],  # Feature 4
            [-0.045902069658041, 0.1593722403049469, 0.26741647720336914, 0.13585197925567627],  # Feature 5
            [-0.20872995257377625, -0.42193400859832764, -0.04040568321943283, 0.1242443099617958],  # Feature 6
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
lcd.text(5, 50, "NN Initialized")
time.sleep(0.5)

# Extract features from accelerometer buffer
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

# Main loop with error handling
update_counter = 0
lcd.clear()
lcd.text(5, 10, "Starting main loop")
time.sleep(0.5)

while True:
    try:
        if has_imu:
            # Get accelerometer data
            x, y, z = imu0.acceleration
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
            
            # Deep learning fall detection (only if we have a full buffer)
            fall_confidence = 0
            features = None
            
            if len(mag_buffer) == window_size:
                features = extract_features()
                if features:
                    # Run neural network
                    fall_confidence = fall_nn.forward(features)
                    
                    # Fall detection logic
                    if not in_potential_fall:
                        # Check for fall with neural network
                        if fall_confidence > 0.8:  # Neural network threshold
                            in_potential_fall = True
                            fall_start_time = current_time
                            fall_sample_count = 1
                            
                            # Show fall detection info
                            lcd.clear()
                            lcd.text(5, 10, "Fall Detected!")
                            lcd.text(5, 30, "NN Conf: " + str(round(fall_confidence, 2)))
                            lcd.text(5, 50, "Monitoring...")
                    else:
                        # Already tracking potential fall
                        fall_sample_count += 1
                        
                        # Look for impact
                        if magnitude > IMPACT_THRESHOLD:
                            # Minimum samples to avoid false positives
                            if fall_sample_count >= MIN_FALL_DURATION:
                                # Only trigger if cooldown period has passed
                                if time.ticks_diff(current_time, last_alert_time) > alert_cooldown:
                                    # FALL CONFIRMED
                                    lcd.clear()
                                    lcd.text(5, 10, "FALL CONFIRMED!")
                                    lcd.text(5, 30, "Conf: " + str(round(fall_confidence, 2)))
                                    lcd.text(5, 50, "Alerting...")
                                    
                                    # TODO: Send alert to Raspberry Pi 400
                                    last_alert_time = current_time
                                    time.sleep(1)  # Slight pause to show alert
                            
                            # Reset fall detection state
                            in_potential_fall = False
                        
                        # If too much time without confirmation, reset
                        if time.ticks_diff(current_time, fall_start_time) > 1000:
                            in_potential_fall = False
            
            # Display updates (less frequent for better performance)
            if update_counter % 15 == 0 and not in_potential_fall:  # Reduced frequency for more stability
                lcd.clear()
                
                # Show deep learning info if we have features
                if features and len(mag_buffer) == window_size:
                    lcd.text(5, 10, "Neural Network")
                    lcd.text(5, 30, "X: " + str(round(x, 2)))
                    lcd.text(5, 50, "Y: " + str(round(y, 2)))
                    lcd.text(5, 70, "Z: " + str(round(z, 2)))
                    lcd.text(5, 90, "Conf: " + str(round(fall_confidence, 2)))
                else:
                    # Standard display
                    lcd.text(5, 10, "Accelerometer:")
                    lcd.text(5, 30, "X: " + str(round(x, 2)))
                    lcd.text(5, 50, "Y: " + str(round(y, 2)))
                    lcd.text(5, 70, "Z: " + str(round(z, 2)))
            
            update_counter += 1
            
            # Periodic garbage collection
            if update_counter >= 100:
                update_counter = 0
                gc.collect()
        
        # Button handling
        if btnA.wasPressed():
            # Manual trigger for testing
            lcd.clear()
            lcd.text(5, 40, "Manual Trigger")
            time.sleep(1)
        
        if btnB.wasPressed():
            # Reset
            lcd.clear()
            lcd.text(5, 40, "Restarting...")
            time.sleep(1)
            import machine
            machine.reset()
        
        # Slow down loop for stability
        time.sleep(0.05)
            
    except Exception as e:
        # Error recovery
        lcd.clear()
        lcd.text(5, 10, "Error:")
        lcd.text(5, 30, str(e)[:20])
        time.sleep(2)
        gc.collect()