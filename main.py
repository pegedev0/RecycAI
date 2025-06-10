import cv2
import numpy as np
import tensorflow as tf
import time
import os
from collections import defaultdict
from picamera2 import Picamera2
from libcamera import Transform
import RPi.GPIO as GPIO
from RPLCD import CharLCD

# Configuración inicial
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Constantes
MODEL_PATH = "/home/guille/Desktop/hackreciclaje123/Modelos/recycling_classifier_2.h5"
CLASS_NAMES = ['Biological', 'Glass', 'Paper', 'Plastic']
HISTORY_SIZE = 10
VIBRATION_TIMEOUT = 5  # segundos
DISTANCE_THRESHOLD = 20  # cm
LCD_MESSAGE = "     456765     Escanear en app!"

# Configuración de hardware
class HardwareConfig:
    def __init__(self):
        # Servos: {clase: pin}
        self.servo_pins = {0: 23, 1: 17, 2: 22, 3: 27}
        # Rangos de movimiento para cada servo (inicio, fin, paso)
        self.servo_ranges = {
            0: (70, 130, 5),   # Biological
            1: (80, 150, 5),   # Glass
            2: (85, 140, 5),   # Paper
            3: (20, 75, 5)     # Plastic
        }
        self.vibration_sensor = 10
        self.ultrasonic = {'TRIG': 16, 'ECHO': 18}
        self.lcd_config = {
            'cols': 16, 'rows': 2,
            'pin_rs': 26, 'pin_e': 19,
            'pins_data': [13, 6, 5, 11],
            'numbering_mode': GPIO.BCM
        }

class RecyclingClassifier:
    def __init__(self):
        self.config = HardwareConfig()
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.prediction_history = []
        self.lid_open = False
        self.show_camera = False
        
        self._setup_gpio()
        self._setup_camera()
        self._setup_servos()
        self.lcd = CharLCD(**self.config.lcd_config)
    
    def _setup_gpio(self):
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.config.vibration_sensor, GPIO.IN)
        GPIO.setup(self.config.ultrasonic['TRIG'], GPIO.OUT)
        GPIO.setup(self.config.ultrasonic['ECHO'], GPIO.IN)
    
    def _setup_camera(self):
        self.camera = Picamera2()
        transform = Transform(hflip=True, vflip=True)
        video_config = self.camera.create_video_configuration(
            main={"size": (640, 480), "format": "RGB888"},
            transform=transform
        )
        self.camera.configure(video_config)
    
    def _setup_servos(self):
        self.servos = {}
        for _, pin in self.config.servo_pins.items():
            GPIO.setup(pin, GPIO.OUT)
            pwm = GPIO.PWM(pin, 50)  # 50 Hz
            pwm.start(0)
            self.servos[pin] = pwm
    
    def detect_object(self):
        """Detecta objetos usando sensor ultrasónico."""
        GPIO.output(self.config.ultrasonic['TRIG'], True)
        time.sleep(0.00001)
        GPIO.output(self.config.ultrasonic['TRIG'], False)

        pulse_start = time.time()
        while GPIO.input(self.config.ultrasonic['ECHO']) == 0:
            pulse_start = time.time()

        pulse_end = time.time()
        while GPIO.input(self.config.ultrasonic['ECHO']) == 1:
            pulse_end = time.time()

        distance = (pulse_end - pulse_start) * 17150  # cm
        print(f"Distancia medida: {distance:.2f} cm")
        return distance < DISTANCE_THRESHOLD
    
    def predict_class(self, frame):
        """Clasifica un frame de imagen."""
        img = cv2.flip(frame, 1)
        img = cv2.resize(img, (180, 180))
        predictions = self.model.predict(np.expand_dims(img, axis=0), verbose=0)
        class_id = np.argmax(predictions)
        confidence = predictions[0][class_id]
        return class_id, confidence
    
    def get_final_prediction(self):
        """Obtiene la predicción más frecuente del historial."""
        if not self.prediction_history:
            return None, 0
            
        freq = defaultdict(int)
        for class_id, _ in self.prediction_history:
            freq[class_id] += 1
        
        most_common = max(freq.items(), key=lambda x: x[1])[0]
        confidences = [conf for cls, conf in self.prediction_history if cls == most_common]
        avg_confidence = sum(confidences) / len(confidences)
        return most_common, avg_confidence
    
    def move_servo(self, class_id, opening=True):
        """Controla el movimiento suave del servo."""
        pin = self.config.servo_pins[class_id]
        start, end, step = self.config.servo_ranges[class_id]
        step = step if opening else -step
        
        for duty_cycle in range(start, end, step):
            self.servos[pin].ChangeDutyCycle(duty_cycle / 10.0)
            time.sleep(0.1)
        self.servos[pin].ChangeDutyCycle(0)
    
    def open_lid(self, class_id):
        """Abre la tapa correspondiente y espera interacción."""
        if self.lid_open:
            return
            
        self.lid_open = True
        self.move_servo(class_id, opening=True)
        
        print(f"Tapa abierta para {CLASS_NAMES[class_id]}")
        self.lcd.clear()
        self.lcd.write_string(LCD_MESSAGE)
        
        # Esperar vibración o timeout
        start_time = time.time()
        while time.time() - start_time < VIBRATION_TIMEOUT:
            if GPIO.input(self.config.vibration_sensor) == 1:
                print("¡Vibración detectada!")
                break
            time.sleep(0.1)
        
        self.close_lid(class_id)
    
    def close_lid(self, class_id):
        """Cierra la tapa correspondiente."""
        self.move_servo(class_id, opening=False)
        self.lcd.clear()
        self.lid_open = False
    
    def run(self):
        """Bucle principal del programa."""
        try:
            self.camera.start()
            print("Sistema iniciado. Esperando objetos...")
            
            while True:
                if self.detect_object():
                    print("Objeto detectado")
                    self.lcd.clear()
                    self.lcd.write_string("Escaneando objeto")
                    
                    if not self.lid_open:
                        frame = self.camera.capture_array("main")
                        class_id, confidence = self.predict_class(frame)
                        
                        self.prediction_history.append((class_id, confidence))
                        if len(self.prediction_history) > HISTORY_SIZE:
                            self.prediction_history.pop(0)
                        
                        if time.time() % 3 < 0.1 and self.prediction_history:
                            final_class, final_conf = self.get_final_prediction()
                            if final_conf > 0.5:
                                print(f"Material: {CLASS_NAMES[final_class]}")
                                self.open_lid(final_class)
                                self.prediction_history = []
                else:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\nDeteniendo el sistema...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Limpia los recursos."""
        self.camera.stop()
        for pwm in self.servos.values():
            pwm.stop()
        GPIO.cleanup()
        self.lcd.clear()

if __name__ == "__main__":
    classifier = RecyclingClassifier()
    classifier.run()