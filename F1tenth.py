import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import cv2

from simulateur import CircuitSimulator, find_valid_start_position, update_animation_speed

def load_map(file_path):
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        _, binary_map = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
        return binary_map

map_file = r"C:\\Users\\nasre\\Documents\\Simulateur F1tenth\\mapsimtoulouse1.jpg"
map_image = load_map(map_file)
initial_position = find_valid_start_position(map_image)
simulator = CircuitSimulator(map_image, initial_position=initial_position, initial_speed=1.0)

fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.2)
slider_speed = Slider(plt.axes([0.1, 0.05, 0.8, 0.03]), 'Vitesse Animation', 0.1, 10.0, valinit=1.0, valstep=0.1)

def update(frame):
    if not simulator.update():
        ani.event_source.stop()
    simulator.render(ax)

ani = animation.FuncAnimation(fig, update, interval=100, repeat=True)
slider_speed.on_changed(lambda val: update_animation_speed(val, simulator, ani, slider_speed))
plt.show()


  


  

