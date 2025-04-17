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
import numpy as np
import math
import time

class Vehicle:
    PIXEL_TO_METER = 0.01055

    def __init__(self, initial_position, initial_speed, map_image, simulator=None):
        self.simulator = simulator
        self.position = np.array(initial_position, dtype=np.float64)
        self.orientation = 0
        self.speed = initial_speed
        self.sensor_data = {'lidar': [], 'speed': initial_speed}
        self.trajectory = [self.position.copy()]
        self.map = map_image
        self.stop_distance = 0.5
        self.angular_speed = 0
        self.last_lidar_update = time.time()
        self.obstacle_right_detected = False
        self.obstacle_left_detected = False
        self.start_time = time.time()
        self.chronometer_running = True
        self.lap_time = None

    def move_vehicle(self, P1, P2, P3, P4, P5, P6):
        self.update_sensors()
        twist_linear = P1
        twist_angular = P2

        if self.obstacle_right_detected:
            twist_linear = P3
            twist_angular = P4
        elif self.obstacle_left_detected:
            twist_linear = P5
            twist_angular = P6

        self.speed = twist_linear
        self.angular_speed = twist_angular

        self.orientation += self.angular_speed * 0.1
        speed_pixels = self.speed / self.PIXEL_TO_METER
        dx = speed_pixels * math.cos(self.orientation) * 0.1
        dy = speed_pixels * math.sin(self.orientation) * 0.1
        self.position += np.array([dx, dy])
        self.trajectory.append(self.position.copy())

        self.check_finish_line()

        if self.is_out_of_bounds():
            print("🚗 Véhicule hors du circuit.")
            return False

        return True

    def update_sensors(self):
        current_time = time.time()
        if current_time - self.last_lidar_update >= 0.025:
            self.sensor_data['lidar'] = self.simulator.simulate_lidar()
            self.sensor_data['speed'] = self.speed
            self.last_lidar_update = current_time

            self.obstacle_right_detected = False
            self.obstacle_left_detected = False

            front_right, front_left = [], []

            for x, y, d in self.sensor_data['lidar']:
                dx, dy = x - self.position[0], y - self.position[1]
                angle = math.atan2(dy, dx)
                angle_diff = (angle - self.orientation + np.pi) % (2 * np.pi) - np.pi

                if -15 * math.pi/180 <= angle_diff <= -5 * math.pi/180:
                    front_right.append(d)
                elif 5 * math.pi/180 <= angle_diff <= 15 * math.pi/180:
                    front_left.append(d)

            if front_right:
                self.obstacle_right_detected = min(front_right) < self.stop_distance
            if front_left:
                self.obstacle_left_detected = min(front_left) < self.stop_distance

    def check_finish_line(self):
        if 598 <= self.position[0] <= 602 and 490 <= self.position[1] <= 600:
            if not self.lap_time:
                self.lap_time = time.time() - self.start_time
                print(f"🏁 Tour terminé en {self.lap_time:.2f}s")

    def is_out_of_bounds(self):
        x, y = int(self.position[0]), int(self.position[1])
        return (
            x < 0 or x >= self.map.shape[1] or
            y < 0 or y >= self.map.shape[0] or
            self.map[y, x] == 0
        )
# simulateur.py
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import time
from navigateur import Vehicle

class CircuitSimulator:
    def __init__(self, map_image, initial_position, initial_speed):
        self.map = map_image
        self.vehicle = Vehicle(initial_position, initial_speed, map_image, simulator=self)

    def simulate_lidar(self):
        max_range_pixels = int(35 / self.vehicle.PIXEL_TO_METER)
        x, y = self.vehicle.position.astype(int)

        lidar_points = []
        angles = np.linspace(0, 2 * np.pi, 360)

        for angle in angles:
            obstacle_found = False
            for d in range(1, max_range_pixels + 1):
                check_x = int(x + math.cos(angle) * d)
                check_y = int(y + math.sin(angle) * d)

                if check_x < 0 or check_x >= self.map.shape[1] or check_y < 0 or check_y >= self.map.shape[0]:
                    break

                if self.map[check_y, check_x] == 0:
                    lidar_points.append((check_x, check_y, d * self.vehicle.PIXEL_TO_METER))
                    obstacle_found = True
                    break

            if not obstacle_found:
                lidar_points.append((x + math.cos(angle) * max_range_pixels,
                                     y + math.sin(angle) * max_range_pixels,
                                     max_range_pixels * self.vehicle.PIXEL_TO_METER))

        return lidar_points

    def display_lidar_data(self):
        if not self.vehicle.sensor_data['lidar']:
            return

        closest_point = min(self.vehicle.sensor_data['lidar'], key=lambda p: p[2])
        closest_x, closest_y, closest_distance = closest_point
        angle = math.degrees(math.atan2(closest_y - self.vehicle.position[1], closest_x - self.vehicle.position[0]))

    def render(self, ax):
        ax.clear()
        ax.imshow(self.map, cmap='gray')

        ax.scatter(self.vehicle.position[0], self.vehicle.position[1], color='red', label='Véhicule', s=100)

        trajectory = np.array(self.vehicle.trajectory)
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color='blue', linewidth=2, alpha=0.6)

        if self.vehicle.sensor_data['lidar']:
            closest_point = min(self.vehicle.sensor_data['lidar'], key=lambda p: p[2])
            closest_x, closest_y, closest_distance = closest_point

            dx = closest_x - self.vehicle.position[0]
            dy = closest_y - self.vehicle.position[1]
            angle_to_closest_point = math.atan2(dy, dx)
            angle_diff = (angle_to_closest_point - self.vehicle.orientation + np.pi) % (2 * np.pi) - np.pi

            print(f"Point LIDAR le plus proche : ({closest_x:.2f}, {closest_y:.2f}) à {closest_distance:.2f} m")
            print(f"Angle entre l'avant du véhicule et l'obstacle : {math.degrees(angle_diff):.2f}°")

            ax.scatter(closest_x, closest_y, color='green', label='Point LIDAR le plus proche', s=50)

        elapsed_time = time.time() - self.vehicle.start_time if self.vehicle.chronometer_running else 0
        lap_time = self.vehicle.lap_time if self.vehicle.lap_time else "N/A"
        ax.set_title(f"Temps: {elapsed_time:.2f}s, Vitesse: {self.vehicle.speed:.2f} m/s, Temps du tour: {lap_time}")

        x1, y1 = 600, 490
        x2, y2 = 600, 600
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=2)

        ax.legend()
        ax.set_xlim(0, self.map.shape[1])
        ax.set_ylim(self.map.shape[0], 0)

    def update(self):
        vecteur = [self.vehicle.speed, 0.0, self.vehicle.speed, 1.75, self.vehicle.speed, -1.75]
        return self.vehicle.move_vehicle(*vecteur)

def find_valid_start_position(map_image):
    return np.array([600, 550])

def update_animation_speed(val, simulator, ani, slider_speed):
    speed_factor = slider_speed.val
    simulator.vehicle.speed = speed_factor
    ani.event_source.interval = 1000 / speed_factor

  


  

