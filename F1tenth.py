import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.animation as animation
from matplotlib.widgets import Slider
import cv2
import time

def load_map(file_path):
    """Charge la carte du circuit depuis un fichier image ou .npy."""
    if file_path.endswith('.npy'):
        return np.load(file_path)
    else:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        _, binary_map = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
        return binary_map

class Vehicle:
    PIXEL_TO_METER = 0.01055  # Conversion pixels en mètres

    def __init__(self, initial_position, initial_speed, map_image):
        self.position = np.array(initial_position, dtype=np.float64)
        self.orientation = 0
        self.speed = initial_speed  # Vitesse initiale
        self.sensor_data = {'lidar': [], 'speed': initial_speed}
        self.trajectory = [self.position.copy()]
        self.map = map_image
        self.stop_distance = 0.5
        self.angular_speed = 0
        self.last_lidar_update = time.time()
        self.obstacle_right_detected = False
        self.obstacle_left_detected = False
        self.start_position = self.position.copy()  # Position de départ
        self.start_time = time.time()  # Démarrer immédiatement le chronomètre
        self.lap_time = None  # Temps du tour
        self.chronometer_running = True  # Chronomètre est actif dès le début
        self.first_lap = True  # Indicateur pour savoir si le véhicule a effectué un premier tour

    def move_vehicle(self, P1, P2, P3, P4, P5, P6):
        """Déplace le véhicule en fonction des données LIDAR et des nouveaux paramètres P1 à P6."""
        self.update_sensors()

        # Initialiser les vitesses linéaires et angulaires par défaut avec P1 et P2
        twist_linear = P1  # Vitesse linéaire (par exemple, P1)
        twist_angular = P2  # Vitesse angulaire (par exemple, P2)

        # Si un obstacle est détecté à droite, on ajuste les vitesses avec P3 et P4
        if self.obstacle_right_detected:
            twist_linear = P3
            twist_angular = P4
            print("🔄 Obstacle à droite, tourne à gauche.")

        # Si un obstacle est détecté à gauche, on ajuste les vitesses avec P5 et P6
        elif self.obstacle_left_detected:
            twist_linear = P5
            twist_angular = P6
            print("🔄 Obstacle à gauche, tourne à droite.")

        self.speed = twist_linear
        self.angular_speed = twist_angular

        # Mise à jour de la position du véhicule
        self.orientation += self.angular_speed * 0.1
        speed_pixels = self.speed / self.PIXEL_TO_METER
        dx = speed_pixels * math.cos(self.orientation) * 0.1
        dy = speed_pixels * math.sin(self.orientation) * 0.1
        self.position += np.array([dx, dy])
        self.trajectory.append(self.position.copy())

        # Vérification si le véhicule a franchi la ligne
        self.check_finish_line()

        # Vérification si le véhicule est hors des limites
        if self.is_out_of_bounds():
            print("🚗 Le véhicule est sorti du circuit ! Fin de la simulation.")
            return False

        return True

    def update_sensors(self):
        """Met à jour les données du LIDAR."""
        current_time = time.time()
        if current_time - self.last_lidar_update >= 0.025:
            self.sensor_data['lidar'] = self.simulate_lidar()
            self.sensor_data['speed'] = self.speed
            self.last_lidar_update = current_time

            self.obstacle_right_detected = False
            self.obstacle_left_detected = False

            self.display_lidar_data()

            # Calcul des distances front_right et front_left basées sur angle_diff
            front_right_distances = []
            front_left_distances = []

            # Pour chaque point LIDAR, on calcule l'angle par rapport à la direction du véhicule
            for p in self.sensor_data['lidar']:
                closest_x, closest_y, closest_distance = p

                # Calcul de l'angle entre l'avant du véhicule et le point LIDAR
                dx = closest_x - self.position[0]
                dy = closest_y - self.position[1]
                angle_to_point = math.atan2(dy, dx)

                # Calcul de la différence d'angle entre la direction du véhicule et le point LIDAR
                angle_diff = angle_to_point - self.orientation
                # Assurer que l'angle est dans l'intervalle [-pi, pi]
                angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

                # Si le point est dans la plage de l'angle droit
                if -15*np.pi/180 <= angle_diff <= -5*np.pi/180:
                    front_right_distances.append(closest_distance)

                # Si le point est dans la plage de l'angle gauche
                elif 5*np.pi/180 <= angle_diff <= 15*np.pi/180:
                    front_left_distances.append(closest_distance)

            # Vérification des obstacles à droite et à gauche
            if front_right_distances:
                self.obstacle_right_detected = min(front_right_distances) < self.stop_distance

            if front_left_distances:
                self.obstacle_left_detected = min(front_left_distances) < self.stop_distance


    def simulate_lidar(self):
        """Simule un LIDAR à 360°."""
        max_range_pixels = int(35 / self.PIXEL_TO_METER)
        x, y = self.position.astype(int)

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
                    lidar_points.append((check_x, check_y, d * self.PIXEL_TO_METER))
                    obstacle_found = True
                    break

            if not obstacle_found:
                lidar_points.append((x + math.cos(angle) * max_range_pixels,
                                    y + math.sin(angle) * max_range_pixels,
                                    max_range_pixels * self.PIXEL_TO_METER))

        return lidar_points

    def display_lidar_data(self):
        """Affiche les coordonnées et l'angle du point LIDAR le plus proche."""
        if not self.sensor_data['lidar']:
            return

        closest_point = min(self.sensor_data['lidar'], key=lambda p: p[2])

        closest_x, closest_y, closest_distance = closest_point
        angle = math.degrees(math.atan2(closest_y - self.position[1], closest_x - self.position[0]))

        # print(f"\n--- Point LIDAR le plus proche ---")
        # print(f"Coordonnées: ({closest_x:.2f}, {closest_y:.2f}), Distance: {closest_distance:.2f} m")
        # print(f"Angle: {angle:.2f}°")
        # print("----------------------------------")

    def check_finish_line(self):
        """Vérifie si le véhicule a franchi la ligne de départ."""
        # Si la position du véhicule correspond à la ligne rouge
        if 598 <= self.position[0] <= 602 and 490 <= self.position[1] <= 600:
            # Calculer et afficher le temps du tour lorsque le véhicule franchit à nouveau la ligne
            lap_time = time.time() - self.start_time
            self.lap_time = lap_time
            print(f"🎉 Le véhicule a terminé son tour en {lap_time:.2f} secondes.")

    def is_out_of_bounds(self):
        """Vérifie si le véhicule est hors des limites de la carte."""
        x, y = int(self.position[0]), int(self.position[1])
        return x < 0 or x >= self.map.shape[1] or y < 0 or y >= self.map.shape[0] or self.map[y, x] == 0

class CircuitSimulator:
    def __init__(self, map_image, initial_position, initial_speed):
        self.map = map_image
        self.vehicle = Vehicle(initial_position, initial_speed, map_image)

    def render(self, ax):
        """Affiche la carte et la position du véhicule sur le circuit."""
        ax.clear()
        ax.imshow(self.map, cmap='gray')

        ax.scatter(self.vehicle.position[0], self.vehicle.position[1], color='red', label='Véhicule', s=100)

        trajectory = np.array(self.vehicle.trajectory)
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], color='blue', linewidth=2, alpha=0.6)

        # Affichage du point LIDAR le plus proche
        if self.vehicle.sensor_data['lidar']:
            # Trouver le point LIDAR le plus proche
            closest_point = min(self.vehicle.sensor_data['lidar'], key=lambda p: p[2])
            closest_x, closest_y, closest_distance = closest_point

            # Calculer l'angle entre l'avant du véhicule et le point LIDAR
            dx = closest_x - self.vehicle.position[0]
            dy = closest_y - self.vehicle.position[1]
            angle_to_closest_point = math.atan2(dy, dx)

            # Calculer l'angle entre la direction du véhicule et le point LIDAR
            angle_diff = angle_to_closest_point - self.vehicle.orientation
            # Assurer que l'angle est dans l'intervalle [-pi, pi]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

            # Affichage des informations sur la console
            print(f"Point LIDAR le plus proche : ({closest_x:.2f}, {closest_y:.2f}) à {closest_distance:.2f} m")
            print(f"Angle entre l'avant du véhicule et l'obstacle : {math.degrees(angle_diff):.2f}°")

            # Affichage du point sur la carte
            ax.scatter(closest_x, closest_y, color='green', label='Point LIDAR le plus proche', s=50)

        # Affichage du chronomètre
        elapsed_time = time.time() - self.vehicle.start_time if self.vehicle.chronometer_running else 0  # Temps écoulé depuis le début de la simulation
        lap_time = self.vehicle.lap_time if self.vehicle.lap_time else "N/A"  # Temps du tour ou N/A si pas de tour terminé
        ax.set_title(f"Temps: {elapsed_time:.2f}s, Vitesse: {self.vehicle.speed:.2f} m/s, Temps du tour: {lap_time}")

        # Ligne de départ
        x1, y1 = 600, 490  # Point de départ
        x2, y2 = 600, 600  # Point d'arrivée
        ax.plot([x1, x2], [y1, y2], color='red', linewidth=2)

        ax.legend()
        ax.set_xlim(0, self.map.shape[1])
        ax.set_ylim(self.map.shape[0], 0)

    def update(self):
        """Met à jour l'état de la simulation et du véhicule."""
        vecteur = [self.vehicle.speed, 0.0, self.vehicle.speed, 1.75, self.vehicle.speed, -1.75]  # Ajuste ce vecteur selon les besoins
        if not self.vehicle.move_vehicle(*vecteur):  # Passe le vecteur ici
            return False
        return True

def find_valid_start_position(map_image):
    # Position fixe du véhicule à (600, 550)
    return np.array([600, 550])

# Chargement de la carte et position de départ
map_file = r"C:\Users\nasre\Documents\Simulateur F1tenth\mapsimtoulouse1.jpg"
map_image = load_map(map_file)
initial_position = find_valid_start_position(map_image)

# Initialisation du simulateur
simulator = CircuitSimulator(map_image, initial_position=initial_position, initial_speed=1.0)

# Création de la figure
fig, ax = plt.subplots(figsize=(6, 6))
plt.subplots_adjust(bottom=0.2)

# Slider pour modifier la vitesse de l'animation
slider_speed = Slider(plt.axes([0.1, 0.05, 0.8, 0.03]), 'Vitesse Animation', 0.1, 10.0, valinit=1.0, valstep=0.1)

def update(frame):
    """Met à jour la position et ajuste l'intervalle de l'animation sans changer la vitesse du véhicule."""
    if not simulator.update():
        ani.event_source.stop()
    simulator.render(ax)

def update_animation_speed(val):
    """Modifie l'intervalle de rafraîchissement de l'animation et la vitesse du véhicule."""
    speed_factor = slider_speed.val
    simulator.vehicle.speed = speed_factor  # Mettre à jour la vitesse du véhicule
    ani.event_source.interval = 1000 / speed_factor  # Ajuste la vitesse d'animation


# Création de l'animation
ani = animation.FuncAnimation(fig, update, interval=100, repeat=True)

# Lier le slider à la mise à jour de la vitesse d'animation
slider_speed.on_changed(update_animation_speed)

plt.show()
