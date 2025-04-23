import os
import csv
import matplotlib.pyplot as plt
from Simulator import Simulator
from Optimizer import Optimizer

# ---------- MAIN ----------
if __name__ == '__main__':
    # Chargement de la carte
    map_file = r"C:\\Users\\nasre\\Documents\\Simulateur F1tenth\\mapsimtoulouse1.jpg"
    map_image = None
    if map_file.endswith('.npy'):
        map_image = np.load(map_file)
    else:
        img = cv2.imread(map_file, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger '{map_file}'")
        _, map_image = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
    print("Carte chargÃ©e, valeurs uniques :", np.unique(map_image))

    # # PrÃ©paration du log CSV
    # csv_file = 'optimizer_log.csv'
    # new_log = not os.path.isfile(csv_file) or os.stat(csv_file).st_size == 0
    # log = open(csv_file, 'a', newline='')
    # writer = csv.writer(log)
    # if new_log:
    #     writer.writerow(['Vit_linear', 'Vit_angular_right', 'Vit_angular_left', 'stop_distance', 'lidar_update_period', 'lap_time'])
    #     log.flush()

    # Fonction objectif utilisant Simulator
    global_counter = 0
    def simulation_objective(params):
        global global_counter
        # Clip des paramÃ¨tres
        Vit_linear = np.clip(params[0], 0.1, 1.0)
        Vit_angular_right = np.clip(params[1], 1.5, 3.0)
        Vit_angular_left = np.clip(params[2], -3.0, -1.5)
        stop_distance = np.clip(params[3], 0.5, 1.5)
        lidar_update_period = np.clip(params[4], 0.025, 0.1)

        # Simulation
        start_pos = np.array([604, 550])
        sim = Simulator(map_image, start_pos,
                        Vit_linear,
                        Vit_angular_right,
                        Vit_angular_left,
                        stop_distance,
                        lidar_update_period,
                        dt=0.1)
        target_laps, max_steps = 1, 5000
        fig, ax = plt.subplots()
        plt.ion()
        step = 0
        while sim.lap_count < target_laps and step < max_steps:
            if not sim.step():
                break
            sim.render(ax, global_counter)
            plt.pause(0.001)
            step += 1
        plt.close(fig)

        lap_time = sim.get_lap_time() if sim.lap_count >= target_laps else 1e6
        if sim.lap_count >= target_laps:
            global_counter += 1
            print(f"Iter {global_counter}: lap_time={lap_time:.2f}s | params=({Vit_linear:.3f},{Vit_angular_right:.3f},{Vit_angular_left:.3f},{stop_distance:.3f},{lidar_update_period:.3f})")

        writer.writerow([Vit_linear, Vit_angular_right, Vit_angular_left, stop_distance, lidar_update_period, lap_time])
        log.flush()
        return lap_time

    # Optimisation
    x0 = [0.4, 1.75, -1.75, 1.2, 0.025]
    optimizer = Optimizer(x0)
    best = optimizer.optimize(simulation_objective, maxiter=20)

  


  

