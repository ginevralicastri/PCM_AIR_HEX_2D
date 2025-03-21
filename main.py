import numpy as np
import matplotlib
matplotlib.use('Agg')  # Usa backend non interattivo
import matplotlib.pyplot as plt
from PCM_model import run_pcm_hex_simulation
import pandas as pd
import os

output_dir = "figures"
excel_file = 'data/20_12_23_12.xlsx'
excel_pcm = 'data/h(T).xlsx'
time_array, T_air_history, H_p1_history, H_p2_history, T_out_list = run_pcm_hex_simulation(
        excel_pcm,  # file con T-H PCM
        excel_file,  # file con velocity, T_in
        n_i=3,  # numero di celle in lunghezza
    dt_min=5,  # passo di tempo [min] tra un record e l'altro
    h_c=10.0,  # coeff. di scambio convettivo [W/m2K]
    A_cell=0.02,  # area di scambio per singola cella
    cp_air=1007.0,  # calore specifico aria [J/kgK]
    rho_air=1.16,  # densita' aria [kg/m3]
    M_piastra=2.0,  # massa PCM totale per piastra (kg)
    channel_area_out=0.017  # sezione in uscita [m2] x calcolo portata
    )

df_exp = pd.read_excel(excel_file)
T_out_exp = df_exp["HEX1_Ta_outlet (degC (Ave))"].values
T_in_exp = df_exp["HEX1_Ta_inlet (degC (Ave))"].values

n_time_sim = len(time_array)
if len(T_out_exp) != n_time_sim:
    print("Attenzione: dimensioni diverse!")

diff = np.array(T_out_list) - T_out_exp
mean_error = np.mean(diff)
print("Errore medio simul - exp (°C):", mean_error)


plt.figure()
plt.plot(time_array, T_out_list, '--', label='Sim T_out')
plt.plot(time_array, T_out_exp, '-', label='Exp T_out')
plt.plot(time_array, T_in_exp, '-', label='T_in')
plt.xlabel('Tempo [min]')
plt.ylabel('Temperatura aria [°C]')
plt.legend()
plt.grid(True)
save_path = os.path.join(output_dir, "grafico.png")
plt.savefig(save_path, dpi=150)

print(f"Grafico salvato in: {save_path}")
