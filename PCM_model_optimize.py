import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

# === Load Data ===
input_data = pd.read_excel("data/input_summer.xlsx")
h_T_data = pd.read_excel("data/h(T)_2.xlsx")

# === Clean and Prepare Enthalpy Table ===
h_T_cleaned = h_T_data[['T', 'H']].dropna()

# === Constants and Geometry ===
cp_air = 1007  # J/kg·K
plate_area = 0.45 * 0.30  # m²
cross_section_area = 0.018  # m²
pcm_mass_total = 2  # kg per plate
x_steps = 3
time_step_s = 300  # 5 min in seconds
num_steps = len(input_data)

# === Function to Compute Convective Heat Transfer Coefficient ===
def compute_convective_h(velocity, L=0.3):
    rho = 1.2  # kg/m³
    mu = 1.8e-5  # Pa·s
    Pr = 0.71
    k = 0.026  # W/m·K

    Re = rho * velocity * L / mu

    if Re < 5e5:
        Nu = 0.664 * Re ** 0.5 * Pr ** (1 / 3)  # Laminar
    else:
        Nu = 0.037 * Re ** (4 / 5) * Pr ** (1 / 3)  # Turbulent

    h_conv = Nu * k / L
    return h_conv

# === Objective Function for Optimization (Scaling Enthalpy) ===
def objective_function(k_factor):
    # Modify enthalpy interpolation with scaling factor
    H_scaled = h_T_cleaned['H'] * k_factor
    h_interp_scaled = interp1d(h_T_cleaned['T'], H_scaled, kind='linear', fill_value="extrapolate")

    # === Initialize Temperatures ===
    air_temp = np.zeros((num_steps, x_steps))
    pcm_temp = np.full((x_steps, 2), 20.5)
    pcm_mass_per_cell = pcm_mass_total / x_steps
    air_temp[0, :] = input_data.iloc[0]['HEX1_Ta_inlet (degC (Ave))']

    # === Time Loop ===
    for t in range(1, num_steps):
        air_temp[t, 0] = input_data.iloc[t]['HEX1_Ta_inlet (degC (Ave))']
        velocity = input_data.iloc[t]['HEX1_v_outlet (m/s)']
        mass_flow_air = velocity * cross_section_area * 1.2  # rho_air ≈ 1.2 kg/m³
        h_conv = compute_convective_h(velocity, L=0.3)

        for x in range(1, x_steps):
            Ta = (air_temp[t-1, x-1] + air_temp[t-1, x]) / 2
            Q_pcm = h_conv * plate_area * (Ta - pcm_temp[x, 0])
            Q_pcm += h_conv * plate_area * (Ta - pcm_temp[x, 1])
            delta_T_air = -2 * Q_pcm / (mass_flow_air * cp_air)
            air_temp[t, x] = air_temp[t-1, x] + delta_T_air

            for y in range(2):
                T_prev = pcm_temp[x, y]
                h_prev = h_interp_scaled(T_prev)
                h_new = h_prev + (Q_pcm / 2) * time_step_s / pcm_mass_per_cell
                T_vals = h_T_cleaned['T'].values
                H_vals = H_scaled.values
                T_new = np.interp(h_new, H_vals, T_vals)
                pcm_temp[x, y] = T_new

    # Compute Mean Squared Error (MSE) for air temperature at outlet (x=2)
    simulated_outlet_temp = air_temp[:, -1]
    experimental_outlet_temp = input_data["HEX1_Ta_outlet (degC (Ave))"].values
    mse = np.mean((simulated_outlet_temp - experimental_outlet_temp) ** 2)

    return mse, air_temp  # Return the error and air_temp array

# === Run Optimization ===
result = minimize_scalar(lambda k: objective_function(k)[0], bounds=(0.1, 5), method='bounded')
optimal_k = result.x

print(f"✅ Optimal scaling factor for H(T): {optimal_k:.4f}")

# === Run Final Simulation with Optimized Enthalpy Scaling ===
_, optimized_air_temp = objective_function(optimal_k)  # Capture the final air_temp

# === Save Plots ===
os.makedirs("figures", exist_ok=True)

# Plot 1: Simulated vs Experimental Air Temp at Outlet
plt.figure(figsize=(10, 5))
plt.plot(input_data["Timestamp"], input_data["HEX1_Ta_inlet (degC (Ave))"], label="Inlet")
plt.plot(input_data["Timestamp"], input_data["HEX1_Ta_outlet (degC (Ave))"], label="Experimental Outlet", linestyle="--")
plt.plot(input_data["Timestamp"], optimized_air_temp[:, -1], label=f"Optimized Simulated Outlet (k={optimal_k:.4f})")
plt.title(f"Optimized Enthalpy Scaling vs Experimental Air Temp")
plt.ylabel("Temperature (°C)")
plt.xlabel("Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/optimized_enthalpy_air_temp_comparison.png")
plt.close()

# === Save the optimized k-value for future reference ===
with open("figures/optimized_enthalpy_k_value.txt", "w") as f:
    f.write(f"Optimal k: {optimal_k:.4f}\n")

print("✅ Optimization complete. Results saved.")
