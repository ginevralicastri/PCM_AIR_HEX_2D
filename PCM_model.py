import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# === Load Data ===
input_data = pd.read_excel("data/input_data.xlsx")
h_T_data = pd.read_excel("data/h(T)_2.xlsx")

# === Clean and Prepare Enthalpy Table ===
h_T_cleaned = h_T_data[['T', 'H']].dropna()
h_interp = interp1d(h_T_cleaned['T'], h_T_cleaned['H'], kind='linear', fill_value="extrapolate")

# === Constants and Geometry ===
cp_air = 1007  # J/kg·K
h_conv = 15    # W/m²·K
plate_area = 0.45 * 0.30  # m²
cross_section_area = 0.018  # m²
pcm_mass_total = 2  # kg per plate
x_steps = 3
time_step_s = 300  # 5 min in seconds
num_steps = len(input_data)

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
# === Initialization ===
air_temp = np.zeros((num_steps, x_steps))
pcm_temp = np.full((x_steps, 2), 20.5)
pcm_mass_per_cell = pcm_mass_total / x_steps
air_temp[0, 0] = input_data.iloc[0]['HEX1_Ta_inlet (degC (Ave))']
air_temp[0, 1:] = air_temp[0, 0]

# === Time Loop ===
for t in range(1, num_steps):
    air_temp[t, 0] = input_data.iloc[t]['HEX1_Ta_inlet (degC (Ave))']
    velocity = input_data.iloc[t]['HEX1_v_outlet (m/s)']
    mass_flow_air = velocity * cross_section_area * 1.2  # rho_air ≈ 1.2 kg/m³
    h_conv= compute_convective_h(velocity, L=0.3)

    for x in range(1, x_steps):
        Ta = (air_temp[t-1, x-1] + air_temp[t-1, x]) / 2
        Q_pcm = h_conv * plate_area * (Ta - pcm_temp[x, 0])
        Q_pcm += h_conv * plate_area * (Ta - pcm_temp[x, 1])
        delta_T_air = -2*Q_pcm / (mass_flow_air * cp_air)
        air_temp[t, x] = air_temp[t-1, x] + delta_T_air

        for y in range(2):
            T_prev = pcm_temp[x, y]
            h_prev = h_interp(T_prev)
            h_new = h_prev + (Q_pcm / 2) * time_step_s / pcm_mass_per_cell
            T_vals = h_T_cleaned['T'].values
            H_vals = h_T_cleaned['H'].values
            T_new = np.interp(h_new, H_vals, T_vals)
            pcm_temp[x, y] = T_new

# === Create DataFrame ===
air_temp_df = pd.DataFrame(air_temp, columns=["x=0", "x=1", "x=2"])
air_temp_df["Timestamp"] = input_data["Timestamp"]
air_temp_df.set_index("Timestamp", inplace=True)

simulated_outlet_temp = air_temp[:, -1]
experimental_outlet_temp = input_data["HEX1_Ta_outlet (degC (Ave))"].values
mae = mean_absolute_error(experimental_outlet_temp, simulated_outlet_temp)
mse = mean_squared_error(experimental_outlet_temp, simulated_outlet_temp)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((experimental_outlet_temp - simulated_outlet_temp) / experimental_outlet_temp)) * 100
r2 = r2_score(experimental_outlet_temp, simulated_outlet_temp)
mbe = np.mean(simulated_outlet_temp - experimental_outlet_temp)
# Create DataFrame
metrics_df = pd.DataFrame({
    "Metric": [
        "Mean Absolute Error (MAE)",
        "Mean Squared Error (MSE)",
        "Root Mean Squared Error (RMSE)",
        "Mean Absolute Percentage Error (MAPE)",
        "Mean Bias Error (MBE)",
        "R² Score"
    ],
    "Value": [
        mae,
        mse,
        rmse,
        mape,
        mbe,
        r2
    ]
})

# Round values for neatness
metrics_df["Value"] = metrics_df["Value"].round(4)
# === Save Plots ===
os.makedirs("figures", exist_ok=True)

# Plot 1: Air temp at x=0,1,2
plt.figure(figsize=(10, 5))
plt.plot(air_temp_df.index, air_temp_df["x=0"], label="x=0")
plt.plot(air_temp_df.index, air_temp_df["x=1"], label="x=1")
plt.plot(air_temp_df.index, air_temp_df["x=2"], label="x=2")
plt.title("Simulated Air Temperature at x=0,1,2")
plt.ylabel("Temperature (°C)")
plt.xlabel("Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/air_temp_simulated.png")
plt.close()

# Plot 2: Simulated vs Experimental at x=2
plt.figure(figsize=(10, 5))
plt.plot(air_temp_df.index, air_temp_df["x=2"], label="Simulated outlet")
plt.plot(air_temp_df.index, input_data['HEX1_Ta_inlet (degC (Ave))'], label="inlet")
plt.plot(input_data["Timestamp"], input_data["HEX1_Ta_outlet (degC (Ave))"],
         label="Experimental Outlet", linestyle="--")
plt.title("Simulated vs Experimental Air Temp at x=2")
plt.ylabel("Temperature (°C)")
plt.xlabel("Time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/air_temp_comparison.png")
plt.close()

# Clean and keep only the needed columns
h_T_cleaned = h_T_data[['T', 'H']].dropna()

# Interpolation function for h(T)
h_interp = interp1d(h_T_cleaned['T'], h_T_cleaned['H'], kind='linear', fill_value="extrapolate")

# Create smooth temperature range for the plot
T_dense = np.linspace(h_T_cleaned['T'].min(), h_T_cleaned['T'].max(), 500)
H_dense = h_interp(T_dense)

# === Plot ===
plt.figure(figsize=(8, 5))
plt.plot(T_dense, H_dense, label="Interpolated h(T)", color="blue")
plt.scatter(h_T_cleaned['T'], h_T_cleaned['H'], color="red", s=10, label="Original Data")
plt.title("Interpolated Enthalpy h(T) vs Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Enthalpy h(T) (J/kg)")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save to figures folder
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/h_vs_T_interpolated.png")
plt.close()
# === Optional: Save results as CSV ===
# air_temp_df.to_csv("results/air_temperature_simulation.csv")
