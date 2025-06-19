import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import math

# 파일 경로
LOCATION_PATH = "https://raw.githubusercontent.com/annemayer30/ROO/main/location.xlsx"
TRAFFIC_PATH = "https://raw.githubusercontent.com/annemayer30/ROO/main/trafficData01.xlsx"
LIGHT_PATH = "https://raw.githubusercontent.com/annemayer30/ROO/main/lightData.xlsx"

@st.cache_data
def load_data():
    location_df = pd.read_excel(LOCATION_PATH)
    traffic_df = pd.read_excel(TRAFFIC_PATH, header=None)
    light_df = pd.read_excel(LIGHT_PATH, header=None)
    return location_df, traffic_df, light_df

def simulate_piezo(traffic_data, light_data, piezo_unit_output, piezo_count, lamp_power, E_ratio_max, E_ratio_min):
    traffic_data = np.array(traffic_data, dtype=np.float64).flatten()[:1440]
    light_data = np.array(light_data, dtype=np.float64).flatten()[:1440]

    traffic_data = np.roll(traffic_data, -420)
    light_data = np.roll(light_data, -420)

    Ppv = traffic_data * piezo_unit_output * piezo_count * 4 / 60
    raw_load = light_data * lamp_power
    total_piezo = np.sum(Ppv)
    total_raw_load = np.sum(raw_load)
    multiplier = max(int(total_piezo // total_raw_load), 1) if total_raw_load > 0 else 1
    Pload = raw_load * multiplier

    N = len(Pload)
    time_hr = np.arange(N) / 60

    temp_flow = Pload - Ppv
    Ebatt_temp = np.cumsum(temp_flow) - np.cumsum(temp_flow)[0]
    energy_range = np.max(Ebatt_temp) - np.min(Ebatt_temp)
    battery_capacity = energy_range / (E_ratio_max - E_ratio_min) if (E_ratio_max - E_ratio_min) != 0 else 1
    Emin = battery_capacity * E_ratio_min
    Emax = battery_capacity * E_ratio_max
    pcs_required = math.ceil(np.max(np.abs(temp_flow)))

    Pbatt_best, Ebatt_best, max_supplied = None, None, -np.inf
    for E_init in np.linspace(Emin, Emax, 100):
        Ebatt = np.zeros(N)
        Pbatt = np.zeros(N)
        Ebatt[0] = E_init

        for t in range(1, N):
            Eb_prev = Ebatt[t-1]
            load = Pload[t]
            pv = Ppv[t]
            pb = 0
            if load > 0:
                required = load - pv
                if required > 0:
                    pb = min(required, max(0, Eb_prev - Emin))
            else:
                pb = -min(pv, max(0, Emax - Eb_prev))

            Pbatt[t] = pb
            Ebatt[t] = Eb_prev - pb

        offset = (Ebatt[-1] - Ebatt[0]) / N
        Pbatt_adj = Pbatt - offset
        Ebatt_adj = np.zeros(N)
        Ebatt_adj[0] = E_init
        for t in range(1, N):
            Ebatt_adj[t] = Ebatt_adj[t-1] - Pbatt_adj[t]

        if np.all((Emin <= Ebatt_adj) & (Ebatt_adj <= Emax)):
            total_supplied = np.sum(np.minimum(Ppv + np.where(Pbatt_adj > 0, Pbatt_adj, 0), Pload))
            if total_supplied > max_supplied:
                max_supplied = total_supplied
                Pbatt_best = Pbatt_adj.copy()
                Ebatt_best = Ebatt_adj.copy()

    return time_hr, Ppv, Pload, Pbatt_best, Ebatt_best, Emax, Emin, battery_capacity, multiplier, pcs_required

def plot_energy_flow(time_hr, Ppv, Pload, Pbatt, Ebatt, Emax, Emin, battery_capacity, multiplier, pcs_required):
    time_hr = np.asarray(time_hr)
    Ppv = np.asarray(Ppv)
    Pload = np.asarray(Pload)
    Pbatt = np.asarray(Pbatt)
    Ebatt = np.asarray(Ebatt)

    min_len = min(len(time_hr), len(Ppv), len(Pload), len(Pbatt), len(Ebatt))
    time_hr, Ppv, Pload, Pbatt, Ebatt = time_hr[:min_len], Ppv[:min_len], Pload[:min_len], Pbatt[:min_len], Ebatt[:min_len]

    mask = np.isfinite(time_hr) & np.isfinite(Ppv) & np.isfinite(Pload) & np.isfinite(Pbatt) & np.isfinite(Ebatt)
    time_hr, Ppv, Pload, Pbatt, Ebatt = time_hr[mask], Ppv[mask], Pload[mask], Pbatt[mask], Ebatt[mask]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(left=0.22, right=0.95)

    Pload_actual = Ppv + np.where(Pbatt > 0, Pbatt, 0)
    ax1.plot(time_hr, Ppv, label='Piezo [W]', color='orange')
    ax1.plot(time_hr, Pload, label='Load [W]', color='blue')
    ax1.fill_between(time_hr, Ppv, Pload_actual, where=Pbatt > 0, interpolate=True, color='red', alpha=0.3, label='Battery Discharge')
    ax1.fill_between(time_hr, 0, Pbatt, where=Pbatt < 0, interpolate=True, color='green', alpha=0.3, label='Battery Charge')
    ax1.set_xlabel("Time [hr]")
    ax1.set_ylabel("Energy Flow [W]")
    ax1.set_xlim([0, 24])
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(time_hr, Ebatt, label='Battery Energy [Wh]', color='purple')
    ax2.axhline(Emax, color='purple', linestyle=':')
    ax2.axhline(Emin, color='purple', linestyle=':')
    ax2.set_ylim([0, battery_capacity])
    ax2.set_ylabel("Battery Energy [Wh]")

    fig.legend(loc='upper left', bbox_to_anchor=(-0.01, 0.85))
    fig.text(0.02, 0.45, f"Streetlamps: {multiplier}\nBattery: {battery_capacity:.0f} Wh\nPCS: {pcs_required} W", fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'))
    st.pyplot(fig)

def main():
    st.title("서울시 압전 발전 ESS 구성 정보")

    piezo_unit_output = st.number_input("Piezo Output per Tile (W)", value=0.620, format="%.8f")
    piezo_count = st.number_input("Number of Piezo Tiles activated by a Single Wheel", value=1000, step=1000)
    lamp_power = st.number_input("Lamp Power (W)", value=100, step=10)
    E_ratio_max = st.slider("SoC Max Ratio", 0.5, 1.0, 0.8, 0.01)
    E_ratio_min = st.slider("SoC Min Ratio", 0.0, 0.5, 0.2, 0.01)

    location_df, traffic_df, light_df = load_data()
    address_list = traffic_df.iloc[0].tolist()
    traffic_values = traffic_df.iloc[1:].T.values
    light_values = np.tile(light_df.values.flatten(), (len(address_list), 1))

    m = folium.Map(location=[37.55, 126.98], zoom_start=11)

    for idx, row in location_df.iterrows():
        addr = row['지점 위치']
        if addr in address_list:
            traffic_idx = address_list.index(addr)
            traffic_series = traffic_values[traffic_idx]
            light_series = light_values[traffic_idx]

            folium.Marker(
                location=[row['위도'], row['경도']],
                popup=folium.Popup(f"<b>{addr}</b><br>Click to see graph", min_width=200, max_width=300),
                tooltip=addr,
                icon=folium.Icon(color='blue')
            ).add_to(m)

    st_data = st_folium(m, width=1000, height=600)

    if st_data and st_data['last_object_clicked_tooltip']:
        clicked_addr = st_data['last_object_clicked_tooltip']
        traffic_idx = address_list.index(clicked_addr)
        traffic_series = traffic_values[traffic_idx]
        light_series = light_values[traffic_idx]

        time_hr, Ppv, Pload, Pbatt, Ebatt, Emax, Emin, battery_capacity, multiplier, pcs_required = simulate_piezo(
            traffic_series, light_series, piezo_unit_output, piezo_count, lamp_power, E_ratio_max, E_ratio_min
        )
        st.subheader(f"ESS 최적화 그래프: {clicked_addr}")
        plot_energy_flow(time_hr, Ppv, Pload, Pbatt, Ebatt, Emax, Emin, battery_capacity, multiplier, pcs_required)

if __name__ == "__main__":
    main()
