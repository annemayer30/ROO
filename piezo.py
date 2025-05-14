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
    traffic_data = np.array(traffic_data).flatten()
    traffic_data = np.roll(traffic_data, -420)
    light_data = np.array(light_data).flatten()
    light_data = np.roll(light_data, -420)

    Ppv = traffic_data * piezo_unit_output * piezo_count * 4
    raw_load = light_data * lamp_power
    total_piezo = np.sum(Ppv)
    total_raw_load = np.sum(raw_load)
    if not np.isfinite(total_raw_load) or total_raw_load == 0:
        multiplier = 1
    else:
        multiplier = max(int(total_piezo // total_raw_load), 1)
    Pload = raw_load * multiplier

    N = len(Pload)
    time_hr = np.arange(N) / 60

    temp_flow = Pload - Ppv
    Ebatt_temp = np.cumsum(temp_flow) - np.cumsum(temp_flow)[0]
    energy_range = np.max(Ebatt_temp) - np.min(Ebatt_temp)
    battery_capacity = energy_range / (E_ratio_max - E_ratio_min)
    Emin = battery_capacity * E_ratio_min
    Emax = battery_capacity * E_ratio_max
    pcs_required = math.ceil(np.max(np.abs(temp_flow)) / 60)

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
                    available_discharge = Eb_prev - Emin
                    pb = min(required, max(0, available_discharge))
            else:
                surplus = pv
                available_charge = Emax - Eb_prev
                pb = -min(surplus, max(0, available_charge))

            Pbatt[t] = pb
            Ebatt[t] = Eb_prev - pb

        offset = (Ebatt[-1] - Ebatt[0]) / N
        Pbatt_adj = Pbatt - offset
        Ebatt_adj = [E_init]
        for t in range(1, N):
            Ebatt_adj.append(Ebatt_adj[-1] - Pbatt_adj[t])
        Ebatt_adj = np.array(Ebatt_adj)

        if np.all(Ebatt_adj >= Emin) and np.all(Ebatt_adj <= Emax):
            total_supplied = np.sum(np.minimum(Ppv + np.where(Pbatt_adj > 0, Pbatt_adj, 0), Pload))
            if total_supplied > max_supplied:
                max_supplied = total_supplied
                Pbatt_best = Pbatt_adj.copy()
                Ebatt_best = Ebatt_adj.copy()

    return time_hr, Ppv, Pload, Pbatt_best, Ebatt_best, Emax, Emin, battery_capacity, multiplier, pcs_required

def plot_energy_flow(time_hr, Ppv, Pload, Pbatt, Ebatt, Emax, Emin, battery_capacity, multiplier, pcs_required):
    # numpy 변환
    time_hr = np.array(time_hr)
    Ppv = np.array(Ppv)
    Pload = np.array(Pload)
    Pbatt = np.array(Pbatt)
    Ebatt = np.array(Ebatt)

    # 길이 맞추기
    min_len = min(len(time_hr), len(Ppv), len(Pload), len(Pbatt), len(Ebatt))
    time_hr = time_hr[:min_len]
    Ppv = Ppv[:min_len]
    Pload = Pload[:min_len]
    Pbatt = Pbatt[:min_len]
    Ebatt = Ebatt[:min_len]

    # 유효값 마스킹
    mask = np.isfinite(time_hr) & np.isfinite(Ppv) & np.isfinite(Pload) & np.isfinite(Pbatt) & np.isfinite(Ebatt)
    time_hr = time_hr[mask]
    Ppv = Ppv[mask]
    Pload = Pload[mask]
    Pbatt = Pbatt[mask]
    Ebatt = Ebatt[mask]

    Pload_actual = Ppv + np.where(Pbatt > 0, Pbatt, 0)

    fig, ax1 = plt.subplots(figsize=(12, 5))
    fig.subplots_adjust(left=0.22, right=0.95)

    ax1.plot(time_hr, Ppv, label='Piezo [Wh/min]', color='orange')
    ax1.plot(time_hr, Pload, label='Load [Wh/min]', color='blue')
    ax1.fill_between(time_hr, Ppv, Pload_actual, where=Pbatt > 0, interpolate=True, color='red', alpha=0.3, label='Battery Discharge')
    ax1.fill_between(time_hr, 0, Pbatt, where=Pbatt < 0, interpolate=True, color='green', alpha=0.3, label='Battery Charge')
    ax1.set_xlabel("Time [hr]")
    ax1.set_ylabel("Energy Flow [Wh/min]")
    ax1.set_xlim([0, 24])
    ax1.grid(True)

    ax2 = ax1.twinx()
    ax2.plot(time_hr, Ebatt, label='Battery Energy [Wh]', color='purple')
    ax2.axhline(Emax, color='purple', linestyle=':')
    ax2.axhline(Emin, color='purple', linestyle=':')
    ax2.set_ylim([0, battery_capacity])
    ax2.set_ylabel("Battery Energy [Wh]")

    fig.legend(loc='upper left', bbox_to_anchor=(0.02, 0.92))
    info_text = f"Streetlamps: {multiplier}\nBattery: {battery_capacity:.0f} Wh\nPCS: {pcs_required} W"
    fig.text(0.02, 0.82, info_text, fontsize=10, bbox=dict(facecolor='white', edgecolor='gray'))
    st.pyplot(fig)

def main():
    st.title("서울시 Piezo 기반 교통 발전 지도")

    piezo_unit_output = st.number_input("Piezo Output per Tile (Wh)", value=0.00000289, format="%.8f")
    piezo_count = st.number_input("Number of Piezo Tiles", value=100000, step=1000)
    lamp_power = st.number_input("Lamp Power (W)", value=100, step=10)
    E_ratio_max = st.slider("ESS Max Ratio", min_value=0.5, max_value=1.0, value=0.8, step=0.01)
    E_ratio_min = st.slider("ESS Min Ratio", min_value=0.0, max_value=0.5, value=0.2, step=0.01)

    location_df, traffic_df, light_df = load_data()
    address_list = traffic_df.iloc[0].tolist()
    traffic_values = traffic_df.iloc[1:].T.values
    light_values = light_df.values

    m = folium.Map(location=[37.55, 126.98], zoom_start=11)

    for idx, row in location_df.iterrows():
        addr = row['지점 위치']
        if addr in address_list:
            traffic_idx = address_list.index(addr)
            traffic_series = traffic_values[traffic_idx].flatten()
            light_series = light_values[traffic_idx].flatten()
            traffic_series = traffic_series[:1440]
            light_series = light_series[:1440]

            iframe = folium.IFrame(f"<b>{addr}</b><br>Click to see graph in app")
            popup = folium.Popup(iframe, min_width=200, max_width=300)
            folium.Marker(
                location=[row['위도'], row['경도']],
                popup=popup,
                tooltip=addr,
                icon=folium.Icon(color='blue')
            ).add_to(m)

    st_data = st_folium(m, width=1000, height=600)

    if st_data and st_data['last_object_clicked_tooltip']:
        clicked_addr = st_data['last_object_clicked_tooltip']
        traffic_idx = address_list.index(clicked_addr)
        traffic_series = traffic_values[traffic_idx].flatten()
        light_series = light_values[traffic_idx].flatten()
        traffic_series = traffic_series[:1440]
        light_series = light_series[:1440]

        time_hr, Ppv, Pload, Pbatt, Ebatt, Emax, Emin, battery_capacity, multiplier, pcs_required = simulate_piezo(
            traffic_series, light_series, piezo_unit_output, piezo_count, lamp_power, E_ratio_max, E_ratio_min
        )
        st.subheader(f"에너지 흐름 시각화: {clicked_addr}")
        plot_energy_flow(time_hr, Ppv, Pload, Pbatt, Ebatt, Emax, Emin, battery_capacity, multiplier, pcs_required)

if __name__ == "__main__":
    main()

