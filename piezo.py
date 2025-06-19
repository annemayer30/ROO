import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import math

LOCATION_PATH = "https://raw.githubusercontent.com/annemayer30/ROO/main/location.xlsx"
TRAFFIC_PATH = "https://raw.githubusercontent.com/annemayer30/ROO/main/trafficData01.xlsx"
LIGHT_PATH   = "https://raw.githubusercontent.com/annemayer30/ROO/main/lightData.xlsx"

@st.cache_data
def load_data():
    return (
        pd.read_excel(LOCATION_PATH),
        pd.read_excel(TRAFFIC_PATH, header=None),
        pd.read_excel(LIGHT_PATH,   header=None),
    )

# --------------------------------------------------
# 시뮬레이션 로직
# --------------------------------------------------

def simulate_piezo(traffic_data, light_data, piezo_unit_output, piezo_count,
                   lamp_power, E_ratio_max, E_ratio_min):
    traffic_data = np.array(traffic_data, dtype=float).flatten()[:1440]
    light_data   = np.array(light_data,   dtype=float).flatten()[:1440]

    traffic_data = np.roll(traffic_data, -420)
    light_data   = np.roll(light_data,   -420)

    Ppv  = traffic_data * piezo_unit_output * piezo_count * 4          # [Wh/min]
    raw_load = light_data * lamp_power                                 # [W] 가로등 1개 기준
    total_piezo     = Ppv.sum()
    total_raw_load  = raw_load.sum()
    multiplier      = max(int(total_piezo // total_raw_load), 1) if total_raw_load>0 else 1
    Pload = raw_load * multiplier                                      # [W]

    N        = len(Pload)
    time_hr  = np.arange(N) / 60

    # 배터리 계산
    temp_flow        = Pload - Ppv
    Ebatt_temp       = np.cumsum(temp_flow) - np.cumsum(temp_flow)[0]
    energy_range     = Ebatt_temp.max() - Ebatt_temp.min()
    battery_capacity = energy_range / (E_ratio_max - E_ratio_min) if E_ratio_max> E_ratio_min else 1
    Emin, Emax       = battery_capacity * np.array([E_ratio_min, E_ratio_max])
    pcs_required     = math.ceil(np.abs(temp_flow).max()/60)

    Pbatt_best = np.zeros(N)
    Ebatt_best = np.linspace(Emin, Emin, N)

    for E_init in np.linspace(Emin, Emax, 200):
        Ebatt = np.zeros(N)
        Pbatt = np.zeros(N)
        Ebatt[0] = E_init
        for t in range(1, N):
            delta = Pload[t] - Ppv[t]
            if delta > 0:                         # 방전 필요
                pb = min(delta, max(0, Ebatt[t-1]-Emin))
            else:                                 # 충전 가능
                pb = -min(-delta, max(0, Emax-Ebatt[t-1]))
            Pbatt[t] = pb
            Ebatt[t] = Ebatt[t-1] - pb
        if (Emin <= Ebatt).all() and (Ebatt <= Emax).all():
            Pbatt_best, Ebatt_best = Pbatt, Ebatt
            break

    return (time_hr, Ppv, Pload, Pbatt_best, Ebatt_best,
            Emax, Emin, battery_capacity, multiplier, pcs_required)

# --------------------------------------------------
# 그래프 그리기
# --------------------------------------------------

def plot_energy_flow(time_hr, Ppv, Pload, Pbatt, Ebatt,
                     Emax, Emin, battery_capacity, multiplier):
    fig, ax1 = plt.subplots(figsize=(12,5))
    fig.subplots_adjust(left=0.22, right=0.95)

    Pload_actual = Ppv + np.where(Pbatt>0, Pbatt, 0)
    ax1.plot(time_hr, Ppv,   label='Piezo [Wh/min]', color='orange')
    ax1.plot(time_hr, Pload, label='Load  [Wh/min]', color='blue')
    ax1.fill_between(time_hr, Ppv, Pload_actual, where=Pbatt>0,
                     interpolate=True, color='red', alpha=0.3,
                     label='Battery Discharge')
    ax1.fill_between(time_hr, 0, Pbatt, where=Pbatt<0,
                     interpolate=True, color='green', alpha=0.3,
                     label='Battery Charge')
    ax1.set_xlabel('Time [hr]')
    ax1.set_ylabel('Energy Flow [Wh/min]')
    ax1.set_xlim(0,24)
    ax1.grid(True)
    ax1.set_yticklabels([])

    ax2 = ax1.twinx()
    ax2.plot(time_hr, Ebatt, color='purple', label='Battery Energy [Wh]')
    ax2.axhline(Emax, color='purple', linestyle=':')
    ax2.axhline(Emin, color='purple', linestyle=':')
    ax2.set_ylim(0, battery_capacity)
    ax2.set_ylabel('Battery Energy [Wh]')
    ax2.set_yticklabels([])

    fig.legend(loc='upper left', bbox_to_anchor=(-0.01,0.85))
    fig.text(0.02, 0.45, f"Streetlamps: {multiplier}", fontsize=10,
             bbox=dict(facecolor='white', edgecolor='gray'))
    st.pyplot(fig)

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

def main():
    st.title('서울시 압전 발전 ESS 구성 정보')

    piezo_unit_output = st.number_input('Piezo Output per Tile (Wh)', value=2.89e-6, format='%.8f')
    piezo_count       = st.number_input('Piezo Tiles per Wheel', value=100000, step=1000)
    lamp_power        = st.number_input('Lamp Power (W)', value=100, step=10)
    E_ratio_max       = st.slider('SoC Max Ratio', 0.5, 1.0, 0.8, 0.01)
    E_ratio_min       = st.slider('SoC Min Ratio', 0.0, 0.5, 0.2, 0.01)

    location_df, traffic_df, light_df = load_data()
    address_list   = traffic_df.iloc[0].tolist()
    traffic_values = traffic_df.iloc[1:].T.values
    light_values   = np.tile(light_df.values.flatten(), (len(address_list),1))

    m = folium.Map(location=[37.55,126.98], zoom_start=11)
    for _, row in location_df.iterrows():
        addr = row['지점 위치']
        if addr in address_list:
            folium.Marker([row['위도'],row['경도']], tooltip=addr,
                          icon=folium.Icon(color='blue')).add_to(m)

    st_data = st_folium(m, width=1000, height=600)

    if st_data and st_data['last_object_clicked_tooltip']:
        addr = st_data['last_object_clicked_tooltip']
        idx  = address_list.index(addr)
        traffic_series = traffic_values[idx]
        light_series   = light_values[idx]

        out = simulate_piezo(traffic_series, light_series,
                             piezo_unit_output, piezo_count,
                             lamp_power, E_ratio_max, E_ratio_min)

        (time_hr, Ppv, Pload, Pbatt, Ebatt,
         Emax, Emin, battery_cap, mult, pcs_req) = out

        st.subheader(f'ESS 최적화 그래프: {addr}')
        plot_energy_flow(time_hr, Ppv, Pload, Pbatt, Ebatt,
                         Emax, Emin, battery_cap, mult)

if __name__ == '__main__':
    main()
