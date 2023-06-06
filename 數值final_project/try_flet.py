#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import folium
import numpy as np
import pandas as pd
from folium.plugins import MarkerCluster
import flet as ft
import webbrowser
from flet import AppBar, ElevatedButton, Page, Text, View, colors
import matplotlib
import matplotlib.pyplot as plt
from flet.matplotlib_chart import MatplotlibChart
from sklearn.linear_model import LinearRegression
import tensorflow as tf

matplotlib.use("svg")

m_0 = folium.Map([22.988087, 120.2], zoom_start=15, tiles='OpenStreetMap')
df_traffic = pd.read_csv(r'.\tinan交通事故原因傷亡.csv')
df_wifi = pd.read_csv(r'.\wifi.csv')
df_parking1 = pd.read_csv(r'.\停車場1.csv')
df_parking2 = pd.read_csv(r'.\停車場2.csv')
df_bus = pd.read_csv(r'.\公車站牌.csv')


acci = []
wifi = []
park = []
star = []
bus = []
#顯示地圖=========================================================================================
def show_map():
    global m_0  # 使用全局變量
    # Create a new map object
    marker_cluster = MarkerCluster().add_to(m_0)
    marker_cluster1 = MarkerCluster().add_to(m_0)
    marker_cluster2 = MarkerCluster().add_to(m_0)
    marker_cluster3 = MarkerCluster().add_to(m_0)

    for index, row in df_traffic.iterrows():
        if (row['latitude'] < 23.027840 and row['latitude'] > 22.98 and row['longitude'] > 120.06 and row['longitude'] < 120.235):
            information = "Traffic_Accident"+ '<br>' + str(row['發生縣市名稱'] + '<br>')
            folium.Marker(location=([row['latitude'], row['longitude']]), popup=information,icon=folium.Icon(color="red",icon="car",prefix='fa')).add_to(marker_cluster)
    for index, row in df_wifi.iterrows():
        if (row['latitude'] < 23.027840 and row['latitude'] > 22.97 and row['longitude'] > 120.06 and row['longitude'] < 120.235):
            information = 'Free_Wifi_Station' + str(row['熱點名稱'])
            folium.Marker(location=([row['latitude'], row['longitude']]), popup=information,
                         icon=folium.Icon(color="blue",icon="wifi",prefix='fa')).add_to(marker_cluster1)
    for index, row in df_parking1.iterrows():
        if (row['latitude'] < 23.027840 and row['latitude'] > 22.97 and row['longitude'] > 120.06 and row['longitude'] < 120.235):
            information = 'Parking_Lot' + str(row['停車場名稱'])
            folium.Marker(location=([row['latitude'], row['longitude']]), popup=information,
                         icon=folium.Icon(color="black",icon="car",prefix='fa')).add_to(marker_cluster2)
    for index, row in df_parking2.iterrows():
        if (row['latitude'] < 23.027840 and row['latitude'] > 22.97 and row['longitude'] > 120.06 and row['longitude'] < 120.235):
            information = 'Parking_Lot' + str(row['停車場地址'])
            folium.Marker(location=([row['latitude'], row['longitude']]), popup=information,
                         icon=folium.Icon(color="black",icon="car",prefix='fa')).add_to(marker_cluster2)
    for index, row in df_bus.iterrows():
        if (row['latitude'] < 23.027840 and row['latitude'] > 22.97 and row['longitude'] > 120.06 and row['longitude'] < 120.235):
            information = '<br>' + "bus"
            folium.Marker(location=([row['latitude'], row['longitude']]), popup=information,
                         icon=folium.Icon(color="cadetblue",icon="bus",prefix='fa')).add_to(marker_cluster3)
    # Save the map as HTML
    m_0.save('map.html')
    
     
    # Open the map HTML file in the default web browser
    webbrowser.open('map.html')
#===============================================================================================

#計算數量=========================================================================================
def count_nearby(latitude, longitude, threshold,df):
    point_vector = np.array([latitude, longitude])
    # 將座標轉換成向量
    vectors = df[['latitude', 'longitude']].values
    # 計算向量和事件位置向量之間的距離
    distances = np.linalg.norm(vectors - point_vector, axis=1)
    # 篩選出位於距離閾值內的事件
    nearby = df[distances <= threshold]
    # 計算位於距離閾值內的事件量
    num_nearby = len(nearby)
    return num_nearby

#flet=========================================================================================
def main(page: ft.Page):
    page.title = "台南中西區、安平區、東區預測地點評級APP"
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER
    page.scroll = ft.ScrollMode.AUTO
    
    fig, ax = plt.subplots(figsize=(8,6))
    chart1 = MatplotlibChart(fig,original_size=True,transparent=True)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    
    #文字===================================================================
    start_text = ft.Text("")
    show_text = ft.Text("")
    show_text3 = ft.Text("")
    compute_text = ft.Text("")
    compute_text2 = ft.Text("")
    compute_text3 = ft.Text("")
    compute_text4 = ft.Text("")
    compute_text_ans = ft.Text("")
    compute_text2_ans = ft.Text("")
    compute_text3_ans = ft.Text("")
    compute_text4_ans = ft.Text("")
    fast_text = ft.Text("")
    line_compute_text = ft.Text("")
    compute_text_final = ft.Text("",size=20, color="blue",weight=ft.FontWeight.BOLD)
    compute2_text_final = ft.Text("",size=20, color="orange",weight=ft.FontWeight.BOLD)

    part1_t  = ft.Text("Part1、加入已知資訊", size=30, color="pink600",weight=ft.FontWeight.BOLD)
    part2_t  = ft.Text("Part2、繪製線性回歸表", size=30, color="pink600",weight=ft.FontWeight.BOLD)
    part3_t  = ft.Text("Part3、利用線性回歸估算評級", size=30, color="pink600",weight=ft.FontWeight.BOLD)
    #=======================================================================
    #輸入框==================================================================
    longitude = ft.TextField(text_align=ft.TextAlign.RIGHT, label="緯度", width=125,hint_text="ex:23.0")###中間的數字，外面有方格(field)
    latitude = ft.TextField(text_align=ft.TextAlign.RIGHT, label="經度", width=125,hint_text="ex:120.2")###中間的數字，外面有方格(field)
    info = ft.TextField(text_align=ft.TextAlign.RIGHT, label="已知評級的店家名稱",width=300,hint_text="ex:露易莎咖啡")###中間的數字，外面有方格(field)
    longitude_ans = ft.TextField(text_align=ft.TextAlign.RIGHT, label="緯度", width=125,hint_text="ex:23.0")###中間的數字，外面有方格(field)
    latitude_ans = ft.TextField(text_align=ft.TextAlign.RIGHT, label="經度", width=125,hint_text="ex:120.2")###中間的數字，外面有方格(field)
    info_ans = ft.TextField(text_align=ft.TextAlign.RIGHT, label="想要推估的店家名稱",width=300,hint_text="ex:麥當勞")###中間的數字，外面有方格(field)
    rc = ft.TextField(text_align=ft.TextAlign.RIGHT, label="想要運算的回合數",width=200,hint_text="ex:10")###中間的數字，外面有方格(field)
    #=======================================================================
    
    #按鈕===================================================================
    def save_click(e):
        folium.Marker(location=[longitude.value, latitude.value], popup=info.value, icon=folium.Icon(color=select_color.value,icon=select_icon.value,prefix='fa')).add_to(m_0)
        start_text.value = f"{info.value}儲存成功"
        m_0.save('map.html')
        num_accidents = count_nearby(float(longitude.value), float(latitude.value),0.1/110.95,df_traffic)
        compute_text.value = f"附近的交通事故數量：{num_accidents}個"
        num_wifi = count_nearby(float(longitude.value), float(latitude.value),0.1/110.95,df_wifi)
        compute_text2.value = f"附近的免費網路基地站：{num_wifi}個"
        num_parking1 = count_nearby(float(longitude.value), float(latitude.value),0.2/110.95,df_parking1)
        num_parking2 = count_nearby(float(longitude.value), float(latitude.value),0.2/110.95,df_parking2)
        num_parking = num_parking1 + num_parking2
        compute_text3.value = f"附近的停車場：{num_parking}個"
        num_bus = count_nearby(float(longitude.value), float(latitude.value),0.1/110.95,df_bus)
        compute_text4.value = f"附近的公車站牌：{num_wifi}個"
        global acci
        global wifi
        global park
        global star
        global bus
        acci.append(num_accidents)
        wifi.append(num_wifi)
        park.append(num_parking)
        star.append(int(select_star.value))
        bus.append(num_bus)
        page.update()
    def Q_save_click(e):
        
        global acci
        global wifi
        global park
        global star
        global bus
        know_accidents = [11,10,15,0,5,20,3,3]
        know_parking = [3,3,1,0,2,1,2,3]
        know_wifi= [4,10,2,0,4,0,2,3]
        know_star = [3,5,1,6,7,1,5,6]
        know_bus = [3,5,1,6,7,1,5,6]#=======================================================================================
        folium.Marker(location=[22.990328, 120.204411], popup="孔廟老街",icon=folium.Icon(color="pink")).add_to(m_0)
        folium.Marker(location=[22.993643, 120.193859], popup="河樂廣場",icon=folium.Icon(color="pink")).add_to(m_0)
        folium.Marker(location=[22.991417, 120.169319], popup="7-11亞萬門市",icon=folium.Icon(color="pink")).add_to(m_0)
        folium.Marker(location=[22.982093, 120.157385], popup="漁光島",icon=folium.Icon(color="pink")).add_to(m_0)
        folium.Marker(location=[22.996702,  120.201477], popup="臺南祀典大天后宮",icon=folium.Icon(color="pink")).add_to(m_0)
        folium.Marker(location=[23.011035,  120.200372], popup="花園夜市",icon=folium.Icon(color="pink")).add_to(m_0)
        folium.Marker(location=[22.982659, 120.219246], popup="大東夜市",icon=folium.Icon(color="pink")).add_to(m_0)
        folium.Marker(location=[22.989834, 120.208218], popup="小古巴手做漢堡",icon=folium.Icon(color="pink")).add_to(m_0)
        m_0.save('map.html')
        
        
        acci=acci+know_accidents
        wifi=wifi+know_wifi
        park=park+know_parking
        star=star+know_star
        bus=bus+know_bus

        fast_text.value = f"匯入孔廟老街、小古巴手做漢堡、河樂廣場、7-11亞萬門市、漁光島、臺南祀典大天后宮、花園夜市、大東夜市"

        page.update()
    def show_click(e):
        show_text.value = f"等待大約15秒..."
        page.update()
        show_map()
    def line_click(e):
        
        
        # 資料集
        wifi_c = wifi # 免費網路覆蓋率
        star_c = star  # 已知地點的評級
        park_c = park  # 人口數
        acci_c = acci  # 交通事故率
        bus_c = bus 
        rate=[]
        for i in range(len(wifi_c)):
            rate.append(wifi_c[i] + bus_c[i] + 5*park_c[i] -acci_c[i])
        
        X = np.array(rate).reshape(-1, 1)

        # 將資料轉換為NumPy陣列
        y = np.array(star_c)
        
        # 構建線性回歸模型
        model = LinearRegression()
        model.fit(X, y)
        # 取得模型參數
        intercept = model.intercept_
        coefficients = model.coef_
    
        # 進行預測
        predictions = model.predict(X)
        # 繪製散點圖和回歸線
        scatter = ax.scatter(rate,star_c, c=acci_c, cmap='viridis')
        ax.plot(rate,predictions , color='red', linewidth=2)
        # 添加顏色條
        cbar = plt.colorbar(scatter)
        cbar.set_label('traffic accident rate')
        # 設定圖表標題和軸標籤
        ax.set_title('Linear Regression: rate vs star')
        ax.set_xlabel('rate')
        ax.set_ylabel('star')
        
        chart1.update()
    def compute_click(e):
        
        line_compute_text.value = f"等待大約10秒"
        page.update()
        
        folium.Marker(location=[longitude_ans.value, latitude_ans.value], popup=info_ans.value).add_to(m_0)
        m_0.save('map.html')
        num_accidents = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.1/110.95,df_traffic)
        compute_text_ans.value = f"附近的交通事故數量：{num_accidents}個"
        num_wifi = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.1/110.95,df_wifi)
        compute_text2_ans.value = f"附近的免費網路基地站：{num_wifi}個"
        num_parking1 = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.2/110.95,df_parking1)
        num_parking2 = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.2/110.95,df_parking2)
        num_parking = num_parking1 + num_parking2
        compute_text3_ans.value = f"附近的停車場：{num_parking}個"
        num_bus = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.1/110.95,df_bus)
        compute_text4_ans.value = f"附近的公車站牌數量：{num_bus}個"
        # 資料集
        wifi_c = wifi # 免費網路覆蓋率
        star_c = star  # 已知地點的評級
        park_c = park  # 人口數
        acci_c = acci  # 交通事故率
        bus_c = bus  # 公車站牌數量
        # 將資料轉換為NumPy陣列
        rate=[]
        for i in range(len(wifi_c)):
            rate.append(wifi_c[i] + bus_c[i] + 5*park_c[i] -acci_c[i])
        
        X = np.array(rate).reshape(-1, 1)
        y = np.array(star_c)
        # 構建線性回歸模型
        model = LinearRegression()
        model.fit(X, y)
        prediction = model.predict([[-num_accidents +num_wifi+5*num_parking]])
        compute_text_final.value = f"推估評級：{prediction[0]}"
        page.update()
    def compute2_click(e):
        
        num_accidents = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.1/110.95,df_traffic)
        num_wifi = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.1/110.95,df_wifi)
        num_parking1 = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.2/110.95,df_parking1)
        num_parking2 = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.2/110.95,df_parking2)
        num_parking = num_parking1 + num_parking2
        num_bus = count_nearby(float(longitude_ans.value), float(latitude_ans.value),0.2/110.95,df_bus)
        # 資料集
        wifi_c = wifi # 免費網路覆蓋率
        star_c = star  # 已知地點的評級
        park_c = park  # 人口數
        acci_c = acci  # 交通事故率
        bus_c = bus # 公車站牌數量
        
        n=int(rc.value)
        # 將資料轉換為NumPy陣列
        show_text3.value = f"等待運算{n}回合,大約5秒...(使用jupyter notebook)，跑執行檔會跑不動"
        page.update()
        features=np.array([[0]])
        for i in range(len(wifi_c)):
            if(i==0):
                features[0][0] = wifi_c[i] + bus_c[i] + 5*park_c[i] -acci_c[i]
            else:
                features = np.concatenate([features,np.array([[wifi_c[i] + bus_c[i] + 5*park_c[i] -acci_c[i]]])])
        ratings = np.array(star_c)
        
        
        # 正規化評價
        normalized_ratings = (ratings - np.min(ratings)) / (np.max(ratings) - np.min(ratings))
        
        # 建立線性回歸模型
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(1, input_shape=(1,))
        ])
        
        # 編譯模型
        model.compile(optimizer='adam', loss='mse')
        
        # 自定義回調函數
        loss_history = []
        def update_loss_history(epoch, logs):
            loss_history.append(logs['loss'])
        
        # 訓練模型
        model.fit(features, normalized_ratings, epochs=n, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=update_loss_history)])
                
        # 使用模型進行預測
        test_data = np.array([[4]])
        normalized_test_data = (test_data - np.min(ratings)) / (np.max(ratings) - np.min(ratings))
        normalized_prediction = model.predict(normalized_test_data)
        
        # 將預測評價轉換回原始範圍
        prediction = normalized_prediction * (np.max(ratings) - np.min(ratings)) + np.min(ratings)
        
        final_loss = loss_history[-1]
        compute2_text_final.value = f"預測評價:{prediction[0][0]}，loss:{final_loss}"
        
        page.update()
        
    save_button = ft.ElevatedButton(text=f"儲存地點", on_click=save_click)
    Q_save_button = ft.ElevatedButton(text=f"懶人鍵,一次匯入8個地點|只能按一次|", on_click=Q_save_click)

    show_button = ft.ElevatedButton(text=f"在預設瀏覽器中顯示地圖", on_click=show_click)
    line_button = ft.ElevatedButton(text=f"顯示回歸線", on_click=line_click)
    
    compute_button = ft.ElevatedButton(text=f"利用sklearn.linear_model計算評級", on_click=compute_click)
    compute2_button = ft.ElevatedButton(text=f"使用TensorFlow進行訓練和預測", on_click=compute2_click)

    #=======================================================================
    #選單===================================================================
    select_icon=ft.Dropdown(
            label="圖標樣式",
            hint_text="選擇圖標樣式",
            options=[
                ft.dropdown.Option("map-marker"),
                ft.dropdown.Option("school"),
                ft.dropdown.Option("coffee"),
                ft.dropdown.Option("wine-bottle"),
                ft.dropdown.Option("landmark"),
            ],
            width=350,
            autofocus=True,
        )   
    select_color=ft.Dropdown(
            label="圖標顏色",
            hint_text="選擇圖標顏色",
            options=[
                ft.dropdown.Option("orange"),
                ft.dropdown.Option("pink"),
                ft.dropdown.Option("green"),
            ],
            width=350,
            autofocus=True,
        )
    select_star=ft.Dropdown(
            label="評級",
            hint_text="選擇評級",
            options=[
                ft.dropdown.Option("1"),
                ft.dropdown.Option("2"),
                ft.dropdown.Option("3"),
                ft.dropdown.Option("4"),
                ft.dropdown.Option("5"),
                ft.dropdown.Option("6"),
                ft.dropdown.Option("7"),
                ft.dropdown.Option("8"),
                ft.dropdown.Option("9"),
                ft.dropdown.Option("10"),
            ],
            width=150,
            autofocus=True,
        )

    #=======================================================================

    
    #擺放===================================================================
    page.add(part1_t)
    page.add(ft.Text("---------------------不知道要輸入啥時---------------------",size=15))
    page.add(ft.Text("22.974587,120.221628 台南文化中心 評級:4",size=15))
    page.add(ft.Text("22.977471,120.202918 水交社公園 評級:5",size=15))
    page.add(ft.Text("22.989758,120.201360 台南市美術館2館 評級:6",size=15))
    page.add(ft.Text("--------------------------------------------------------------",size=15))
    page.add(
        ft.Row(###橫列擺放
            [
                longitude,
                latitude,
                info,
                select_star
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    page.add(
        ft.Row(###橫列擺放
            [
                select_icon,
                select_color,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    page.add(save_button)
    page.add(start_text)
    page.add(
        ft.Row(###橫列擺放
            [
                compute_text,
                compute_text2,
                compute_text3,
                compute_text4
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    page.add(
        ft.Row(###橫列擺放
            [
                Q_save_button,
                show_button,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    page.add(fast_text)
    page.add(show_text)

    page.add(part2_t)
    page.add(line_button)
    page.add(chart1)
    page.add(part3_t)
    page.add(
        ft.Row(###橫列擺放
            [
                longitude_ans,
                latitude_ans,
                info_ans,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    page.add(compute_button)
    page.add(line_compute_text)
    page.add(
        ft.Row(###橫列擺放
            [
                compute_text_ans,
                compute_text2_ans,
                compute_text3_ans,
                compute_text4_ans
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    
    page.add(compute_text_final)
    page.add(
        ft.Row(###橫列擺放
            [
                rc,
                compute2_button,
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        )
    )
    page.add(ft.Text("輸入的資料過少可能造成預測不準確",size=15))
    page.add(show_text3)
    page.add(compute2_text_final)
    
    #=======================================================================
ft.app(target=main)



# In[ ]:




