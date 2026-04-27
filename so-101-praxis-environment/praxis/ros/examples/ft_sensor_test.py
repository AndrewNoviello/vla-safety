from netft_py import NetFT

ft = NetFT("192.168.1.20")

while True:
    data = ft.getMeasurement()
    print(data)
    # time.sleep(0.01)