from datetime import datetime

import pandas as pd
import requests
from hikvisionapi import Client

cam = Client('http://192.168.227.17', 'admin', 'adminONPPIG')
# cam = Client('http://172.20.56.101', 'admin', 'adminONPPIG')

# url = 'http://localhost/deeps/public/people'
# token = '2|Zfk7iQS9QeSFGwJkTevsJfz1SuprcZwLZBgUdwrX'

# url = 'http://deeps/people'
# token ='1|TkrpwarqxtY3tW7ayoXmeISiCspCYCzZlhAwyYrp'

url = 'http://nec.cse.buet.ac.bd/people'
token = '1|DNOTyGn8ettsFyezwHEzEkqVrHvy2tz0NDgmNJ6n'


base_path = 'hajj_images/'
idx_file = base_path + 'hajj_images.csv'


def post_data(payload):
    headers = {
        'Authorization': token,
    }

    files = {
        'image': payload['image']
    }

    data = {
        'captured_at': payload['captured_at']
    }

    try:
        print(data)
        res = requests.post(url, files=files, data=data, headers=headers)
        print(res.text)
    except:
        print('could not be sent')


def capture_image():
    cap_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    response = cam.Streaming.channels[102].picture(method='get', type='opaque_data')

    img_file = 'data/' + cap_time + '.jpg'

    with open(img_file, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk: f.write(chunk)

    return {
        'captured_at' : cap_time,
        'image': open(img_file, 'rb')
    }


if __name__ == '__main__':
    # data = pd.read_csv(idx_file)
    #
    # for idx, row in data.iterrows():
    #     img = base_path + row['file_name']
    while True:
        payload = capture_image()
        post_data(payload)
        # break
