# pip install --upgrade pygeocoder

from pygeocoder import Geocoder

def get_zip_code(lat, lon):
    result = pd.Series()
    for i in range(len(list(lat))):
        result.set_value(label=i,value=Geocoder.reverse_geocode(lat[i], lon[i]).formatted_address.split(',')[2][4:9],takeable=False)
        
    return result