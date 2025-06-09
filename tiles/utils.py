import math

def lon2tile(lon, zoom):
    return int((lon + 180.0) / 360.0 * (2**zoom))

def lat2tile(lat, zoom):
    lat_rad = math.radians(lat)
    return int(
        (1.0 - math.log(math.tan(lat_rad) + 1 / math.cos(lat_rad)) / math.pi)
        / 2.0
        * (2**zoom)
    )

def tile2lon(x, zoom):
    return x / (2**zoom) * 360.0 - 180.0

def tile2lat(y, zoom):
    n = math.pi - 2.0 * math.pi * y / (2**zoom)
    return math.degrees(math.atan(math.sinh(n)))


if __name__ == "__main__":
    lat1_val, lon1_val = 50.383018073266285, 30.444404923939615  # Upper-left corner
    lat2_val, lon2_val = 50.3748645582537, 30.4618999860186      # Lower-right corner

    print(lon2tile(lon1_val, 10))
    print(lat2tile(lat1_val, 10))

    print(tile2lon(lon2tile(lon1_val, 10), 10))
    print(tile2lat(lat2tile(lat1_val, 10), 10))

