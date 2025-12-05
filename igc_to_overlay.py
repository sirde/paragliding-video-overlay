# Configure only this:

# MODE TEST
TEST_MODE = False  # True = Generates test_overlay.png, False = generates video

# TODO: make auto switch between summer/winter saving time
HEURE_ETE = 1  # Add 1 if summer time
TOTAL_FLIGHT_DIST = 78  # Total distance with "3 points de contournenment"
speed_acc = 16  # Acceleration factor for video, typically x16

# Url to igc, or local path to igc, at least one
file_url =  None #"https://www.syride.com/scripts/downloadIGC.php?idSession=2121335&key=0356809495924"
# file_path = None
# file_path = r"C:\Users\sirde\Dropbox\Parapente\Vercofly\Tracks 2025\CEDRIC-GERBER (2).igc"
file_path = r"C:\Users\sirde\Dropbox\Parapente\Overlay\2025-05-18-XCT-CGE-12.igc"
qualite_compression = 15  # valeur à augmenter pour avoir une meilleure qualité, 10 est bien suffisant

##
# Do not touch below
##


import mediapy as m
import urllib.request
import pixie
from math import pi
from matplotlib import cm
import io
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
from math import sin, cos, sqrt, atan2, radians
from datetime import date as date_creator
from aerofiles.igc import Reader
from scipy.interpolate import interp1d
from PIL import Image
import tempfile
import requests

urllib.request.urlretrieve("https://github.com/treeform/pixie-python/raw/master/examples/data/Ubuntu-Regular_1.ttf",
                           "Ubuntu-Regular_1.ttf")

A = 1920
B = 1080


def abtoxy(a, b=None):
    if isinstance(a, tuple):
        b = a[1]
        a = a[0]
    x = a - A // 2
    y = B - b
    return x, y


def xytoab(x, y=None, returnint=True, ref_xy=[0, 0]):
    if isinstance(x, tuple):
        y = x[1]
        x = x[0]
    a = x + A // 2 + ref_xy[0]
    b = B - y - ref_xy[1]
    if returnint:
        return round(a), round(b)
    else:
        return a, b


def rotate_cw_xy(x, y, theta, xy_center=None):
    if not (xy_center is None):
        x = x - xy_center[0]
        y = y - xy_center[1]
    t = theta * pi / 180
    return cos(t) * x + sin(t) * y + xy_center[0], cos(t) * y - sin(t) * x + xy_center[1]


def rotate(a, b, theta):
    x, y = abtoxy(a, b)
    x, y = rotate_cw_xy(x, y, -theta)
    return xytoab(x, y)


def create_origin_triangle_point_xy():
    t1x = -237
    t1y = 0

    t2x = -237
    t2y = 22

    t3x = -220
    t3y = 11
    return t1x, t1y, t2x, t2y, t3x, t3y


def tripath_to_strpath(t1a, t1b, t2a, t2b, t3a, t3b):
    return "M" + str(t1a) + ' ' + str(t1b) + ' L' + str(t2a) + ' ' + str(t2b) + ' L' + str(t3a) + ' ' + str(t3b) + ' H'


def create_compteur(ref_xy=[0, 0], nb_trait=24, traits_to_draw='all', rayon_compt=195, largeur_trait=20,
                    epaisseur_trait=7):
    compteur = pixie.Image(A, B)
    compteur.fill(pixie.parse_color("#000000"))

    paint = pixie.Paint(pixie.SOLID_PAINT)
    all_col_p, _ = precompute_colorwheel(nb_trait)
    if traits_to_draw == 'all':
        traits_to_draw = [True for _ in range(nb_trait)]
    else:
        assert len(traits_to_draw) == nb_trait

    ctx = compteur.new_context()
    ctx.stroke_style = paint
    ctx.line_width = epaisseur_trait

    x1, y1 = -rayon_compt - largeur_trait // 2, 11
    x2, y2 = -rayon_compt + largeur_trait // 2, 11

    xy_center = (0, 11)
    for i in range(nb_trait):
        angle_to_rot = 180.0 * i / (nb_trait - 1.0)
        new_xy = rotate_cw_xy(x1, y1, angle_to_rot, xy_center=xy_center)
        a1, b1 = xytoab(new_xy, ref_xy=ref_xy)
        new_xy = rotate_cw_xy(x2, y2, angle_to_rot, xy_center=xy_center)
        a2, b2 = xytoab(new_xy, ref_xy=ref_xy)

        if traits_to_draw[i]:
            paint.color = all_col_p[i]
            ctx.stroke_style = paint
        else:
            paint.color = pixie.parse_color("#FFFFFF")
            ctx.stroke_style = paint

        ctx.stroke_segment(a1, b1, a2, b2)

    return compteur


def create_pointeur_compteur(ref_xy=[0, 0], angle_to_rot=90):
    t1x, t1y, t2x, t2y, t3x, t3y = create_origin_triangle_point_xy()
    xy_center = (0, 11)
    t1x, t1y = rotate_cw_xy(t1x, t1y, angle_to_rot, xy_center=xy_center)
    t2x, t2y = rotate_cw_xy(t2x, t2y, angle_to_rot, xy_center=xy_center)
    t3x, t3y = rotate_cw_xy(t3x, t3y, angle_to_rot, xy_center=xy_center)

    t1a, t1b = xytoab(t1x, t1y, ref_xy=ref_xy)
    t2a, t2b = xytoab(t2x, t2y, ref_xy=ref_xy)
    t3a, t3b = xytoab(t3x, t3y, ref_xy=ref_xy)
    strpath = tripath_to_strpath(t1a, t1b, t2a, t2b, t3a, t3b)
    path = pixie.parse_path(strpath)
    mask = pixie.Mask(A, B)
    mask.fill_path(path)
    return mask


def mask_to_img(msk, color="#FFFFFF"):
    img = pixie.Image(A, B)
    if isinstance(color, str):
        color_pix = pixie.parse_color(color)
    else:
        color_pix = color
    img.fill(color_pix)
    img.mask_draw(msk)
    return img


def precompute_colorwheel(nb_trait):
    all_p_col = []
    cmap = cm.get_cmap('seismic')  # 'bwr' 'seismic'
    for i in range(nb_trait):
        ratio = i / (nb_trait - 1.0)
        rgba = cmap(ratio)
        all_p_col.append(pixie.Color(rgba[0], rgba[1], rgba[2], rgba[3]))
    return all_p_col, cmap


def create_full_compteur(vz, vz_min, vz_max, ref_xy=[0, 0], nb_trait=24):
    assert nb_trait % 2 == 0
    if vz >= 0:
        angle_up = min(90, 90 * vz / vz_max)
        angle_to_rot = angle_up + 90
        nb_trait_updown = angle_up / 90 * (nb_trait // 2)
        traits_to_draw = [False for i in range(nb_trait // 2)] + [True if i < nb_trait_updown else False for i in
                                                                  range(nb_trait // 2)]
    else:
        angle_down = min(90, 90 * vz / vz_min)
        angle_to_rot = 90 - angle_down
        nb_trait_updown = angle_down / 90 * (nb_trait // 2) + 1
        traits_to_draw = [True if i >= (nb_trait // 2) - nb_trait_updown else False for i in range(nb_trait // 2)] + [
            False for i in range(nb_trait // 2)]

    _, cmap = precompute_colorwheel(nb_trait)
    color_pointeur_rgba = cmap(angle_to_rot / 180)
    pix_color = pixie.Color(color_pointeur_rgba[0], color_pointeur_rgba[1],
                            color_pointeur_rgba[2], color_pointeur_rgba[3])

    pointeur_m = create_pointeur_compteur(ref_xy=ref_xy, angle_to_rot=angle_to_rot)
    pointeur = mask_to_img(pointeur_m, color=pix_color)
    compteur = create_compteur(ref_xy=ref_xy, traits_to_draw=traits_to_draw)
    compteur.draw(pointeur)
    return compteur


def base_compteur():
    ref_xy = [0, 0]
    pointeur_m = create_pointeur_compteur(ref_xy=ref_xy)
    pointeur = mask_to_img(pointeur_m)
    compteur = create_compteur(ref_xy=ref_xy)
    compteur.draw(pointeur)
    compteur.write_file('1.png')


def add_vario_to_img(img, vario_val, font=None):
    if font is None:
        font = pixie.read_font("Ubuntu-Regular_1.ttf")
        font.size = 80
        font.paint.color = pixie.parse_color('#FFFFFF')

    text = "{:.1f}".format(abs(vario_val))
    x_var, y_var = -55, 145
    a, b = xytoab(x_var, y_var)
    img.fill_text(font, text, bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    if vario_val < 0:
        textpm = '-'
        x_var, y_var = x_var - 35, y_var
        a, b = xytoab(x_var, y_var)
    else:
        textpm = '+'
        x_var, y_var = x_var - 45, y_var
        a, b = xytoab(x_var, y_var)
    img.fill_text(font, textpm, bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = -49, 60
    a, b = xytoab(x_var, y_var)
    font.size = 40
    font.paint.color = pixie.parse_color('#AAAAAA')
    img.fill_text(font, 'm / s', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = -75 + 19, 145 + 40
    a, b = xytoab(x_var, y_var)
    img.fill_text(font, 'VARIO', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    return img


def add_alti_to_img(img, alti, font=None):
    if font is None:
        font = pixie.read_font("Ubuntu-Regular_1.ttf")
        font.size = 80
        font.paint.color = pixie.parse_color('#FFFFFF')

    text = str(int(alti))
    x_var, y_var = round(-455 + 44.6 * (4 - len(text))), 145
    a, b = xytoab(x_var, y_var)
    img.fill_text(font, text, bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = -465, y_var + 40
    a, b = xytoab(x_var, y_var)
    font.size = 40
    font.paint.color = pixie.parse_color('#AAAAAA')
    img.fill_text(font, 'ALTITUDE', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = -320, 60
    a, b = xytoab(x_var, y_var)
    img.fill_text(font, 'm', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    return img


def add_speed_to_img(img, speed, font=None):
    if font is None:
        font = pixie.read_font("Ubuntu-Regular_1.ttf")
        font.size = 80
        font.paint.color = pixie.parse_color('#FFFFFF')

    text = str(int(speed))
    x_var, y_var = round(196 + 44.6 * (4 - len(text))), 145
    a, b = xytoab(x_var, y_var)
    img.fill_text(font, text, bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = 289, y_var + 40
    a, b = xytoab(x_var, y_var)
    font.size = 40
    font.paint.color = pixie.parse_color('#AAAAAA')
    img.fill_text(font, 'SPEED', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = 289, 60
    a, b = xytoab(x_var, y_var)
    img.fill_text(font, 'km/h', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    return img


def add_time_to_img(img, time, font=None):
    if font is None:
        font = pixie.read_font("Ubuntu-Regular_1.ttf")

        font.paint.color = pixie.parse_color('#FFFFFF')

    font.size = 55
    text = time
    x_var, y_var = -750 + 85, 145 + 40 + 8
    a, b = xytoab(x_var, y_var)
    img.fill_text(font, text, bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = -800 + 6, 145 + 40
    a, b = xytoab(x_var, y_var)
    font.size = 40
    font.paint.color = pixie.parse_color('#AAAAAA')
    img.fill_text(font, 'HOUR', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    return img


def add_flight_time_to_img(img, time, font=None):
    if font is None:
        font = pixie.read_font("Ubuntu-Regular_1.ttf")

        font.paint.color = pixie.parse_color('#FFFFFF')

    font.size = 55
    text = time
    x_var, y_var = -750 + 85, 145 + 40 + 8 - 62
    a, b = xytoab(x_var, y_var)
    img.fill_text(font, text, bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = -835 - 77, 145 + 40 - 62
    a, b = xytoab(x_var, y_var)
    font.size = 40
    font.paint.color = pixie.parse_color('#AAAAAA')
    img.fill_text(font, 'FLIGHT TIME', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    return img


def add_flight_dist_to_img(img, time, font=None):
    if font is None:
        font = pixie.read_font("Ubuntu-Regular_1.ttf")

        font.paint.color = pixie.parse_color('#FFFFFF')

    font.size = 55
    text = time
    x_var, y_var = -750 + 85, 145 + 40 + 8 - 62 - 62
    a, b = xytoab(x_var, y_var)
    img.fill_text(font, text, bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = -835 - 70, 145 + 40 - 62 - 62
    a, b = xytoab(x_var, y_var)
    font.size = 40
    font.paint.color = pixie.parse_color('#AAAAAA')
    img.fill_text(font, 'FLIGHT DIST', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    x_var, y_var = -835 + 70 + 193, 145 + 40 - 62 - 62
    a, b = xytoab(x_var, y_var)
    font.size = 40
    font.paint.color = pixie.parse_color('#AAAAAA')
    img.fill_text(font, 'Km', bounds=pixie.Vector2(500, 500), transform=pixie.translate(a, b))

    return img


def generate_dummy_igc(filename='dummy_flight.igc', num_points=60):
    """
    Generate a dummy IGC file with realistic paragliding data.

    Parameters:
    -----------
    filename : str
        Output filename for the IGC file
    num_points : int
        Number of GPS points to generate

    Returns:
    --------
    str : Path to the generated IGC file
    """
    from datetime import datetime, timedelta

    print(f"Generating dummy IGC file: {filename} with {num_points} points")

    # Start position (somewhere in the Alps)
    start_lat = 45.5  # degrees
    start_lon = 6.5   # degrees
    start_alt = 1500  # meters
    start_time = datetime(2025, 5, 18, 14, 0, 0)  # 14h00

    with open(filename, 'w') as f:
        # Write IGC header
        f.write("AXXX Example IGC file\n")
        f.write("HFDTE180525\n")  # Date: 18/05/2025
        f.write("HFPLTPILOTINCHARGE:Dummy Pilot\n")
        f.write("HFGTYGLIDERTYPE:Test Wing\n")
        f.write("HFGIDGLIDERID:TEST123\n")
        f.write("HFDTM100GPSDATUM:WGS-1984\n")

        # Generate flight points
        for i in range(num_points):
            # Time progression (1 second per point)
            current_time = start_time + timedelta(seconds=i)

            # Simulate flight path with some variation
            # Altitude: start at 1500m, climb to ~1650m, then descend
            if i < 30:
                altitude = start_alt + i * 5 + np.sin(i / 3) * 20
            else:
                altitude = start_alt + 150 - (i - 30) * 3 + np.sin(i / 3) * 20

            # Position: drift slightly (simulate wind)
            lat = start_lat + (i * 0.0001) + np.sin(i / 5) * 0.0002
            lon = start_lon + (i * 0.00015) + np.cos(i / 5) * 0.0002

            # Convert to IGC format
            # Format: BHHMMSS DDMMmmmN DDDMMmmmE A PPPPP GGGGG
            lat_deg = int(lat)
            lat_min = (lat - lat_deg) * 60
            lat_min_int = int(lat_min)
            lat_min_frac = int((lat_min - lat_min_int) * 1000)

            lon_deg = int(lon)
            lon_min = (lon - lon_deg) * 60
            lon_min_int = int(lon_min)
            lon_min_frac = int((lon_min - lon_min_int) * 1000)

            time_str = current_time.strftime("%H%M%S")
            lat_str = f"{lat_deg:02d}{lat_min_int:02d}{lat_min_frac:03d}N"
            lon_str = f"{lon_deg:03d}{lon_min_int:02d}{lon_min_frac:03d}E"
            press_alt = int(altitude)
            gps_alt = int(altitude + 5)  # GPS slightly different

            # B record (fix)
            f.write(f"B{time_str}{lat_str}{lon_str}A{press_alt:05d}{gps_alt:05d}\n")

    print(f"[OK] Dummy IGC generated: {filename}")
    return filename


def do_sin_compteur():
    font = pixie.read_font("Ubuntu-Regular_1.ttf")
    font.size = 80
    font.paint.color = pixie.parse_color('#FFFFFF')

    for i in tqdm(range(100)):
        t = -i * 12 * pi / 180
        vz = sin(t) * 3.5
        cmpt = create_full_compteur(vz, -3, 3, ref_xy=[0, 0], nb_trait=24)
        cmpt = add_vario_to_img(cmpt, vz)
        cmpt = add_alti_to_img(cmpt, (1 + i % 4) * '1')
        cmpt = add_speed_to_img(cmpt, (1 + i % 2) * '2')
        #         cmpt = add_time_to_img(cmpt,'09h'+str((10+i)%60))
        #         cmpt = add_flight_time_to_img(cmpt,'09h'+str((10+i)%60))
        #         cmpt = add_flight_dist_to_img(cmpt,str(i+50))
        cmpt.write_file('all_angle\\' + str(i) + '.png')


def str60(x):
    if x < 10:
        return '0' + str(x)
    else:
        return str(x)


def timesec_to_string(tsec, h_ete=0):
    h = tsec // 3600
    tsec = tsec - h * 3600
    m = tsec // 60
    s = int(tsec - 6 * 60)

    return str60(int(h) + h_ete) + 'h' + str60(int(m))


def gen_img_from_smoothed_list(all_speed, all_vz, all_alti, all_time, all_time_full,
                                 graph_config=None):
    """
    Generate overlay images with optional time-series graphs.

    Optimized for performance by reusing temp file and pre-parsing colors.

    Parameters:
    -----------
    all_speed, all_vz, all_alti : np.array
        Flight metric arrays (reshaped to video framerate)
    all_time, all_time_full : np.array
        Time arrays
    graph_config : dict or None
        If provided, adds time-series graphs to overlay

    Yields:
    -------
    np.array : RGB image array for each frame
    """
    # Pre-compute reusable elements for performance
    font = pixie.read_font("Ubuntu-Regular_1.ttf")
    black_color = pixie.parse_color("#000000")
    temp_file = os.path.join(tempfile.gettempdir(), 'overlay_frame.png')

    for i in tqdm(range(all_speed.shape[0])):
        # Create blank black background (reuse parsed color)
        cmpt = pixie.Image(A, B)
        cmpt.fill(black_color)

        # Add text overlays
        cmpt = add_time_to_img(cmpt, timesec_to_string(all_time[i], h_ete=HEURE_ETE), font)
        cmpt = add_flight_time_to_img(cmpt, timesec_to_string(all_time[i] - all_time_full[0], h_ete=0), font)
        cmpt = add_flight_dist_to_img(cmpt, str(int(
            TOTAL_FLIGHT_DIST * (all_time[i] - all_time_full[0]) / (all_time_full[-1] - all_time_full[0]))), font)

        # Add time-series graphs if configured
        if graph_config is not None:
            cmpt = add_time_series_graphs(
                cmpt, i,
                all_speed, all_vz, all_alti,
                graph_config, font
            )

        # Write to disk and read back (reuse same temp file path)
        cmpt.write_file(temp_file)
        img = Image.open(temp_file)
        yield np.array(img)[:, :, :3]



def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = np.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]


def get_date_time_dif(start_time, stop_time):
    date = date_creator(1, 1, 1)
    datetime1 = datetime.combine(date, start_time)
    datetime2 = datetime.combine(date, stop_time)
    time_elapsed = datetime1 - datetime2
    return time_elapsed.total_seconds()


def compute_dist(lat1, lon1, lat2, lon2, rad=True):
    if not (rad):
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        lat2 = radians(lat2)
        lon2 = radians(lon2)
    # approximate radius of earth in km
    R = 6373.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return 1000 * distance


def get_all_time_end():
    path = "D:\\sur le disque dur de anne\\rush_2021_05_23_montlamb revanche\\Camera01\\"
    path = "D:\\sur le disque dur de anne\\rush_12_06_21_bivigap_lac\\Camera01\\"
    for file_name in os.listdir(path):
        if "_11_" in file_name:
            lm = os.path.getmtime(path + file_name)
            d = datetime.fromtimestamp(lm).strftime("%I:%M:%S")

            print(file_name, d)


# TODO: Document this
def remove_zero_from_alti(alti):
    if alti[0] < 10:
        alti[0] = (alti[1] + alti[2]) / 2
    if alti[-1] < 10:
        alti[-1] = (alti[-2] + alti[-3]) / 2
    for i, alt in enumerate(alti):
        if alt < 10:
            alti[i] = (alti[i - 1] + alti[i + 1]) / 2
    return alti


def read_igc(file_url=None, file_path=None):
    if file_url is not None:
        igc = requests.get(file_url)
        print("Getting ", file_url)
        parsed_igc_file = Reader().read(io.StringIO(igc.text))
    elif file_path is not None:
        print("Reading ", file_path)
        with open(file_path, 'r') as igc_file:
            parsed_igc_file = Reader().read(igc_file)

    assert(len(parsed_igc_file['fix_records'][1]) > 0)

    print('igc_file read')
    previous_lat = 0
    previous_lon = 0

    all_speed = np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_vz = np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_alti = np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_time = [0 for _ in range(len(parsed_igc_file['fix_records'][1]))]

    for i, record in tqdm(enumerate(parsed_igc_file['fix_records'][1])):
        record['time'] = record['time'].replace(hour=record['time'].hour + 1)
        if previous_lon == 0:
            # Init previous value with first record
            previous_lat = record['lat']
            previous_lon = record['lon']
            previous_datetime = record['time']
            previous_alt_gps = record['gps_alt']
            previous_alt_baro = record['pressure_alt']
            all_time[i] = record['time']
        else:
            dxy = abs(compute_dist(previous_lat, previous_lon, record['lat'], record['lon'], rad=False))
            dz = previous_alt_baro - record['pressure_alt']
            dz = previous_alt_gps - record['gps_alt']
            dt = get_date_time_dif(record['time'], previous_datetime)
            if dt > 1: print('Delta T between points > 1 at time %s: : %s' % (record['time'], dt))

            all_speed[i] = min(100, sqrt(dxy ** 2 + 0 * dz ** 2) / dt * 3.6)
            all_vz[i] = dz / dt
            all_alti[i] = record['gps_alt']
            all_time[i] = record['time']

            previous_lat = record['lat']
            previous_lon = record['lon']
            previous_datetime = record['time']
            previous_alt_gps = record['gps_alt']
            previous_alt_baro = record['pressure_alt']


    return all_speed, all_vz, remove_zero_from_alti(all_alti), all_time


def reshape_array(arr, time_vid):
    nb_img_by_sec = 24

    t_true = np.linspace(time_vid[0], time_vid[-1], num=len(time_vid), endpoint=True)
    t_inter = np.linspace(time_vid[0], time_vid[-1], num=int(len(time_vid) * nb_img_by_sec / speed_acc), endpoint=True)
    f = interp1d(t_true, arr, kind='cubic')

    return f(t_inter)


def smooth_igc_output(L_all):
    all_ret = []
    for l_val in L_all:
        l_val[0] = np.mean(l_val[:int(len(l_val) / 10)])
        smoothed = smooth(l_val, 50, 'hanning')
        all_ret.append(smoothed)
    return all_ret


def plot_smooth_non_smooth(smooth, non_smooth):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(18, 9))
    plt.plot(non_smooth)

    plt.plot(smooth)

    plt.show()


def get_last_date_of_all_raw_file(path_raw_file):
    delta_time_writing = 20
    all_ending_time = []
    for file in os.listdir(path_raw_file):
        if "_11_" in file:
            time_end = os.path.getmtime(path_raw_file + '\\' + file)
            all_ending_time.append(datetime.fromtimestamp(time_end - delta_time_writing).time())
    return all_ending_time


def convert_time_to_sec(all_time):
    for i in range(len(all_time)):
        all_time[i] = all_time[i].hour * 3600 + all_time[i].minute * 60 + all_time[i].second
    return np.array(all_time, dtype=np.float32)


##
# GRAPH DRAWING FUNCTIONS
##

def draw_grid_lines(img, left_x, right_x, bottom_y, top_y, y_min, y_max, num_lines=5):
    """Draw horizontal grid lines across the graph."""
    ctx = img.new_context()
    paint = pixie.Paint(pixie.SOLID_PAINT)
    paint.color = pixie.Color(0.2, 0.2, 0.2, 0.8)  # Dark gray
    ctx.stroke_style = paint
    ctx.line_width = 1

    for i in range(num_lines):
        # Calculate y position in data space
        y_data = y_min + (y_max - y_min) * i / (num_lines - 1)

        # Convert to screen space
        y_screen = bottom_y + (top_y - bottom_y) * (y_data - y_min) / (y_max - y_min)

        # Convert to ab coordinates
        a1, b = xytoab(left_x, y_screen)
        a2, _ = xytoab(right_x, y_screen)

        # Draw line
        ctx.stroke_segment(a1, b, a2, b)


def draw_line_graph(img, data_slice, left_x, right_x, bottom_y, top_y,
                    y_min, y_max, color_line):
    """Draw the line connecting data points."""
    if len(data_slice) < 2:
        return

    ctx = img.new_context()
    paint = pixie.Paint(pixie.SOLID_PAINT)
    paint.color = pixie.parse_color(color_line)
    ctx.stroke_style = paint
    ctx.line_width = 3

    # Map data points to screen coordinates
    for i in range(len(data_slice) - 1):
        # Current point
        x1_norm = i / (len(data_slice) - 1)  # Normalize to 0-1
        x1_screen = left_x + (right_x - left_x) * x1_norm
        y1_data = data_slice[i]
        y1_norm = (y1_data - y_min) / (y_max - y_min)  # Normalize to 0-1
        y1_screen = bottom_y + (top_y - bottom_y) * y1_norm

        # Next point
        x2_norm = (i + 1) / (len(data_slice) - 1)
        x2_screen = left_x + (right_x - left_x) * x2_norm
        y2_data = data_slice[i + 1]
        y2_norm = (y2_data - y_min) / (y_max - y_min)
        y2_screen = bottom_y + (top_y - bottom_y) * y2_norm

        # Convert to ab and draw
        a1, b1 = xytoab(x1_screen, y1_screen)
        a2, b2 = xytoab(x2_screen, y2_screen)
        ctx.stroke_segment(a1, b1, a2, b2)


def draw_current_marker(img, data_slice, marker_position,
                        left_x, right_x, bottom_y, top_y,
                        y_min, y_max):
    """Draw vertical line at current time position."""
    if len(data_slice) == 0:
        return

    # Calculate x position of marker
    x_norm = marker_position / (len(data_slice) - 1) if len(data_slice) > 1 else 0.5
    x_screen = left_x + (right_x - left_x) * x_norm

    # Draw vertical line
    ctx = img.new_context()
    paint_line = pixie.Paint(pixie.SOLID_PAINT)
    paint_line.color = pixie.Color(1.0, 1.0, 1.0, 0.8)  # White, semi-transparent
    ctx.stroke_style = paint_line
    ctx.line_width = 2

    a, b_bottom = xytoab(x_screen, bottom_y)
    _, b_top = xytoab(x_screen, top_y)
    ctx.stroke_segment(a, b_bottom, a, b_top)

    # Draw dot at current value using a small filled rectangle
    if marker_position < len(data_slice):
        y_data = data_slice[marker_position]
        y_norm = (y_data - y_min) / (y_max - y_min)
        y_screen = bottom_y + (top_y - bottom_y) * y_norm

        # Draw small filled square as marker
        a, b = xytoab(x_screen, y_screen)
        ctx_dot = img.new_context()
        paint_dot = pixie.Paint(pixie.SOLID_PAINT)
        paint_dot.color = pixie.Color(1.0, 1.0, 0.0, 1.0)  # Yellow
        ctx_dot.fill_style = paint_dot
        ctx_dot.fill_rect(a-4, b-4, 8, 8)  # 8x8 pixel square


def draw_y_axis_scale(img, left_x, bottom_y, top_y, y_min, y_max,
                       num_ticks=5, font=None, unit=""):
    """Draw Y-axis scale with tick marks and labels."""
    if font is None:
        font = pixie.read_font("Ubuntu-Regular_1.ttf")

    font.size = 14
    font.paint.color = pixie.parse_color('#888888')

    for i in range(num_ticks):
        # Calculate y value and position
        y_value = y_min + (y_max - y_min) * i / (num_ticks - 1)
        y_screen = bottom_y + (top_y - bottom_y) * (y_value - y_min) / (y_max - y_min)

        # Draw tick mark (subtle)
        ctx = img.new_context()
        paint = pixie.Paint(pixie.SOLID_PAINT)
        paint.color = pixie.Color(0.3, 0.3, 0.3, 0.6)
        ctx.stroke_style = paint
        ctx.line_width = 1

        a_tick, b_tick = xytoab(left_x - 5, y_screen)
        a_graph, _ = xytoab(left_x, y_screen)
        ctx.stroke_segment(a_tick, b_tick, a_graph, b_tick)

        # Draw label with unit (e.g., "42km/h")
        if unit:
            label_text = f"{int(y_value)}{unit}"
        else:
            label_text = f"{int(y_value)}"

        label_x = left_x - 65
        label_y = y_screen + 5
        a, b = xytoab(label_x, label_y)
        img.fill_text(font, label_text, bounds=pixie.Vector2(80, 30),
                      transform=pixie.translate(a, b))


def draw_x_axis_scale(img, center_x, left_x, right_x, bottom_y,
                       time_window_sec=10, num_ticks=3, font=None):
    """Draw X-axis time scale in minutes format."""
    if font is None:
        font = pixie.read_font("Ubuntu-Regular_1.ttf")

    font.size = 16
    font.paint.color = pixie.parse_color('#888888')

    # Convert seconds to minutes for display
    time_window_min = time_window_sec / 60.0

    for i in range(num_ticks):
        # Calculate time offset and x position
        time_offset_sec = -time_window_sec + (2 * time_window_sec) * i / (num_ticks - 1)
        time_offset_min = time_offset_sec / 60.0
        x_norm = i / (num_ticks - 1)
        x_screen = left_x + (right_x - left_x) * x_norm

        # Draw tick mark (subtle)
        ctx = img.new_context()
        paint = pixie.Paint(pixie.SOLID_PAINT)
        paint.color = pixie.Color(0.3, 0.3, 0.3, 0.6)
        ctx.stroke_style = paint
        ctx.line_width = 1

        a_tick, b_bottom = xytoab(x_screen, bottom_y)
        _, b_tick = xytoab(x_screen, bottom_y - 5)
        ctx.stroke_segment(a_tick, b_bottom, a_tick, b_tick)

        # Draw label
        if abs(time_offset_min) < 0.01:  # Center position
            label_text = "Now"
        elif time_offset_min > 0:
            label_text = f"+{int(abs(time_offset_min))}min"
        else:
            label_text = f"-{int(abs(time_offset_min))}min"

        label_x = x_screen - 20
        label_y = bottom_y - 25
        a, b = xytoab(label_x, label_y)
        img.fill_text(font, label_text, bounds=pixie.Vector2(60, 30),
                      transform=pixie.translate(a, b))


def draw_graph_labels(img, title, unit, center_x, center_y,
                      width, height, font):
    """Draw title and unit labels."""
    # Title at top center
    font.size = 30
    font.paint.color = pixie.parse_color('#CCCCCC')
    title_x = center_x - width // 2 + 100
    title_y = center_y + height // 2 + 30
    a, b = xytoab(title_x - 100, title_y)  # Offset for centering
    img.fill_text(font, title, bounds=pixie.Vector2(200, 50),
                  transform=pixie.translate(a, b))

    # Unit at bottom right (now removed - shown with value instead)
    # font.size = 20
    # font.paint.color = pixie.parse_color('#999999')
    # unit_x = center_x + width // 2 - 40
    # unit_y = center_y - height // 2 - 10
    # a, b = xytoab(unit_x, unit_y)
    # img.fill_text(font, unit, bounds=pixie.Vector2(100, 30),
    #               transform=pixie.translate(a, b))


def draw_time_series_graph(img, data_slice, marker_position,
                           center_x, center_y, width, height,
                           y_min, y_max,
                           title, unit, color_line="#00AAFF",
                           font=None, show_current_value=False, current_value=None,
                           time_window_sec=300):
    """
    Draw a time-series graph on the image.

    Parameters:
    -----------
    img : pixie.Image
        Image to draw on
    data_slice : np.array
        Data points for the time window
    marker_position : int
        Index of current moment in data_slice
    center_x, center_y : float
        Center position in xy coordinates
    width, height : int
        Graph dimensions in pixels
    y_min, y_max : float
        Y-axis scale (fixed for entire flight)
    title : str
        Graph title
    unit : str
        Unit label (e.g., "km/h")
    color_line : str
        Color for the line graph
    font : pixie.Font
        Font for labels
    show_current_value : bool
        Whether to display the current value prominently on the graph
    current_value : float
        Current value to display (if None, uses data_slice[marker_position])
    time_window_sec : int
        Time window in seconds for X-axis (±this value, default 300 = ±5 min)

    Returns:
    --------
    pixie.Image : Modified image with graph drawn
    """
    if font is None:
        font = pixie.read_font("Ubuntu-Regular_1.ttf")

    # Calculate graph bounds in xy coordinates
    left_x = center_x - width // 2
    right_x = center_x + width // 2
    bottom_y = center_y - height // 2
    top_y = center_y + height // 2

    # Convert to ab coordinates
    left_a, bottom_b = xytoab(left_x, bottom_y)
    right_a, top_b = xytoab(right_x, top_y)

    # Draw background rectangle (semi-transparent dark)
    ctx = img.new_context()
    paint_bg = pixie.Paint(pixie.SOLID_PAINT)
    paint_bg.color = pixie.Color(0.1, 0.1, 0.1, 0.5)  # Dark semi-transparent
    ctx.fill_style = paint_bg
    ctx.fill_rect(left_a, top_b, width, height)

    # Draw grid lines (horizontal reference lines)
    draw_grid_lines(img, left_x, right_x, bottom_y, top_y,
                    y_min, y_max, num_lines=5)

    # Draw Y-axis scale (left side)
    draw_y_axis_scale(img, left_x, bottom_y, top_y, y_min, y_max,
                      num_ticks=5, font=font, unit=unit)

    # Draw X-axis scale (bottom)
    # Use the time_window_sec parameter passed to this function
    draw_x_axis_scale(img, center_x, left_x, right_x, bottom_y,
                      time_window_sec=time_window_sec, num_ticks=3, font=font)

    # Draw axes border
    ctx = img.new_context()
    paint_border = pixie.Paint(pixie.SOLID_PAINT)
    paint_border.color = pixie.Color(0.3, 0.3, 0.3, 1.0)  # Gray
    ctx.stroke_style = paint_border
    ctx.line_width = 2
    ctx.stroke_rect(left_a, top_b, width, height)

    # Draw line graph
    draw_line_graph(img, data_slice, left_x, right_x, bottom_y, top_y,
                    y_min, y_max, color_line)

    # Draw current value marker (vertical line)
    draw_current_marker(img, data_slice, marker_position,
                        left_x, right_x, bottom_y, top_y,
                        y_min, y_max)

    # Draw labels
    draw_graph_labels(img, title, unit, center_x, center_y,
                      width, height, font)

    # Draw current value prominently if requested (above the graph like in reference)
    if show_current_value:
        if current_value is None and marker_position < len(data_slice):
            current_value = data_slice[marker_position]

        if current_value is not None:
            # Format value with unit based on type
            if title == "SPEED":
                value_text = f"{current_value:.1f}{unit}"  # 2 decimal places for speed
            elif title == "ALTITUDE":
                value_text = f"{int(current_value)}{unit}"  # Integer for altitude
            else:  # VARIO or others
                value_text = f"{current_value:.1f}{unit}"  # 1 decimal place

            # Display above the graph, centered
            font.size = 24
            font.paint.color = pixie.parse_color('#FFFFFF')

            # Position above graph, centered horizontally
            if title == "ALTITUDE":
                 value_x = center_x - 10  # Center offset for alignment
            else:
                value_x = center_x - 40  # Center offset for alignment
            value_y = top_y + 25  # Above the graph
            a, b = xytoab(value_x, value_y)
            img.fill_text(font, value_text, bounds=pixie.Vector2(150, 50),
                          transform=pixie.translate(a, b))

    return img


##
# GRAPH UTILITY FUNCTIONS
##

def compute_graph_parameters(all_speed, all_vz, all_alti, speed_acc, fps=24):
    """
    Compute min/max values and time window parameters for graphs.

    Parameters:
    -----------
    all_speed, all_vz, all_alti : np.array
        Full flight data arrays
    speed_acc : int
        Video acceleration factor (e.g., 16x)
    fps : int
        Frames per second (24)

    Returns:
    --------
    dict : Configuration for graph rendering
    """
    # Calculate min/max with 5% padding for visual clarity
    speed_range = np.max(all_speed) - np.min(all_speed)
    vz_range = np.max(all_vz) - np.min(all_vz)
    alti_range = np.max(all_alti) - np.min(all_alti)

    # Time window: 300 seconds (5 minutes) each side = 10 minutes total display
    time_window_sec = 300  # ±5 minutes

    graph_config = {
        'speed_min': np.min(all_speed) - 0.05 * speed_range,
        'speed_max': np.max(all_speed) + 0.05 * speed_range,
        'vz_min': np.min(all_vz) - 0.05 * vz_range,
        'vz_max': np.max(all_vz) + 0.05 * vz_range,
        'alti_min': np.min(all_alti) - 0.05 * alti_range,
        'alti_max': np.max(all_alti) + 0.05 * alti_range,
        'time_window_frames': int((time_window_sec / speed_acc) * fps),  # frames for time window
        'time_window_sec': time_window_sec,  # Time window in seconds
        'fps': fps
    }

    return graph_config


def extract_time_window(data_array, current_index, window_frames):
    """
    Extract time window centered on current_index.

    Handles edge cases at beginning/end of flight.

    Parameters:
    -----------
    data_array : np.array
        Full data array
    current_index : int
        Current frame index
    window_frames : int
        Number of frames for half-window (e.g., 10sec worth)

    Returns:
    --------
    tuple: (data_slice, marker_position)
        data_slice: numpy array of data for time window
        marker_position: index within slice where current moment is (0 to len-1)
    """
    total_frames = len(data_array)

    # Calculate window bounds
    start_idx = max(0, current_index - window_frames)
    end_idx = min(total_frames, current_index + window_frames + 1)

    # Extract slice
    data_slice = data_array[start_idx:end_idx]

    # Calculate where current moment is within the slice
    marker_position = current_index - start_idx

    return data_slice, marker_position


def add_time_series_graphs(img, current_index,
                            all_speed, all_vz, all_alti,
                            graph_config, font):
    """
    Add all three time-series graphs to the image.

    Parameters:
    -----------
    img : pixie.Image
        Base image with existing elements
    current_index : int
        Current frame index
    all_speed, all_vz, all_alti : np.array
        Full flight data arrays
    graph_config : dict
        Precomputed graph parameters
    font : pixie.Font
        Font for labels

    Returns:
    --------
    pixie.Image : Image with graphs added
    """
    window_frames = graph_config['time_window_frames']

    # Extract time windows for each metric
    speed_slice, speed_marker = extract_time_window(all_speed, current_index, window_frames)
    vz_slice, vz_marker = extract_time_window(all_vz, current_index, window_frames)
    alti_slice, alti_marker = extract_time_window(all_alti, current_index, window_frames)

    # Graph dimensions and position - 20% smaller, on the right
    graph_width = 320  # 20% smaller (was 400)
    graph_height = 144  # 20% smaller (was 180)
    graph_y = 120  # Bottom area position

    # Get time window from config (default to 300 seconds = ±5 min)
    time_window_sec = graph_config.get('time_window_sec', 300)

    # Draw all three graphs shifted left to show +5min label

    # Draw altitude graph (left of trio)
    img = draw_time_series_graph(
        img, alti_slice, alti_marker,
        center_x=-100, center_y=graph_y,  # Shifted left
        width=graph_width, height=graph_height,
        y_min=graph_config['alti_min'],
        y_max=graph_config['alti_max'],
        title="ALTITUDE", unit="m",
        color_line="#4CAF50",  # Green
        font=font,
        show_current_value=True,
        current_value=alti_slice[alti_marker] if alti_marker < len(alti_slice) else None,
        time_window_sec=time_window_sec
    )

    # Draw vario graph (center of trio)
    img = draw_time_series_graph(
        img, vz_slice, vz_marker,
        center_x=320, center_y=graph_y,  # Shifted left
        width=graph_width, height=graph_height,
        y_min=graph_config['vz_min'],
        y_max=graph_config['vz_max'],
        title="VARIO", unit="m/s",
        color_line="#FF5722",  # Red/Orange
        font=font,
        show_current_value=True,
        current_value=vz_slice[vz_marker] if vz_marker < len(vz_slice) else None,
        time_window_sec=time_window_sec
    )

    # Draw speed graph (rightmost)
    img = draw_time_series_graph(
        img, speed_slice, speed_marker,
        center_x=740, center_y=graph_y,  # Shifted left for +5min
        width=graph_width, height=graph_height,
        y_min=graph_config['speed_min'],
        y_max=graph_config['speed_max'],
        title="SPEED", unit="km/h",
        color_line="#2196F3",  # Blue
        font=font,
        show_current_value=True,
        current_value=speed_slice[speed_marker] if speed_marker < len(speed_slice) else None,
        time_window_sec=time_window_sec
    )

    return img


if __name__ == '__main__':

    # Create output folder if it doesn't exist
    output_folder = 'output'
    os.makedirs(output_folder, exist_ok=True)

    # Initialize IGC
    if TEST_MODE:
        print("=" * 60)
        print("Test Mode activated")
        print("=" * 60)
        dummy_igc = generate_dummy_igc('dummy_test_flight.igc', num_points=60)
        igc_source = dummy_igc
        output_name = 'test_overlay.png'
    else:
        print("=" * 60)
        print("Video mode activated")
        print("=" * 60)
        igc_source = file_path
        # Extract base name from input file and use for output
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        output_name = os.path.join(output_folder, f'{base_name}_overlay.mp4')

    # Process data (same for both modes)
    print('def ok')
    all_speed, all_vz, all_alti, all_time = read_igc(file_url=file_url if not TEST_MODE else None,
                                                       file_path=igc_source)
    print('\n process_igc ok')

    all_speed_smooth, all_vz_smooth, all_alti_smooth = smooth_igc_output([all_speed, all_vz, all_alti])
    print('igc smoothed')

    all_vz2 = np.zeros(all_alti_smooth.shape[0])
    for i, alti in enumerate(list(all_alti_smooth)):
        if i >= all_alti_smooth.shape[0] - 1:
            pass
        else:
            all_vz2[i] = all_alti_smooth[i + 1] - all_alti_smooth[i]

    print('smoothed vario ok')

    speed_vid = all_speed_smooth
    vz_vid = all_vz2
    alti_vid = all_alti_smooth
    time_vid = convert_time_to_sec(all_time)
    print('time to sec ok')

    time_vid_reshaped = reshape_array(time_vid, time_vid)
    time_vid_full_reshaped = reshape_array(all_time, all_time)
    speed_vid_reshaped = reshape_array(speed_vid, time_vid)
    vz_vid_reshaped = reshape_array(vz_vid, time_vid)
    alti_vid_reshaped = reshape_array(alti_vid, time_vid)
    print('reshaped ok')

    # Compute graph parameters
    graph_config = compute_graph_parameters(
        speed_vid_reshaped, vz_vid_reshaped, alti_vid_reshaped,
        speed_acc=speed_acc, fps=24
    )
    print('graph config computed')
    print(f"  Time window: +/-{graph_config['time_window_sec']/speed_acc:.2f} real seconds = {graph_config['time_window_frames']} frames")
    print(f"  Speed range: {graph_config['speed_min']:.1f} - {graph_config['speed_max']:.1f} km/h")
    print(f"  Vario range: {graph_config['vz_min']:.1f} - {graph_config['vz_max']:.1f} m/s")
    print(f"  Altitude range: {graph_config['alti_min']:.0f} - {graph_config['alti_max']:.0f} m")

    # Generate frames
    img_gen = gen_img_from_smoothed_list(
        speed_vid_reshaped, vz_vid_reshaped, alti_vid_reshaped,
        time_vid_reshaped, time_vid_full_reshaped,
        graph_config=graph_config
    )

    # Video processing
    if TEST_MODE:
        # Save 30th frame only
        print(f"\nSaving frame 30 as {output_name}...")
        target_frame = 29  # 0-indexed
        for i, frame in enumerate(img_gen):
            if i == target_frame:
                img_pil = Image.fromarray(frame, 'RGB')
                img_pil.save(output_name)
                print(f"[OK] Frame {target_frame + 1} saved as {output_name}")
                break
            elif i > target_frame:
                break
        print("\nPour générer la vidéo complète, mettez TEST_MODE = False")
    else:
        # Save full video
        print(f"\nGenerating video: {output_name}")
        m.write_video(output_name, img_gen, fps=24, qp=qualite_compression)
        print(f"\nVideo ok: {output_name}")






