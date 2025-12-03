##
# REMPLIR UNIQUEMENT LES 4 PROCHAINES LIGNES
##
from idlelib.outwin import file_line_pats

# MODE TEST - Mettre à True pour générer une seule image de test
TEST_MODE = True  # True = génère test_overlay.png, False = génère la vidéo complète

HEURE_ETE = 1  # rajoute 1 à l'heure si on est à l'heure d'été
TOTAL_FLIGHT_DIST = 78  # la distance total parcouru en mode "3 points de contournenment"
speed_acc = 16  # l'accélération utilisée pour la vidéo, moi je suis en X16
file_url =  None #"https://www.syride.com/scripts/downloadIGC.php?idSession=2121335&key=0356809495924"  # day2
# file_path = None
# file_path = r"C:\Users\sirde\Dropbox\Parapente\Vercofly\Tracks 2025\CEDRIC-GERBER (2).igc"
file_path = r"C:\Users\sirde\Dropbox\Parapente\Overlay\2025-05-18-XCT-CGE-12.igc"
qualite_compression = 15  # valeur à augmenter pour avoir une meilleure qualité, 10 est bien suffisant

##
# NE PAS TOUCHER LA SUITE, UNIQUEMENT LANCER LA CELLULE AVEC LE BOUTON PLAY
##


# !command - v
# ffmpeg > / dev / null | | (apt update & & apt install -y ffmpeg)

import mediapy as m
import numpy as np
import urllib.request

urllib.request.urlretrieve("https://github.com/treeform/pixie-python/raw/master/examples/data/Ubuntu-Regular_1.ttf",
                           "Ubuntu-Regular_1.ttf")
import pixie
from math import pi, cos, sin, tan
from matplotlib import cm
from tqdm import tqdm
import os
import io

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


def generate_test_image(output_filename='test_overlay.png'):
    """
    Génère une seule image de test avec des valeurs représentatives
    pour ajuster les positions des éléments sans traiter tout le fichier IGC
    """
    print(f"Génération de l'image de test: {output_filename}")

    # Valeurs de test représentatives
    test_vz = 2.3          # Variomètre: +2.3 m/s (montée)
    test_altitude = 1847   # Altitude: 1847m
    test_speed = 42        # Vitesse: 42 km/h
    test_time = '14h27'    # Heure: 14h27
    test_flight_time = '01h23'  # Temps de vol: 1h23
    test_distance = 45     # Distance: 45 km

    # Créer l'overlay complet
    cmpt = create_full_compteur(test_vz, -3, 3, ref_xy=[0, 0], nb_trait=24)
    cmpt = add_vario_to_img(cmpt, test_vz)
    cmpt = add_alti_to_img(cmpt, test_altitude)
    cmpt = add_speed_to_img(cmpt, test_speed)
    cmpt = add_time_to_img(cmpt, test_time)
    cmpt = add_flight_time_to_img(cmpt, test_flight_time)
    cmpt = add_flight_dist_to_img(cmpt, str(test_distance))

    # Sauvegarder l'image
    cmpt.write_file(output_filename)
    print(f"[OK] Image de test generee: {output_filename}")
    print(f"  Resolution: {A}x{B}")
    print(f"  Valeurs de test utilisees:")
    print(f"    - Vario: {test_vz:+.1f} m/s")
    print(f"    - Altitude: {test_altitude} m")
    print(f"    - Vitesse: {test_speed} km/h")
    print(f"    - Heure: {test_time}")
    print(f"    - Temps de vol: {test_flight_time}")
    print(f"    - Distance: {test_distance} km")


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


def gen_img_from_smoothed_list(all_speed, all_vz, all_alti, all_time, all_time_full):
    # Create a temporary file path that works on all platforms
    temp_file = os.path.join(tempfile.gettempdir(), 'tmp.png')

    for i in tqdm(range(all_speed.shape[0])):
        cmpt = create_full_compteur(all_vz[i], -3, 3, ref_xy=[0, 0], nb_trait=24)
        cmpt = add_vario_to_img(cmpt, all_vz[i])
        cmpt = add_alti_to_img(cmpt, str(int(all_alti[i])))
        cmpt = add_speed_to_img(cmpt, str(int(all_speed[i])))
        cmpt = add_time_to_img(cmpt, timesec_to_string(all_time[i], h_ete=HEURE_ETE))
        cmpt = add_flight_time_to_img(cmpt, timesec_to_string(all_time[i] - all_time_full[0], h_ete=0))
        cmpt = add_flight_dist_to_img(cmpt, str(int(
            TOTAL_FLIGHT_DIST * (all_time[i] - all_time_full[0]) / (all_time_full[-1] - all_time_full[0]))))
        cmpt.write_file(temp_file)
        img = Image.open(temp_file)
        yield np.array(img)[:, :, :3]


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

import numpy
import requests


def smooth(x, window_len=11, window='hanning'):
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")
    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")
    if window_len < 3:
        return x
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")
    s = numpy.r_[2 * x[0] - x[window_len - 1::-1], x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':  # moving average
        w = numpy.ones(window_len, 'd')
    else:
        w = eval('numpy.' + window + '(window_len)')
    y = numpy.convolve(w / w.sum(), s, mode='same')
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
        parsed_igc_file = Reader().read(io.StringIO(igc.text))
    elif file_path is not None:
        with open(file_path, 'r') as igc_file:
            parsed_igc_file = Reader().read(igc_file)

    print('igc_file created')
    previous_lat = 0
    previous_lon = 0

    all_speed = np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_vz = np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_alti = np.zeros(len(parsed_igc_file['fix_records'][1]))
    all_time = [0 for _ in range(len(parsed_igc_file['fix_records'][1]))]

    for i, record in tqdm(enumerate(parsed_igc_file['fix_records'][1])):
        record['time'] = record['time'].replace(hour=record['time'].hour + 1)
        if previous_lon == 0:
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
            if dt > 1: print('dtttttt>1 ', dt)

            all_speed[i] = min(100, sqrt(dxy ** 2 + 0 * dz ** 2) / dt * 3.6)
            all_vz[i] = dz / dt
            all_alti[i] = record['gps_alt']
            all_time[i] = record['time']

            previous_lat = record['lat']
            previous_lon = record['lon']
            previous_datetime = record['time']
            previous_alt_gps = record['gps_alt']
            previous_alt_baro = record['pressure_alt']

            # if previous_datetime.hour>10 :
            #     all_speed = all_speed[:i]
            #     all_vz = all_vz[:i]
            #     all_alti = all_alti[:i]
            #     break
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


if __name__ == '__main__':

    if TEST_MODE:
        # MODE TEST: Générer une seule image pour ajuster les positions
        print("=" * 60)
        print("MODE TEST ACTIVÉ")
        print("=" * 60)
        generate_test_image('test_overlay.png')
        print("\nPour générer la vidéo complète, mettez TEST_MODE = False")
    else:
        # MODE NORMAL: Générer la vidéo complète
        print("=" * 60)
        print("MODE VIDÉO - Génération de l'overlay complet")
        print("=" * 60)

        print('def ok')
        all_speed, all_vz, all_alti, all_time = read_igc(file_url=file_url, file_path=file_path)
        print('\n process_igc ok')

        all_speed_smooth, all_vz_smooth, all_alti_smooth = smooth_igc_output([all_speed, all_vz, all_alti])
        all_vz2 = np.zeros(all_alti_smooth.shape[0])

        print('igc smothed')
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
        # ffmpeg -framerate 24 -i C:\tmp_vid\vid1\%01d.png C:\tmp_vid\1.mp4
        img_gen = gen_img_from_smoothed_list(speed_vid_reshaped, vz_vid_reshaped, alti_vid_reshaped, time_vid_reshaped,
                                             time_vid_full_reshaped)
        m.write_video('overlay.mp4', img_gen, fps=24, qp=qualite_compression)
        print("\n[OK] Video generee: overlay.mp4")






