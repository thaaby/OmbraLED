

import cv2
import numpy as np
import mediapipe as mp
from collections import namedtuple, deque
import math
import json
import time
import sys
from datetime import datetime
import socket
import glob
try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[!] pyserial non installato — Arduino video disabilitato (pip install pyserial)")

# ============================================================
# DATABASE COLORI
# ============================================================
ColorDef = namedtuple('ColorDef', ['name', 'name_it', 'rgb', 'hex_code'])

COLOR_DATABASE = [
    # Rossi
    ColorDef('Red', 'Rosso', (255, 0, 0), '#FF0000'),
    ColorDef('Dark Red', 'Rosso Scuro', (139, 0, 0), '#8B0000'),
    ColorDef('Crimson', 'Cremisi', (220, 20, 60), '#DC143C'),
    ColorDef('Indian Red', 'Rosso Indiano', (205, 92, 92), '#CD5C5C'),
    ColorDef('Light Coral', 'Corallo Chiaro', (240, 128, 128), '#F08080'),
    ColorDef('Salmon', 'Salmone', (250, 128, 114), '#FA8072'),
    ColorDef('Dark Salmon', 'Salmone Scuro', (233, 150, 122), '#E9967A'),
    ColorDef('Light Salmon', 'Salmone Chiaro', (255, 160, 122), '#FFA07A'),
    ColorDef('Fire Brick', 'Mattone', (178, 34, 34), '#B22222'),
    ColorDef('Maroon', 'Marrone Rosso', (128, 0, 0), '#800000'),
    # Arancioni
    ColorDef('Orange', 'Arancione', (255, 165, 0), '#FFA500'),
    ColorDef('Dark Orange', 'Arancione Scuro', (255, 140, 0), '#FF8C00'),
    ColorDef('Orange Red', 'Rosso Arancio', (255, 69, 0), '#FF4500'),
    ColorDef('Tomato', 'Pomodoro', (255, 99, 71), '#FF6347'),
    ColorDef('Coral', 'Corallo', (255, 127, 80), '#FF7F50'),
    ColorDef('Peach', 'Pesca', (255, 218, 185), '#FFDAB9'),
    ColorDef('Apricot', 'Albicocca', (251, 206, 177), '#FBCEB1'),
    ColorDef('Tangerine', 'Mandarino', (255, 159, 0), '#FF9F00'),
    ColorDef('Burnt Orange', 'Arancione Bruciato', (204, 85, 0), '#CC5500'),
    ColorDef('Pumpkin', 'Zucca', (255, 117, 24), '#FF7518'),
    # Gialli
    ColorDef('Yellow', 'Giallo', (255, 255, 0), '#FFFF00'),
    ColorDef('Light Yellow', 'Giallo Chiaro', (255, 255, 224), '#FFFFE0'),
    ColorDef('Lemon', 'Limone', (255, 247, 0), '#FFF700'),
    ColorDef('Gold', 'Oro', (255, 215, 0), '#FFD700'),
    ColorDef('Golden Yellow', 'Giallo Dorato', (255, 223, 0), '#FFDF00'),
    ColorDef('Mustard', 'Senape', (255, 219, 88), '#FFDB58'),
    ColorDef('Canary Yellow', 'Giallo Canarino', (255, 239, 0), '#FFEF00'),
    ColorDef('Banana Yellow', 'Giallo Banana', (255, 225, 53), '#FFE135'),
    ColorDef('Amber', 'Ambra', (255, 191, 0), '#FFBF00'),
    ColorDef('Champagne', 'Champagne', (247, 231, 206), '#F7E7CE'),
    ColorDef('Cream', 'Crema', (255, 253, 208), '#FFFDD0'),
    ColorDef('Khaki', 'Cachi', (240, 230, 140), '#F0E68C'),
    ColorDef('Dark Khaki', 'Cachi Scuro', (189, 183, 107), '#BDB76B'),
    # Verdi
    ColorDef('Green', 'Verde', (0, 128, 0), '#008000'),
    ColorDef('Lime', 'Lime', (0, 255, 0), '#00FF00'),
    ColorDef('Bright Green', 'Verde Brillante', (102, 255, 0), '#66FF00'),
    ColorDef('Dark Green', 'Verde Scuro', (0, 100, 0), '#006400'),
    ColorDef('Forest Green', 'Verde Foresta', (34, 139, 34), '#228B22'),
    ColorDef('Sea Green', 'Verde Mare', (46, 139, 87), '#2E8B57'),
    ColorDef('Medium Sea Green', 'Verde Mare Medio', (60, 179, 113), '#3CB371'),
    ColorDef('Light Green', 'Verde Chiaro', (144, 238, 144), '#90EE90'),
    ColorDef('Pale Green', 'Verde Pallido', (152, 251, 152), '#98FB98'),
    ColorDef('Spring Green', 'Verde Primavera', (0, 255, 127), '#00FF7F'),
    ColorDef('Lawn Green', 'Verde Prato', (124, 252, 0), '#7CFC00'),
    ColorDef('Chartreuse', 'Chartreuse', (127, 255, 0), '#7FFF00'),
    ColorDef('Yellow Green', 'Giallo Verde', (154, 205, 50), '#9ACD32'),
    ColorDef('Olive', 'Oliva', (128, 128, 0), '#808000'),
    ColorDef('Olive Drab', 'Oliva Opaco', (107, 142, 35), '#6B8E23'),
    ColorDef('Dark Olive', 'Oliva Scuro', (85, 107, 47), '#556B2F'),
    ColorDef('Mint', 'Menta', (189, 252, 201), '#BDFCC9'),
    ColorDef('Emerald', 'Smeraldo', (80, 200, 120), '#50C878'),
    ColorDef('Jade', 'Giada', (0, 168, 107), '#00A86B'),
    ColorDef('Teal Green', 'Verde Petrolio', (0, 128, 128), '#008080'),
    # Ciano/Acqua
    ColorDef('Cyan', 'Ciano', (0, 255, 255), '#00FFFF'),
    ColorDef('Aqua', 'Acqua', (0, 200, 200), '#00C8C8'),
    ColorDef('Light Cyan', 'Ciano Chiaro', (224, 255, 255), '#E0FFFF'),
    ColorDef('Dark Cyan', 'Ciano Scuro', (0, 139, 139), '#008B8B'),
    ColorDef('Turquoise', 'Turchese', (64, 224, 208), '#40E0D0'),
    ColorDef('Dark Turquoise', 'Turchese Scuro', (0, 206, 209), '#00CED1'),
    ColorDef('Medium Turquoise', 'Turchese Medio', (72, 209, 204), '#48D1CC'),
    ColorDef('Pale Turquoise', 'Turchese Pallido', (175, 238, 238), '#AFEEEE'),
    ColorDef('Aquamarine', 'Acquamarina', (127, 255, 212), '#7FFFD4'),
    ColorDef('Teal', 'Petrolio', (0, 128, 128), '#008080'),
    ColorDef('Cadet Blue', 'Blu Cadetto', (95, 158, 160), '#5F9EA0'),
    # Blu
    ColorDef('Blue', 'Blu', (0, 0, 255), '#0000FF'),
    ColorDef('Light Blue', 'Blu Chiaro', (173, 216, 230), '#ADD8E6'),
    ColorDef('Sky Blue', 'Azzurro Cielo', (135, 206, 235), '#87CEEB'),
    ColorDef('Light Sky Blue', 'Azzurro Cielo Chiaro', (135, 206, 250), '#87CEFA'),
    ColorDef('Deep Sky Blue', 'Azzurro Intenso', (0, 191, 255), '#00BFFF'),
    ColorDef('Dodger Blue', 'Blu Dodger', (30, 144, 255), '#1E90FF'),
    ColorDef('Cornflower Blue', 'Blu Fiordaliso', (100, 149, 237), '#6495ED'),
    ColorDef('Steel Blue', 'Blu Acciaio', (70, 130, 180), '#4682B4'),
    ColorDef('Royal Blue', 'Blu Reale', (65, 105, 225), '#4169E1'),
    ColorDef('Medium Blue', 'Blu Medio', (0, 0, 205), '#0000CD'),
    ColorDef('Dark Blue', 'Blu Scuro', (0, 0, 139), '#00008B'),
    ColorDef('Navy', 'Blu Navy', (0, 0, 128), '#000080'),
    ColorDef('Midnight Blue', 'Blu Mezzanotte', (25, 25, 112), '#191970'),
    ColorDef('Cobalt Blue', 'Blu Cobalto', (0, 71, 171), '#0047AB'),
    ColorDef('Electric Blue', 'Blu Elettrico', (125, 249, 255), '#7DF9FF'),
    ColorDef('Azure', 'Azzurro', (0, 127, 255), '#007FFF'),
    ColorDef('Powder Blue', 'Blu Polvere', (176, 224, 230), '#B0E0E6'),
    ColorDef('Alice Blue', 'Blu Alice', (240, 248, 255), '#F0F8FF'),
    # Viola/Porpora
    ColorDef('Purple', 'Viola', (128, 0, 128), '#800080'),
    ColorDef('Violet', 'Violetto', (238, 130, 238), '#EE82EE'),
    ColorDef('Dark Violet', 'Violetto Scuro', (148, 0, 211), '#9400D3'),
    ColorDef('Blue Violet', 'Blu Violetto', (138, 43, 226), '#8A2BE2'),
    ColorDef('Dark Orchid', 'Orchidea Scura', (153, 50, 204), '#9932CC'),
    ColorDef('Medium Orchid', 'Orchidea Media', (186, 85, 211), '#BA55D3'),
    ColorDef('Orchid', 'Orchidea', (218, 112, 214), '#DA70D6'),
    ColorDef('Plum', 'Prugna', (221, 160, 221), '#DDA0DD'),
    ColorDef('Medium Purple', 'Porpora Medio', (147, 112, 219), '#9370DB'),
    ColorDef('Indigo', 'Indaco', (75, 0, 130), '#4B0082'),
    ColorDef('Slate Blue', 'Blu Ardesia', (106, 90, 205), '#6A5ACD'),
    ColorDef('Dark Slate Blue', 'Blu Ardesia Scuro', (72, 61, 139), '#483D8B'),
    ColorDef('Lavender', 'Lavanda', (230, 230, 250), '#E6E6FA'),
    ColorDef('Thistle', 'Cardo', (216, 191, 216), '#D8BFD8'),
    ColorDef('Mauve', 'Malva', (224, 176, 255), '#E0B0FF'),
    ColorDef('Amethyst', 'Ametista', (153, 102, 204), '#9966CC'),
    ColorDef('Grape', 'Uva', (111, 45, 168), '#6F2DA8'),
    ColorDef('Eggplant', 'Melanzana', (97, 64, 81), '#614051'),
    # Rosa/Magenta
    ColorDef('Pink', 'Rosa', (255, 192, 203), '#FFC0CB'),
    ColorDef('Light Pink', 'Rosa Chiaro', (255, 182, 193), '#FFB6C1'),
    ColorDef('Hot Pink', 'Rosa Acceso', (255, 105, 180), '#FF69B4'),
    ColorDef('Deep Pink', 'Rosa Intenso', (255, 20, 147), '#FF1493'),
    ColorDef('Medium Violet Red', 'Rosso Violetto Medio', (199, 21, 133), '#C71585'),
    ColorDef('Pale Violet Red', 'Rosso Violetto Pallido', (219, 112, 147), '#DB7093'),
    ColorDef('Magenta', 'Magenta', (255, 0, 255), '#FF00FF'),
    ColorDef('Fuchsia', 'Fucsia', (255, 0, 200), '#FF00C8'),
    ColorDef('Rose', 'Rosa', (255, 0, 127), '#FF007F'),
    ColorDef('Blush', 'Rosa Cipria', (222, 93, 131), '#DE5D83'),
    ColorDef('Carnation Pink', 'Rosa Garofano', (255, 166, 201), '#FFA6C9'),
    ColorDef('Flamingo', 'Fenicottero', (252, 142, 172), '#FC8EAC'),
    ColorDef('Raspberry', 'Lampone', (227, 11, 92), '#E30B5C'),
    ColorDef('Cerise', 'Ciliegia', (222, 49, 99), '#DE3163'),
    # Marroni
    ColorDef('Brown', 'Marrone', (165, 42, 42), '#A52A2A'),
    ColorDef('Dark Brown', 'Marrone Scuro', (101, 67, 33), '#654321'),
    ColorDef('Saddle Brown', 'Marrone Sella', (139, 69, 19), '#8B4513'),
    ColorDef('Sienna', 'Terra di Siena', (160, 82, 45), '#A0522D'),
    ColorDef('Chocolate', 'Cioccolato', (210, 105, 30), '#D2691E'),
    ColorDef('Peru', 'Peru', (205, 133, 63), '#CD853F'),
    ColorDef('Sandy Brown', 'Marrone Sabbia', (244, 164, 96), '#F4A460'),
    ColorDef('Burly Wood', 'Legno', (222, 184, 135), '#DEB887'),
    ColorDef('Tan', 'Cuoio', (210, 180, 140), '#D2B48C'),
    ColorDef('Rosy Brown', 'Marrone Rosato', (188, 143, 143), '#BC8F8F'),
    ColorDef('Moccasin', 'Mocassino', (255, 228, 181), '#FFE4B5'),
    ColorDef('Navajo White', 'Bianco Navajo', (255, 222, 173), '#FFDEAD'),
    ColorDef('Wheat', 'Grano', (245, 222, 179), '#F5DEB3'),
    ColorDef('Bisque', 'Biscotto', (255, 228, 196), '#FFE4C4'),
    ColorDef('Blanched Almond', 'Mandorla', (255, 235, 205), '#FFEBCD'),
    ColorDef('Cornsilk', 'Seta di Mais', (255, 248, 220), '#FFF8DC'),
    ColorDef('Beige', 'Beige', (245, 245, 220), '#F5F5DC'),
    ColorDef('Antique White', 'Bianco Antico', (250, 235, 215), '#FAEBD7'),
    ColorDef('Papaya Whip', 'Papaya', (255, 239, 213), '#FFEFD5'),
    ColorDef('Linen', 'Lino', (250, 240, 230), '#FAF0E6'),
    ColorDef('Old Lace', 'Pizzo Antico', (253, 245, 230), '#FDF5E6'),
    ColorDef('Coffee', 'Caffè', (111, 78, 55), '#6F4E37'),
    ColorDef('Caramel', 'Caramello', (255, 213, 154), '#FFD59A'),
    ColorDef('Rust', 'Ruggine', (183, 65, 14), '#B7410E'),
    ColorDef('Copper', 'Rame', (184, 115, 51), '#B87333'),
    ColorDef('Bronze', 'Bronzo', (205, 127, 50), '#CD7F32'),
    ColorDef('Terracotta', 'Terracotta', (226, 114, 91), '#E2725B'),
    ColorDef('Burgundy', 'Borgogna', (128, 0, 32), '#800020'),
    ColorDef('Wine', 'Vino', (114, 47, 55), '#722F37'),
    ColorDef('Vermillion', 'Vermiglio', (227, 66, 52), '#E34234'),
    ColorDef('Ochre', 'Ocra', (204, 119, 34), '#CC7722'),
    ColorDef('Marigold', 'Calendula', (234, 162, 33), '#EAA221'),
    ColorDef('Dark Teal', 'Petrolio Scuro', (0, 109, 111), '#006D6F'),
    ColorDef('Persian Red', 'Rosso Persiano', (204, 51, 51), '#CC3333'),
    ColorDef('Sapphire', 'Zaffiro', (15, 82, 186), '#0F52BA'),
    # Grigi
    ColorDef('Gray', 'Grigio', (128, 128, 128), '#808080'),
    ColorDef('Dark Gray', 'Grigio Scuro', (169, 169, 169), '#A9A9A9'),
    ColorDef('Dim Gray', 'Grigio Tenue', (105, 105, 105), '#696969'),
    ColorDef('Light Gray', 'Grigio Chiaro', (211, 211, 211), '#D3D3D3'),
    ColorDef('Silver', 'Argento', (192, 192, 192), '#C0C0C0'),
    ColorDef('Light Slate Gray', 'Grigio Ardesia Chiaro', (119, 136, 153), '#778899'),
    ColorDef('Slate Gray', 'Grigio Ardesia', (112, 128, 144), '#708090'),
    ColorDef('Dark Slate Gray', 'Grigio Ardesia Scuro', (47, 79, 79), '#2F4F4F'),
    ColorDef('Charcoal', 'Carbone', (54, 69, 79), '#36454F'),
    ColorDef('Ash Gray', 'Grigio Cenere', (178, 190, 181), '#B2BEB5'),
    ColorDef('Gainsboro', 'Gainsboro', (220, 220, 220), '#DCDCDC'),
    ColorDef('White Smoke', 'Fumo Bianco', (245, 245, 245), '#F5F5F5'),
    # Bianco e Nero
    ColorDef('White', 'Bianco', (255, 255, 255), '#FFFFFF'),
    ColorDef('Snow', 'Neve', (255, 250, 250), '#FFFAFA'),
    ColorDef('Honeydew', 'Melata', (240, 255, 240), '#F0FFF0'),
    ColorDef('Mint Cream', 'Crema Menta', (245, 255, 250), '#F5FFFA'),
    ColorDef('Ghost White', 'Bianco Fantasma', (248, 248, 255), '#F8F8FF'),
    ColorDef('Floral White', 'Bianco Floreale', (255, 250, 240), '#FFFAF0'),
    ColorDef('Seashell', 'Conchiglia', (255, 245, 238), '#FFF5EE'),
    ColorDef('Ivory', 'Avorio', (255, 255, 240), '#FFFFF0'),
    ColorDef('Black', 'Nero', (0, 0, 0), '#000000'),
    ColorDef('Jet Black', 'Nero Corvino', (52, 52, 52), '#343434'),
    ColorDef('Onyx', 'Onice', (53, 56, 57), '#353839'),
    ColorDef('Ebony', 'Ebano', (85, 93, 80), '#555D50'),
]

# ============================================================
# CONFIGURAZIONE MAXISCHERMO ESP (MULTI-PANNELLO UDP)
# ============================================================
# IP dei 3 ESP (da sinistra a destra)
ESP_IPS = ["192.168.1.61", "192.168.1.62", "192.168.1.63", "192.168.1.64", "192.168.1.65",  "192.168.1.68"]
ESP_PORT = 4210

# Dimensioni di ogni singolo pannello LED ESP
PANEL_WIDTH = 15
PANEL_HEIGHT = 44
TOTAL_WIDTH = PANEL_WIDTH * len(ESP_IPS)  # Larghezza totale del ledwall

# Orientamento e serpentina ESP
ESP_SERPENTINE_HORIZONTAL = True
ESP_START_BOTTOM = False  # False = in alto a sx. True = in basso a sx.

# ============================================================
# CONFIGURAZIONE ARDUINO VIDEO (SERIALE)
# ============================================================
ARDUINO_ENABLED = True            # Abilita streaming video via seriale
ARDUINO_PORT = "auto"             # Porta seriale ('auto' per rilevamento automatico)
ARDUINO_BAUD = 500000             # Deve corrispondere al baud rate dello sketch
ARDUINO_ROWS = 32                 # Altezza totale
ARDUINO_COLS = 56                 # Larghezza totale (7 pannelli da 8 = 56)
# Configurazione Orientamento Pannelli Arduino
ARDUINO_PANEL_W = 8               # Larghezza del singolo pannello fisico
ARDUINO_PANEL_H = 32              # Altezza del singolo pannello fisico
ARDUINO_PANELS_COUNT = 7          # 7 pannelli 8x32 = 56x32

# Configurazione del cablaggio (0 = estrema destra, 6 = estrema sinistra):
ARDUINO_PANEL_ORDER = [6, 5, 4, 3, 2, 1, 0]
ARDUINO_PANEL_START_BOTTOM = [False, False, False, False, False, False, False]
ARDUINO_SERPENTINE_X = True         # Zigzag orizzontale dentro il pannello

GAMMA = 2.5           
COMMON_ANODE = False  # Premi [I] per invertire i colori

# Tabella gamma pre-calcolata
gamma_table = np.array([((i / 255.0) ** GAMMA) * 255 for i in np.arange(0, 256)]).astype("uint8")

def apply_gamma(color):
    """Applica la correzione gamma al colore RGB"""
    return gamma_table[color]

def create_udp_socket():
    """Crea un socket UDP per comunicare con i pannelli ESP."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[OK] Socket UDP creato -> {len(ESP_IPS)} pannelli sulla porta {ESP_PORT}")
        for i, ip in enumerate(ESP_IPS):
            print(f"  Pannello {i}: {ip}")
        return sock
    except Exception as e:
        print(f"[X] Errore creazione socket UDP: {e}")
        return None


def create_arduino_serial():
    """Crea connessione seriale all'Arduino per video streaming.
    Se ARDUINO_PORT è 'auto', cerca automaticamente la porta."""
    if not ARDUINO_ENABLED or not HAS_SERIAL:
        if not HAS_SERIAL:
            print("[!] pyserial non disponibile. Installa con: pip install pyserial")
        return None
    
    port = ARDUINO_PORT
    
    # Auto-detect: cerca porte seriali disponibili
    if port == "auto":
        porte_trovate = (
            glob.glob('/dev/ttyUSB*') + 
            glob.glob('/dev/ttyACM*') + 
            glob.glob('/dev/cu.usbmodem*') +   # macOS
            glob.glob('/dev/cu.usbserial*') +  # macOS
            glob.glob('/dev/tty.*')            # macOS generico per R4
        )
        if not porte_trovate:
            print("[!] Nessuna porta seriale trovata!")
            print("    Controlla che l'Arduino sia collegato via USB.")
            return None
        port = porte_trovate[0]
        print(f"[AUTO] Porta seriale rilevata: {port}")
    
    try:
        ser = serial.Serial(port, ARDUINO_BAUD, timeout=0.01) # Timeout basso per non bloccare il loop
        time.sleep(2)  # Attendi reset Arduino
        print(f"[OK] Arduino connesso su {port} @ {ARDUINO_BAUD} baud")
        print(f"     Matrice: {ARDUINO_COLS}x{ARDUINO_ROWS} ({ARDUINO_COLS * ARDUINO_ROWS} LED)")
        
        # Test: pulizia buffer in ingresso
        ser.read_all()
        return ser
    except serial.SerialException as e:
        print(f"[X] Errore apertura porta {port}: {e}")
        return None
    except Exception as e:
        print(f"[!] Arduino non connesso: {e}")
        return None


def map_frame_to_leds(frame_rgb):
    """Mappa l'immagine 56x32 nell'ordine fisico dei 7 pannelli LED (8x32 ciascuno).

    7 pannelli da 8x32 collegati in serie = 56x32 totali.
    ARDUINO_PANEL_ORDER definisce quale posizione X occupa ogni pannello fisico.
    All'interno di ogni pannello, il segnale è a zigzag orizzontale (serpentina).
    """
    out_buffer = bytearray(ARDUINO_COLS * ARDUINO_ROWS * 3)
    idx = 0
    
    # Ciclo sui pannelli fisici nell'ordine in cui sono collegati (0, 1, 2, 3)
    for p in range(ARDUINO_PANELS_COUNT):
        
        # Qual è l'indice X di partenza nell'immagine totale per questo pannello?
        # Il pannello 0 è a destra (indice 3 da sinistra), quindi X inizia a 3*8 = 24.
        panel_pos_x = ARDUINO_PANEL_ORDER[p]
        start_x = panel_pos_x * ARDUINO_PANEL_W
        
        # Direzione verticale: parte dal basso o dall'alto?
        starts_bottom = ARDUINO_PANEL_START_BOTTOM[p]
        
        # Iteriamo su ciascun LED (32 righe x 8 colonne) = 256 pixel
        # "y_local" è la riga FISICA del pannello (0 = l'ingresso, 31 = l'uscita)
        for y_local in range(ARDUINO_PANEL_H):
            
            # Calcola la Y globale dell'immagine
            if starts_bottom:
                # Se parte dal basso, entry point è Y=31. Man mano che y_local cresce, saliamo verso Y=0
                global_y = (ARDUINO_PANEL_H - 1) - y_local
            else:
                # Se parte dall'alto, entry point è Y=0. Man mano che y_local cresce, scendiamo verso Y=31
                global_y = y_local
                
            # Calcola la X
            for x_local in range(ARDUINO_PANEL_W):
                
                eff_x = x_local
                # Serpentine X locale:
                # Di solito se si parte dal basso e si fa zigzag, la prima riga (y_local=0) va in una direzione,
                # la seconda (y_local=1) torna indietro.
                if ARDUINO_SERPENTINE_X and (y_local % 2 == 1):
                    eff_x = (ARDUINO_PANEL_W - 1) - x_local
                
                # Attenzione: se il segnale entra in basso a *destra* o in basso a *sinistra* cambia.
                # Assumiamo che il Data In locale sia sempre a sinistra (0) se non zigzagato.
                # Se l'immagine risulta specchiata all'interno del pannello, basta ribaltare:
                # eff_x = (ARDUINO_PANEL_W - 1) - eff_x
                
                global_x = start_x + eff_x
                
                pixel = frame_rgb[global_y, global_x]
                
                out_buffer[idx]   = pixel[0]
                out_buffer[idx+1] = pixel[1]
                out_buffer[idx+2] = pixel[2]
                idx += 3
                
    return bytes(out_buffer)


def send_arduino_frame(ser, frame, use_gamma=True):
    """Invia un frame webcam all'Arduino come pixel RGB grezzi.
    
    1. Ridimensiona il frame a ARDUINO_COLS × ARDUINO_ROWS
    2. Converte BGR -> RGB
    3. Applica gamma (opzionale)
    4. Rimappa serpentina
    5. Invia: byte MAGIC HEADER (0xFF 0x4C 0x45) + 3072 byte RGB
    """
    # Ridimensiona alla risoluzione della matrice LED
    small = cv2.resize(frame, (ARDUINO_COLS, ARDUINO_ROWS), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = cv2.flip(rgb, 0)  # Corregge orientamento verticale (pannelli montati a testa in giù)
    
    # Correggi gamma
    if use_gamma:
        rgb = gamma_table[rgb]
    
    # Inverti se Common Anode
    if COMMON_ANODE:
        rgb = 255 - rgb
    
    # Applica mappatura fisica complessa
    rgb_bytes = map_frame_to_leds(rgb)
    
    # Invia: header MAGIC_HEADER + dati RGB mappati
    ser.write(b'\xFF\x4C\x45' + rgb_bytes)


def niente(x):
    """Callback vuota per lo slider di OpenCV."""
    pass


# ============================================================
# FUNZIONI COLORE (LAB + Delta-E CIE2000)
# ============================================================

def rgb_to_lab(rgb):
    """Converte RGB in spazio colore CIE LAB."""
    r, g, b = [x / 255.0 for x in rgb]
    
    def linearize(c):
        return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
    
    r, g, b = linearize(r), linearize(g), linearize(b)
    
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    x, y, z = x / 0.95047, y / 1.0, z / 1.08883
    
    def f(t):
        return t ** (1/3) if t > 0.008856 else (7.787 * t + 16/116)
    
    L = 116 * f(y) - 16
    a = 500 * (f(x) - f(y))
    b_val = 200 * (f(y) - f(z))
    
    return (L, a, b_val)


def delta_e_cie2000(lab1, lab2):
    """Distanza Delta-E CIE2000."""
    L1, a1, b1 = lab1
    L2, a2, b2 = lab2
    
    kL, kC, kH = 1.0, 1.0, 1.0
    
    C1 = np.sqrt(a1**2 + b1**2)
    C2 = np.sqrt(a2**2 + b2**2)
    C_avg = (C1 + C2) / 2
    
    G = 0.5 * (1 - np.sqrt(C_avg**7 / (C_avg**7 + 25**7)))
    
    a1_prime = a1 * (1 + G)
    a2_prime = a2 * (1 + G)
    
    C1_prime = np.sqrt(a1_prime**2 + b1**2)
    C2_prime = np.sqrt(a2_prime**2 + b2**2)
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360
    
    delta_L_prime = L2 - L1
    delta_C_prime = C2_prime - C1_prime
    
    delta_h_prime = h2_prime - h1_prime
    if abs(delta_h_prime) > 180:
        delta_h_prime -= 360 * np.sign(delta_h_prime)
    
    delta_H_prime = 2 * np.sqrt(C1_prime * C2_prime) * np.sin(np.radians(delta_h_prime / 2))
    
    L_avg_prime = (L1 + L2) / 2
    C_avg_prime = (C1_prime + C2_prime) / 2
    
    h_avg_prime = (h1_prime + h2_prime) / 2
    if abs(h1_prime - h2_prime) > 180:
        h_avg_prime += 180
    
    T = (1 - 0.17 * np.cos(np.radians(h_avg_prime - 30)) +
         0.24 * np.cos(np.radians(2 * h_avg_prime)) +
         0.32 * np.cos(np.radians(3 * h_avg_prime + 6)) -
         0.20 * np.cos(np.radians(4 * h_avg_prime - 63)))
    
    SL = 1 + (0.015 * (L_avg_prime - 50)**2) / np.sqrt(20 + (L_avg_prime - 50)**2)
    SC = 1 + 0.045 * C_avg_prime
    SH = 1 + 0.015 * C_avg_prime * T
    
    delta_theta = 30 * np.exp(-((h_avg_prime - 275) / 25)**2)
    RC = 2 * np.sqrt(C_avg_prime**7 / (C_avg_prime**7 + 25**7))
    RT = -RC * np.sin(np.radians(2 * delta_theta))
    
    delta_E = np.sqrt(
        (delta_L_prime / (kL * SL))**2 +
        (delta_C_prime / (kC * SC))**2 +
        (delta_H_prime / (kH * SH))**2 +
        RT * (delta_C_prime / (kC * SC)) * (delta_H_prime / (kH * SH))
    )
    
    return delta_E


# Pre-calcola i valori LAB
COLOR_LAB_CACHE = {color.name: rgb_to_lab(color.rgb) for color in COLOR_DATABASE}


def find_closest_color(rgb):
    """Trova il colore più vicino con Delta-E CIE2000."""
    target_lab = rgb_to_lab(rgb)
    
    min_distance = float('inf')
    closest_color = None
    
    for color in COLOR_DATABASE:
        color_lab = COLOR_LAB_CACHE[color.name]
        distance = delta_e_cie2000(target_lab, color_lab)
        
        if distance < min_distance:
            min_distance = distance
            closest_color = color
    
    return (closest_color.name, closest_color.name_it, closest_color.hex_code, min_distance)


def rgb_to_hex(r, g, b):
    return f'#{r:02X}{g:02X}{b:02X}'


# ============================================================
# PIPELINE ACCURATEZZA (CLAHE + K-Means)
# ============================================================

CLAHE_CLIP_LIMIT = 2.0
KMEANS_CLUSTERS = 3


def _apply_clahe(roi):
    """CLAHE: Equalizzazione luminosità locale."""
    if roi.size == 0:
        return roi
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=(4, 4))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _extract_dominant_kmeans(roi, n_clusters=3):
    """Colore dominante via K-Means."""
    pixels = roi.reshape(-1, 3).astype(np.float32)
    
    if len(pixels) < n_clusters:
        return np.mean(pixels, axis=0).astype(int)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    
    dominant_idx = np.argmax(np.bincount(labels.flatten()))
    return centers[dominant_idx].astype(int)


# ============================================================
# GRIGLIA + PALETTE
# ============================================================

GRID_SIZES = [3, 5, 7]


def detect_grid_colors(frame, grid_size=5, sample_size=12):
    """Campiona colori su una griglia NxN."""
    height, width = frame.shape[:2]
    colors = []
    
    margin_x = int(width * 0.28)
    margin_y = int(height * 0.28)
    
    for row in range(grid_size):
        for col in range(grid_size):
            px = margin_x + int((width - 2 * margin_x) * col / max(1, grid_size - 1))
            py = margin_y + int((height - 2 * margin_y) * row / max(1, grid_size - 1))
            
            half = sample_size // 2
            x1 = max(0, px - half)
            y1 = max(0, py - half)
            x2 = min(width, px + half)
            y2 = min(height, py + half)
            
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            
            roi = _apply_clahe(roi)
            
            if roi.size > 9:
                avg_bgr = _extract_dominant_kmeans(roi, min(KMEANS_CLUSTERS, roi.shape[0] * roi.shape[1]))
            else:
                avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)
            
            b, g, r = avg_bgr
            rgb = (int(r), int(g), int(b))
            
            name_en, name_it, hex_code, distance = find_closest_color(rgb)
            
            colors.append({
                'rgb': rgb,
                'bgr': (int(b), int(g), int(r)),
                'hex': rgb_to_hex(*rgb),
                'name_en': name_en,
                'name_it': name_it,
                'pos': (px, py),
                'distance': distance
            })
    
    return colors


def export_palette(palette, grid_size):
    """Esporta la palette come JSON + PNG."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSON
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "grid": f"{grid_size}x{grid_size}",
        "n_colors": len(palette),
        "colors": []
    }
    for color in palette:
        json_data["colors"].append({
            "name_en": color['name_en'],
            "name_it": color['name_it'],
            "hex": color['hex'],
            "rgb": list(color['rgb']),
        })
    
    json_filename = f"palette_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)
    
    # PNG
    swatch_w = 100
    swatch_h = 100
    text_h = 35
    n = len(palette)
    if n == 0:
        return json_filename
    
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    
    img_w = cols * swatch_w
    img_h = rows * (swatch_h + text_h)
    img = np.zeros((img_h, img_w, 3), dtype=np.uint8)
    img[:] = (30, 30, 30)
    
    for i, color in enumerate(palette):
        col = i % cols
        row = i // cols
        x = col * swatch_w
        y = row * (swatch_h + text_h)
        
        cv2.rectangle(img, (x + 2, y + 2), (x + swatch_w - 2, y + swatch_h - 2), color['bgr'], -1)
        cv2.putText(img, color['hex'], (x + 5, y + swatch_h + 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, color['name_it'][:10], (x + 5, y + swatch_h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (150, 150, 150), 1)
    
    png_filename = f"palette_{timestamp}.png"
    cv2.imwrite(png_filename, img)
    
    return json_filename


# ============================================================
# DISEGNO GRIGLIA MINIMALE (canvas sintetico, no webcam feed)
# ============================================================

def draw_minimal_grid(grid_colors, grid_size, win_w=600, win_h=600):
    """
    Genera un canvas che riempie esattamente la finestra.
    I quadrati si adattano alle dimensioni della finestra.
    """
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    
    # Calcola dimensione quadrati per riempire tutto
    swatch_w = win_w // grid_size
    swatch_h = win_h // grid_size
    
    for i, color in enumerate(grid_colors):
        row = i // grid_size
        col = i % grid_size
        
        x = col * swatch_w
        y = row * swatch_h
        
        cv2.rectangle(canvas, (x, y), (x + swatch_w, y + swatch_h), color['bgr'], -1)
    
    return canvas


# ============================================================
# MAIN
# ============================================================

def list_cameras():
    """Elenca le webcam disponibili."""
    cameras = []
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cameras.append(i)
            cap.release()
    return cameras


def select_camera():
    """Seleziona webcam."""
    print("\n[SCAN] Ricerca webcam...")
    cameras = list_cameras()
    
    if not cameras:
        print("[!] Nessuna webcam trovata, provo ID 0...")
        return 0
    
    print(f"[CAM] Trovate: {len(cameras)}")
    for cam_id in cameras:
        print(f"  [{cam_id}] Camera {cam_id}")
    
    if len(cameras) == 1:
        print(f"[OK] Camera {cameras[0]} selezionata")
        return cameras[0]
    
    while True:
        try:
            choice = input(f"> Seleziona camera (0-{cameras[-1]}): ")
            cam_id = int(choice)
            if cam_id in cameras:
                return cam_id
            print("[X] Camera non valida!")
        except ValueError:
            print("[X] Inserisci un numero!")


def detect_center_color(frame, center_size=50):
    """Rileva il colore dominante al centro del frame."""
    height, width = frame.shape[:2]
    cx, cy = width // 2, height // 2
    half = center_size // 2
    half = min(half, cx, cy, width - cx, height - cy)
    if half < 1:
        half = 1
    
    roi = frame[cy - half:cy + half, cx - half:cx + half]
    roi = _apply_clahe(roi)
    
    if roi.size > 9:
        avg_bgr = _extract_dominant_kmeans(roi, min(KMEANS_CLUSTERS, roi.shape[0] * roi.shape[1]))
    else:
        avg_bgr = np.mean(roi, axis=(0, 1)).astype(int)
    
    b, g, r = avg_bgr
    rgb = (int(r), int(g), int(b))
    name_en, name_it, hex_code, distance = find_closest_color(rgb)
    
    return {
        'rgb': rgb,
        'bgr': (int(b), int(g), int(r)),
        'hex': rgb_to_hex(*rgb),
        'name_en': name_en,
        'name_it': name_it,
        'distance': distance
    }


# ============================================================
# TRACKING SILHOUETTE (Multi-color via YOLO)
# ============================================================
class SimpleTracker:
    def __init__(self):
        pass  # Nessuno stato — tutto è live, frame per frame

    def get_dominant_color(self, frame_bgr, mask_2d):
        """Estrae il colore dominante REALE dai pixel della persona — 100% live.
        
        1. K-Means (k=3) separa automaticamente pelle, vestiti e capelli/ombre
        2. Il cluster più grande = vestiti = colore dominante
        3. Boost saturazione +50% e luminosità +20% (resa LED)
        """
        pixels = frame_bgr[mask_2d]
        if len(pixels) == 0:
            return (128, 128, 128)
        
        pixels_f = pixels.astype(np.float32)
        
        # Sottocampiona per velocità (max 1500 pixel — K-Means su 90k sarebbe troppo lento)
        if len(pixels_f) > 1500:
            indices = np.random.choice(len(pixels_f), 1500, replace=False)
            pixels_f = pixels_f[indices]
        
        # K-Means: separa i colori in 3 gruppi (pelle, vestiti, capelli/ombre)
        n_clusters = min(3, len(pixels_f))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels_f, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )
        
        # Il cluster con più pixel = colore dominante (solitamente i vestiti)
        dominant_idx = np.argmax(np.bincount(labels.flatten()))
        dominant_bgr = centers[dominant_idx]
        
        # Boost saturazione e luminosità per resa su LED
        hsv_pixel = cv2.cvtColor(
            np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV
        )[0][0].astype(np.float64)
        hsv_pixel[1] = min(255, hsv_pixel[1] * 1.5)   # Saturazione +50%
        hsv_pixel[2] = min(255, hsv_pixel[2] * 1.2)   # Luminosità  +20%
        hsv_pixel[2] = max(60, hsv_pixel[2])           # Floor: anche il nero resta visibile
        boosted_bgr = cv2.cvtColor(
            np.uint8([[hsv_pixel]]), cv2.COLOR_HSV2BGR
        )[0][0].astype(np.float64)
        
        return (int(boosted_bgr[0]), int(boosted_bgr[1]), int(boosted_bgr[2]))


def main():
    print("\n" + "=" * 50)
    print("  REGIA LEDWALL - Multi-Pannello + Arduino Video")
    print(f"  ESP: {len(ESP_IPS)} pannelli ({TOTAL_WIDTH}x{PANEL_HEIGHT})")
    if ARDUINO_ENABLED:
        print(f"  Arduino: {ARDUINO_COLS}x{ARDUINO_ROWS} video seriale")
    print("=" * 50)
    
    # --- MODIFICA 1: Bypassata la selezione manuale per avvii automatici in background ---
    camera_id = 0 
    
    print(f"\n[CAM] Avvio webcam {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print("[X] Impossibile aprire la webcam!")
        return
    
    # Risoluzione bassa per risparmiare risorse
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n" + "-" * 50)
    print("  CONTROLLI:")
    print("  [F] - Fullscreen (toggle)")
    print("  [I] - Inverti colori per LED (Common Anode)")
    print("  [S] - Cambia stile Silhouette (Tinta Unita / Fotocamera)")
    print("  [Q/ESC] - Esci")
    print("-" * 50 + "\n")
    
    # --- ULTRALYTICS YOLOv8 INSTANCE SEGMENTATION ---
    import os
    import sys
    import time
    
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[!] IL MOTORE PROFESSIONALE YOLOv8 NON È INSTALLATO.")
        print("[!] Esegui questo comando nel terminale per installarlo:")
        print("    pip install ultralytics")
        print("\nTerminazione in corso...")
        sys.exit(1)
        
    print("[AI] Caricamento del modello professionale YOLOv8 Segmentation...")
    # Scaricherà in automatico yolov8n-seg.pt (pochi MB) al primissimo avvio
    model = YOLO('yolov8n-seg.pt')
    
    silhouette_tracker = SimpleTracker()
    solid_silhouette = True
    
    # --- CONNESSIONI ---
    global COMMON_ANODE
    udp_sock = create_udp_socket()
    arduino_ser = create_arduino_serial()
    
    fullscreen = False
    
    # Finestra ridimensionabile (compatibile GTK/Cocoa/Qt)
    cv2.namedWindow('Regia Ledwall', cv2.WINDOW_NORMAL)
    
    # Slider per controllare i pacchetti al secondo (FPS rete)
    cv2.createTrackbar('FPS Rete', 'Regia Ledwall', 30, 60, niente)
    
    arduino_status = "+ Arduino" if arduino_ser else "(no Arduino)"
    
    # Variabili per Handshake Arduino
    arduino_ready = True
    arduino_last_send_time = time.time()
    print(f"\n[STREAM] Inviando streaming video a {len(ESP_IPS)} pannelli ESP {arduino_status}...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[X] Errore lettura frame!")
                break
                
            # Specchia orizzontalmente la webcam come richiesto
            frame = cv2.flip(frame, 1)
            
            # Salva il frame originale PRIMA della segmentazione (serve per estrarre i colori reali)
            original_frame = frame.copy()
            
            # --- YOLOv8 ULTRALYTICS INSTANCE SEGMENTATION & TRACKING ---
            # persist=True abilita il ByteTrack integrato che non perde MAI i soggetti
            # classes=[0] processa ESCLUSIVAMENTE gli esseri umani, senza alcun limite di numero
            results = model.track(frame, persist=True, classes=[0], verbose=False)
            
            bg_image = np.zeros(frame.shape, dtype=np.uint8)
            final_frame_float = np.zeros(frame.shape, dtype=np.float32)
            r = results[0]
            
            if r.masks is not None and r.boxes.id is not None:
                # Estraiamo gli ID fisici assegnati in modo infallibile dal tracker
                ids = r.boxes.id.cpu().numpy().astype(int)
                # Estraiamo le maschere 2D indipendenti di ciascun corpo
                masks_data = r.masks.data.cpu().numpy()
                
                aggregate_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                
                for i, person_id in enumerate(ids):
                    # Ridimensioniamo la maschera alla risoluzione originale della webcam
                    mask_raw = masks_data[i]
                    mask_resized = cv2.resize(mask_raw, (frame.shape[1], frame.shape[0]))
                    # YOLO genera maschere pulite, la soglia 0.5 è ottimale
                    mask_2d = mask_resized > 0.5 
                    
                    if solid_silhouette:
                        # Estrae il colore dominante REALE dai vestiti/corpo della persona
                        current_color = silhouette_tracker.get_dominant_color(original_frame, mask_2d)
                        
                        # TRUCCO CHIRURGICO: DILATAZIONE DELL'AURA
                        # Essendo l'IA chirurgica, i corpi si oscurano ma non condividono mai gli stessi pixel.
                        # Espandendo matematicamente la maschera di ~35 pixel verso l'esterno, creiamo
                        # un'area di cuscinetto. Quando 2 persone si sfiorano, queste espansioni si incrociano
                        # coprendo letteralmente gli stessi pixel. (35px diventeranno 1-2 LED fisici sovrapposti)
                        kernel = np.ones((35, 35), np.uint8)
                        mask_expanded = cv2.dilate(mask_2d.astype(np.uint8), kernel, iterations=1)
                        mask_overlap = mask_expanded > 0
                        
                        # SOVRAPPOSIZIONE ADDITIVA (Luce pura!)
                        color_array = np.array(current_color, dtype=np.float32)
                        final_frame_float[mask_overlap] += color_array
                    else:
                        aggregate_mask[mask_2d] = 255
                        
                if solid_silhouette:
                    # NORMALIZZAZIONE: solo per gestire le sovrapposizioni (overlap additivo).
                    # Scala GIÙ i pixel che superano 255 (due aure sovrapposte),
                    # ma NON scala MAI SU i pixel scuri (vestiti neri restano neri).
                    max_vals = np.max(final_frame_float, axis=-1, keepdims=True)
                    max_vals[max_vals == 0] = 1.0
                    
                    # Il fattore di scala è min(1.0, 255/max): se max < 255, scala = 1.0 (non tocca).
                    # Se max > 255 (overlap), scala = 255/max (riduce proporzionalmente).
                    scale = np.minimum(1.0, 255.0 / max_vals)
                    final_frame_float = final_frame_float * scale
                    
                    frame = final_frame_float.astype(np.uint8)
                else:
                    condition = np.stack((aggregate_mask > 0,) * 3, axis=-1)
                    frame = np.where(condition, frame, bg_image)
            else:
                frame = bg_image
            
            fps_voluti = cv2.getTrackbarPos('FPS Rete', 'Regia Ledwall')
            if fps_voluti == 0:
                fps_voluti = 1  # Evita divisione per zero
            delay_ms = int(1000 / fps_voluti)
            
            # --- RISOLUZIONE DEL PROBLEMA DELLO STRETCH ---
            # Calcola il fattore di scala per mantenere le proporzioni (Aspect Ratio reale)
            # Inoltre aggiungiamo * 0.9 per fare la silhouette leggermente più piccola e non toccare i bordi.
            h_in, w_in = frame.shape[:2]
            scala_reale = min(PANEL_HEIGHT / h_in, TOTAL_WIDTH / w_in) * 0.90
            
            new_w = int(w_in * scala_reale)
            new_h = int(h_in * scala_reale)
            
            frame_scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Crea un "tappeto" nero grande quanto l'intero Ledwall
            frame_ridimensionato = np.zeros((PANEL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
            
            # Incolla l'immagine proporzionata esattamente al centro del ledwall
            y_offset = (PANEL_HEIGHT - new_h) // 2
            x_offset = (TOTAL_WIDTH - new_w) // 2
            frame_ridimensionato[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame_scaled
            
            frame_rgb = cv2.cvtColor(frame_ridimensionato, cv2.COLOR_BGR2RGB)
            
            # Applica gamma all'intero frame
            frame_rgb = gamma_table[frame_rgb]
            
            # Inverti colori se Common Anode
            if COMMON_ANODE:
                frame_rgb = 255 - frame_rgb
            
            # --- AFFETTA E SPEDISCI A OGNI PANNELLO ---
            if udp_sock is not None:
                for indice, ip in enumerate(ESP_IPS):
                    taglio_x_inizio = indice * PANEL_WIDTH
                    taglio_x_fine = (indice + 1) * PANEL_WIDTH
                    
                    # Estrae la fetta per questo pannello
                    fetta = frame_rgb[:, taglio_x_inizio:taglio_x_fine]
                    
                    # La matrice fisica è cablata per RIGHE orizzontali, non colonne! Niente transpose!
                    fetta_righe = fetta.copy().astype(np.uint8)
                    
                    # Applica Serpentina Orizzontale ESP
                    if ESP_START_BOTTOM:
                        fetta_righe = fetta_righe[::-1, :, :]
                    
                    if ESP_SERPENTINE_HORIZONTAL:
                        # Invertiamo l'ordine dei pixel X (colonne) per le righe Y dispari
                        fetta_righe[1::2] = fetta_righe[1::2, ::-1, :]
                        
                    dati_grezzi = fetta_righe.flatten().tobytes()
                    
                    # Taglio esattamente a metà dei byte (15x44 = 660 LED = 1980 byte -> metà = 990 byte = 330 LED)
                    meta = len(dati_grezzi) // 2
                    pacchetto_0 = bytes([0]) + dati_grezzi[:meta]
                    pacchetto_1 = bytes([1]) + dati_grezzi[meta:]
                    
                    try:
                        udp_sock.sendto(pacchetto_0, (ip, ESP_PORT))
                        udp_sock.sendto(pacchetto_1, (ip, ESP_PORT))
                        
                        # Pausa 3ms per far respirare il router tra un pannello e l'altro
                        time.sleep(0.003)
                    except Exception:
                        pass
            
            # --- INVIA FRAME ALL'ARDUINO (video seriale) ---
            if arduino_ser is not None:
                # 1. Controlla se l'Arduino ha mandato l'ACK 'K' (resiste agli scollegamenti fisici)
                try:
                    if arduino_ser.in_waiting > 0:
                        risposta = arduino_ser.read_all()
                        if b'K' in risposta:
                            arduino_ready = True
                except OSError:
                    arduino_ser = None
                    print("\n[!] ATTENZIONE: Cavo Arduino scollegato improvvisamente. Disattivazione sicura seriale completata.")
                
                # 1b. Timeout di sicurezza: se Arduino ha perso un frame e non ha mandato K
                # per più di 0.5 secondi, forza l'invio per sbloccare la situazione.
                if not arduino_ready and (time.time() - arduino_last_send_time > 0.5):
                    arduino_ready = True
                
                # 2. Se l'Arduino è pronto, invia un nuovo frame
                if arduino_ready:
                    try:
                        send_arduino_frame(arduino_ser, frame)
                        arduino_ready = False # Aspetta il prossimo 'K'
                        arduino_last_send_time = time.time()
                    except Exception as e:
                        print(f"\n[X] Errore invio frame ad Arduino: {e}")
                        arduino_ser = None  # Disabilita per non floodare la console
                        print("  -> Arduino scollegato a causa dell'errore.")
            
            # Mostra l'immagine ridimensionata a schermo (anteprima ledwall)
            cv2.imshow('Regia Ledwall', frame_ridimensionato)

            # Input - delay_ms controllato dallo slider FPS
            key = cv2.waitKey(delay_ms) & 0xFF
            
            if key == ord('q') or key == 27:
                print("\n[BYE] Arrivederci!")
                break
            elif key == ord('f'):
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty('Regia Ledwall', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty('Regia Ledwall', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            elif key == ord('i'):
                COMMON_ANODE = not COMMON_ANODE
                state = "ATTIVA" if COMMON_ANODE else "DISATTIVA"
                print(f"\n[TOGGLE] Modalità Inversione: {state}")
            elif key == ord('s'):
                solid_silhouette = not solid_silhouette
                state = "TINTA UNITA (Colore Dominante Reale)" if solid_silhouette else "FOTOCAMERA (Colori originali)"
                print(f"\n[TOGGLE] Stile Silhouette: {state}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if udp_sock:
            try:
                print("[LED] Spegnimento LED ESP...")
                # Invia frame nero a tutti i pannelli ESP
                frame_nero = bytes(PANEL_WIDTH * PANEL_HEIGHT * 3)
                for ip in ESP_IPS:
                    meta = len(frame_nero) // 2
                    udp_sock.sendto(bytes([0]) + frame_nero[:meta], (ip, ESP_PORT))
                    udp_sock.sendto(bytes([1]) + frame_nero[meta:], (ip, ESP_PORT))
                    time.sleep(0.02)
                udp_sock.close()
                print("[OK] LED ESP spenti. Socket UDP chiuso.")
            except Exception:
                pass
        if arduino_ser:
            try:
                print("[LED] Spegnimento LED Arduino...")
                # Invia frame nero all'Arduino
                arduino_ser.write(b'\xFF\x4C\x45' + bytes(ARDUINO_ROWS * ARDUINO_COLS * 3))
                time.sleep(0.1)
                arduino_ser.close()
                print("[OK] LED Arduino spenti. Seriale chiusa.")
            except Exception:
                pass


if __name__ == "__main__":
    main()