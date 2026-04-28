import cv2
import numpy as np
import time
import socket
import glob
import os
import wave
import random
import threading
import sys

try:
    import serial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    print("[!] pyserial non installato — Arduino video disabilitato")

try:
    import pygame
    import pygame.sndarray
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("[!] pygame non installato — audio disabilitato")

# ============================================================
# CONFIGURAZIONE (OTTIMIZZATA PER RASPBERRY PI)
# ============================================================
ESP_IPS = ["192.168.1.61", "192.168.1.62", "192.168.1.63", "192.168.1.64", "192.168.1.65",  "192.168.1.68"]
ESP_PORT = 4210
PANEL_WIDTH = 15
PANEL_HEIGHT = 44
TOTAL_WIDTH = PANEL_WIDTH * len(ESP_IPS)
ESP_SERPENTINE_HORIZONTAL = True
ESP_START_BOTTOM = False

ARDUINO_ENABLED = True
ARDUINO_PORT = "auto"
ARDUINO_BAUD = 500000
ARDUINO_ROWS = 32
ARDUINO_COLS = 56
ARDUINO_PANEL_W = 8
ARDUINO_PANEL_H = 32
ARDUINO_PANELS_COUNT = 7
ARDUINO_PANEL_ORDER = [6, 5, 4, 3, 2, 1, 0]
ARDUINO_PANEL_START_BOTTOM = [False] * 7
ARDUINO_SERPENTINE_X = True

GAMMA = 2.5           
COMMON_ANODE = False  

gamma_table = np.array([((i / 255.0) ** GAMMA) * 255 for i in np.arange(0, 256)]).astype("uint8")

# ============================================================
# PRECOMPILAZIONE MAPPATURA ARDUINO (O(1) a runtime)
# ============================================================
def build_arduino_mapping():
    """Genera la mappatura O(1) per convertire un'immagine nella stringa byte Seriale."""
    map_y = np.zeros(ARDUINO_ROWS * ARDUINO_COLS, dtype=int)
    map_x = np.zeros(ARDUINO_ROWS * ARDUINO_COLS, dtype=int)
    idx = 0
    for p in range(ARDUINO_PANELS_COUNT):
        panel_pos_x = ARDUINO_PANEL_ORDER[p]
        start_x = panel_pos_x * ARDUINO_PANEL_W
        starts_bottom = ARDUINO_PANEL_START_BOTTOM[p]
        for y_local in range(ARDUINO_PANEL_H):
            global_y = (ARDUINO_PANEL_H - 1 - y_local) if starts_bottom else y_local
            for x_local in range(ARDUINO_PANEL_W):
                eff_x = (ARDUINO_PANEL_W - 1 - x_local) if (ARDUINO_SERPENTINE_X and y_local % 2 == 1) else x_local
                map_y[idx] = global_y
                map_x[idx] = start_x + eff_x
                idx += 1
    return map_y, map_x

MAP_Y, MAP_X = build_arduino_mapping()

# ============================================================
# NETWORKING
# ============================================================
def create_udp_socket():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"[OK] Socket UDP creato per {len(ESP_IPS)} pannelli")
        return sock
    except Exception as e:
        print(f"[X] Errore socket UDP: {e}")
        return None

def create_arduino_serial():
    if not ARDUINO_ENABLED or not HAS_SERIAL: return None
    port = ARDUINO_PORT
    if port == "auto":
        ports = glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*') + glob.glob('/dev/cu.*') + glob.glob('/dev/tty.*')
        if not ports:
            print("[!] Nessuna seriale trovata per l'Arduino.")
            return None
        port = ports[0]
    try:
        ser = serial.Serial(port, ARDUINO_BAUD, timeout=0.01)
        time.sleep(2)
        ser.read_all()
        print(f"[OK] Arduino connesso su {port}")
        return ser
    except Exception as e:
        print(f"[X] Errore Arduino: {e}")
        return None

def send_arduino_frame(ser, frame, use_gamma=True):
    small = cv2.resize(frame, (ARDUINO_COLS, ARDUINO_ROWS), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    rgb = cv2.flip(rgb, 0)
    if use_gamma: rgb = gamma_table[rgb]
    if COMMON_ANODE: rgb = 255 - rgb
    
    # Fast indexing mapping (risparmia molta CPU rispetto ai loop in Python)
    mapped = rgb[MAP_Y, MAP_X]
    ser.write(b'\xFF\x4C\x45' + mapped.tobytes())


# ============================================================
# AUDIO (Estrema ottimizzazione RAM/CPU con Pre-Caching)
# ============================================================
PENTATONIC_SEMITONES = [-12, -10, -8, -5, -3, 0, 2, 4, 7, 9, 12, 14, 16, 19, 21]

class SoundManager:
    def __init__(self, sound_dir='sound', sample_rate=44100):
        self.enabled = False
        if not HAS_PYGAME: return
        self.sample_rate = sample_rate
        try:
            # Buffer più alto (1024) per evitare audio stutters su Raspberry Pi
            pygame.mixer.init(frequency=sample_rate, size=-16, channels=2, buffer=1024)
            pygame.mixer.set_num_channels(24) # Ridotti per leggerezza (max 24 note sovrapposte)
        except Exception as e:
            print(f"[!] Errore audio: {e}")
            return
            
        # Reverb attenuato in complessità (meno riflessioni) per far girare leggero il precalcolo
        self.reverb_early_taps =  [0.023, 0.041, 0.067, 0.097] 
        self.reverb_early_gains = [0.30,  0.22,  0.15,  0.10]
        self.reverb_late_taps =   [0.19,  0.37,  0.63,  1.03,  1.67] 
        self.reverb_late_gains =  [0.16,  0.12,  0.08,  0.05,  0.02]
        self.reverb_wet_mix = 1.0
        
        self.precomputed = {} # Cache dict: (fname, semitone) -> (numpy_array, pygame_Sound)
        self.person_wav = {}
        self.person_last_pos = {}
        self.person_last_trigger = {}
        self.movement_threshold = 20
        self.cooldown = 0.15
        
        self.delay_time = 0.4
        self.delay_volume = 0.25
        self.delay_feedback_base = 0.3
        self.delay_feedback_max = 0.88
        self.silence_threshold = 2.5
        self.silence_ramp = 5.0
        self.max_delay_generations = 6 # Massimo 6 generazioni di delay per non fondere la CPU
        self.last_note_time = time.time()
        self.pending_delays = []
        self.delay_lock = threading.Lock()
        
        self._load_and_precompute(sound_dir)
        
        if self.precomputed:
            self.enabled = True
            self._delay_running = True
            self._delay_thread = threading.Thread(target=self._delay_loop, daemon=True)
            self._delay_thread.start()
            print(f"[♪] Audio Pronta: pre-caching di tutti i pitch/reverb completato ({len(self.person_wav_keys)} WAV base)")
            
    def _load_and_precompute(self, sound_dir):
        """Carica, Normalizza, e pre-applica Pitch e Reverb creando Oggetti Audio pronti."""
        if not os.path.isdir(sound_dir): return
        print("[♪] Precalcolo audio per Raspberry Pi. Attendere qualche secondo...")
        raw_data = {}
        global_peak = 0
        self.person_wav_keys = []
        
        for fname in sorted(os.listdir(sound_dir)):
            if fname.lower().endswith('.wav'):
                path = os.path.join(sound_dir, fname)
                try:
                    with wave.open(path, 'rb') as wf:
                        sr, n_frames = wf.getframerate(), wf.getnframes()
                        raw = wf.readframes(n_frames)
                        audio = np.frombuffer(raw, dtype=np.int16).copy()
                        if wf.getnchannels() > 1:
                            audio = audio.reshape(-1, wf.getnchannels())[:, :2]
                        else:
                            audio = np.column_stack([audio, audio])
                        
                        if sr != self.sample_rate:
                            audio = self._resample_rate(audio, sr, self.sample_rate)
                            
                        raw_data[fname] = audio
                        peak = np.max(np.abs(audio.astype(np.float64)))
                        if peak > global_peak: global_peak = peak
                except Exception: pass
                
        if not raw_data: return
        norm_factor = (32767 * 0.85) / global_peak if global_peak > 0 else 1.0
        
        # Generatore: per ogni wav, pre-calcola i 15 semitoni possibili
        for fname, audio in raw_data.items():
            self.person_wav_keys.append(fname)
            norm_audio = (audio.astype(np.float64) * norm_factor).clip(-32767, 32767).astype(np.int16)
            for semitone in set(PENTATONIC_SEMITONES):
                pitched = self._pitch_shift(norm_audio, semitone)
                reverbed = self._apply_reverb(pitched)
                sound_obj = pygame.sndarray.make_sound(np.ascontiguousarray(reverbed))
                self.precomputed[(fname, semitone)] = (reverbed, sound_obj)

    def _resample_rate(self, audio, old_sr, new_sr):
        ratio = new_sr / old_sr
        indices = np.linspace(0, len(audio) - 1, max(1, int(len(audio) * ratio)))
        res = np.zeros((len(indices), 2), dtype=np.int16)
        res[:, 0] = np.interp(indices, np.arange(len(audio)), audio[:, 0].astype(np.float64))
        res[:, 1] = np.interp(indices, np.arange(len(audio)), audio[:, 1].astype(np.float64))
        return res

    def _pitch_shift(self, audio, semitones):
        if semitones == 0: return audio.copy()
        factor = 2 ** (semitones / 12.0)
        indices = np.linspace(0, len(audio) - 1, max(1, int(len(audio) / factor)))
        res = np.zeros((len(indices), 2), dtype=np.int16)
        res[:, 0] = np.interp(indices, np.arange(len(audio)), audio[:, 0].astype(np.float64))
        res[:, 1] = np.interp(indices, np.arange(len(audio)), audio[:, 1].astype(np.float64))
        return res

    def _apply_reverb(self, audio):
        sr = self.sample_rate
        all_taps = self.reverb_early_taps + self.reverb_late_taps
        all_gains = self.reverb_early_gains + self.reverb_late_gains
        max_tap = int(max(all_taps) * sr)
        audio_f = audio.astype(np.float64)
        total_len = len(audio_f) + max_tap
        wet = np.zeros((total_len, 2), dtype=np.float64)
        
        for tap_sec, gain in zip(all_taps, all_gains):
            d = int(tap_sec * sr)
            n = min(d + len(audio_f), total_len) - d
            wet[d:d+n] += audio_f[:n] * gain
            
        for tap_sec, gain in zip(self.reverb_late_taps[:3], self.reverb_late_gains[:3]):
            d = int(tap_sec * sr)
            src_end = min(len(wet) - d, len(wet))
            if src_end > 0 and d < len(wet):
                wet[d:d + src_end] += wet[:src_end] * (gain * 0.4)
                
        dry_gain = 1.0 - self.reverb_wet_mix * 0.5
        res = np.zeros((total_len, 2), dtype=np.float64)
        res[:len(audio_f)] = audio_f * dry_gain
        res += wet * self.reverb_wet_mix
        
        max_val = np.max(np.abs(res))
        if max_val > 32767: res *= (32767 / max_val)
        return res.astype(np.int16)

    def _apply_lowpass(self, audio, strength=0.4):
        # Filtro cumulativo vettorizzato (molto più veloce di np.convolve)
        k = max(3, int(strength * 15))
        if k < 2 or len(audio) <= k: return audio
        ret = np.cumsum(audio.astype(np.float64), axis=0)
        ret[k:] = ret[k:] - ret[:-k]
        return (ret[k - 1:] / k).astype(np.int16)

    def _delay_loop(self):
        while self._delay_running:
            time.sleep(0.05)  # Tick rate rilassato per risparmiare CPU RPi
            if not self.enabled: continue
            
            now = time.time()
            to_play, remaining = [], []
            
            with self.delay_lock:
                for audio, play_time, gen in self.pending_delays:
                    if now >= play_time:
                        silence = now - self.last_note_time
                        fb = self.delay_feedback_base
                        if silence > self.silence_threshold:
                            t = min(1.0, (silence - self.silence_threshold) / self.silence_ramp)
                            fb += t * (self.delay_feedback_max - self.delay_feedback_base)
                            
                        vol = self.delay_volume * (fb ** (gen * 0.5))
                        if vol > 0.03 and gen < self.max_delay_generations:
                            to_play.append((audio, vol))
                            filtered = self._apply_lowpass(audio, strength=0.3 + gen * 0.05)
                            remaining.append((filtered, now + self.delay_time, gen + 1))
                    else:
                        remaining.append((audio, play_time, gen))
                self.pending_delays = remaining
                
            for audio, vol in to_play:
                try:
                    scaled = audio.astype(np.float64) * vol
                    np.clip(scaled, -32767, 32767, out=scaled)
                    pygame.sndarray.make_sound(np.ascontiguousarray(scaled.astype(np.int16))).play()
                except Exception: pass

    def update(self, person_id, cx, cy, frame_w):
        if not self.enabled: return
        
        if person_id not in self.person_wav:
            self.person_wav[person_id] = random.choice(self.person_wav_keys)
            self.person_last_pos[person_id] = (cx, cy)
            self.person_last_trigger[person_id] = 0
            return
            
        lx, ly = self.person_last_pos[person_id]
        dist = np.hypot(cx - lx, cy - ly)
        now = time.time()
        
        if dist > self.movement_threshold and (now - self.person_last_trigger[person_id]) > self.cooldown:
            x_ratio = cx / max(1, frame_w)
            idx = int(x_ratio * (len(PENTATONIC_SEMITONES) - 1))
            idx = max(0, min(idx, len(PENTATONIC_SEMITONES) - 1))
            semi = PENTATONIC_SEMITONES[idx]
            
            # Accesso O(1) invece di calcoli in realtime!
            audio_arr, sound_obj = self.precomputed.get((self.person_wav[person_id], semi), (None, None))
            
            if sound_obj:
                try: sound_obj.play()
                except Exception: pass
                
                with self.delay_lock:
                    self.pending_delays.append((audio_arr, now + self.delay_time, 1))
            
            self.last_note_time = now
            self.person_last_pos[person_id] = (cx, cy)
            self.person_last_trigger[person_id] = now
        elif dist > self.movement_threshold:
            self.person_last_pos[person_id] = (cx, cy)

    def cleanup(self):
        self._delay_running = False
        if HAS_PYGAME and self.enabled: pygame.mixer.quit()

# ============================================================
# TRACKING VISIONE (Ottimizzato)
# ============================================================
class SimpleTracker:
    def get_dominant_color(self, frame_bgr, mask_2d):
        pixels = frame_bgr[mask_2d]
        if len(pixels) == 0: return (128, 128, 128)
        
        # Sottocampionamento drastico via numpy slicing (O(1)) anziché np.random
        max_samples = 300 
        if len(pixels) > max_samples:
            step = len(pixels) // max_samples
            pixels = pixels[::step][:max_samples]
            
        pixels_f = pixels.astype(np.float32)
        n_clusters = min(3, len(pixels_f))
        
        # Meno iterazioni per sgravare la CPU
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 5, 1.0)
        _, labels, centers = cv2.kmeans(pixels_f, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        
        dominant_bgr = centers[np.argmax(np.bincount(labels.flatten()))]
        
        hsv = cv2.cvtColor(np.uint8([[dominant_bgr]]), cv2.COLOR_BGR2HSV)[0][0].astype(np.float64)
        hsv[1] = min(255, hsv[1] * 1.5) 
        hsv[2] = max(60, min(255, hsv[2] * 1.2)) 
        boosted = cv2.cvtColor(np.uint8([[hsv]]), cv2.COLOR_HSV2BGR)[0][0]
        
        return (int(boosted[0]), int(boosted[1]), int(boosted[2]))


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "=" * 50)
    print("  REGIA LEDWALL LIGHT - Ottimizzato per Raspberry Pi")
    print("=" * 50)
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[X] Impossibile aprire la webcam!")
        return
        
    # Check per l'Arduino (spostato DOPO la webcam come richiesto)
    arduino_ser = create_arduino_serial()
    arduino_ready = True
    arduino_last_send_time = time.time()
    
    # Risoluzione estremamente bassa in lettura per liberare la CPU
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    
    try:
        from ultralytics import YOLO
        # Il flag imgsz=160 dimezza la pesantezza IA in cambio di precisione bordi
        model = YOLO('yolov8n-seg.pt')
    except Exception as e:
        print(f"[!] Errore YOLO: {e}")
        return

    silhouette_tracker = SimpleTracker()
    solid_silhouette = True
    sound_manager = SoundManager(sound_dir='sound')
    
    global COMMON_ANODE
    udp_sock = create_udp_socket()
    
    # Pre-allocazione buffer
    bg_image = None
    final_frame_float = None
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            original_frame = frame.copy()
            
            # Parametro imgsz basso fa girare la rete IA più velocemente
            results = model.track(frame, persist=True, classes=[0], verbose=False, imgsz=160)
            
            if bg_image is None or bg_image.shape != frame.shape:
                bg_image = np.zeros(frame.shape, dtype=np.uint8)
                final_frame_float = np.zeros(frame.shape, dtype=np.float32)
            else:
                final_frame_float.fill(0)
                
            r = results[0]
            if r.masks is not None and r.boxes.id is not None:
                ids = r.boxes.id.cpu().numpy().astype(int)
                masks_data = r.masks.data.cpu().numpy()
                boxes = r.boxes.xyxy.cpu().numpy()
                
                if not solid_silhouette:
                    aggregate_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                    
                # Maschere gestite tramite generatore/slicing
                for i, person_id in enumerate(ids):
                    mask_resized = cv2.resize(masks_data[i], (frame.shape[1], frame.shape[0])) > 0.5
                    
                    if i < len(boxes):
                        x1, y1, x2, y2 = boxes[i]
                        sound_manager.update(person_id, (x1+x2)/2, (y1+y2)/2, frame.shape[1])
                        
                    if solid_silhouette:
                        col = silhouette_tracker.get_dominant_color(original_frame, mask_resized)
                        # Dilation minima per i bordi ma superleggera su frame 320x240
                        mask_expanded = cv2.dilate(mask_resized.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
                        final_frame_float[mask_expanded] += np.array(col, dtype=np.float32)
                    else:
                        aggregate_mask[mask_resized] = 255
                        
                if solid_silhouette:
                    # Normalizzazione senza divisioni non necessarie
                    max_vals = np.max(final_frame_float, axis=-1, keepdims=True)
                    scale = np.ones_like(max_vals)
                    np.divide(255.0, max_vals, out=scale, where=max_vals > 255.0)
                    final_frame_float *= scale
                    frame = final_frame_float.astype(np.uint8)
                else:
                    frame = np.where(np.stack((aggregate_mask > 0,)*3, axis=-1), frame, bg_image)
            else:
                frame = bg_image
                
            # Scala l'immagine
            h_in, w_in = frame.shape[:2]
            scala = min(PANEL_HEIGHT / h_in, TOTAL_WIDTH / w_in) * 0.90
            new_w, new_h = int(w_in * scala), int(h_in * scala)
            
            frame_scaled = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            frame_ridimensionato = np.zeros((PANEL_HEIGHT, TOTAL_WIDTH, 3), dtype=np.uint8)
            yo, xo = (PANEL_HEIGHT - new_h) // 2, (TOTAL_WIDTH - new_w) // 2
            frame_ridimensionato[yo:yo+new_h, xo:xo+new_w] = frame_scaled
            
            frame_rgb = cv2.cvtColor(frame_ridimensionato, cv2.COLOR_BGR2RGB)
            frame_rgb = gamma_table[frame_rgb]
            if COMMON_ANODE: frame_rgb = 255 - frame_rgb
            
            # --- Invio pacchetti UDP ---
            if udp_sock:
                for i, ip in enumerate(ESP_IPS):
                    fetta = frame_rgb[:, i*PANEL_WIDTH : (i+1)*PANEL_WIDTH].copy()
                    if ESP_START_BOTTOM: fetta = fetta[::-1, :, :]
                    if ESP_SERPENTINE_HORIZONTAL: fetta[1::2] = fetta[1::2, ::-1, :]
                        
                    raw_d = fetta.tobytes()
                    m = len(raw_d) // 2
                    try:
                        udp_sock.sendto(bytes([0]) + raw_d[:m], (ip, ESP_PORT))
                        udp_sock.sendto(bytes([1]) + raw_d[m:], (ip, ESP_PORT))
                    except: pass
                    
            # --- Invio seriale Arduino ---
            if arduino_ser:
                try:
                    if arduino_ser.in_waiting > 0 and b'K' in arduino_ser.read_all():
                        arduino_ready = True
                except OSError:
                    arduino_ser = None
                    
                if not arduino_ready and (time.time() - arduino_last_send_time > 0.5):
                    arduino_ready = True
                    
                if arduino_ready:
                    try:
                        send_arduino_frame(arduino_ser, frame, use_gamma=True) 
                        arduino_ready = False
                        arduino_last_send_time = time.time()
                    except: arduino_ser = None
            
            # Mostra la GUI su monitor (riattivato per i test)
            cv2.imshow('Regia Ledwall Light', frame_ridimensionato)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'): break
            
    finally:
        cap.release()
        if udp_sock:
            try:
                nero = bytes(PANEL_WIDTH * PANEL_HEIGHT * 3)
                m = len(nero) // 2
                for ip in ESP_IPS:
                    udp_sock.sendto(bytes([0]) + nero[:m], (ip, ESP_PORT))
                    udp_sock.sendto(bytes([1]) + nero[m:], (ip, ESP_PORT))
                udp_sock.close()
            except: pass
        if arduino_ser:
            try:
                arduino_ser.write(b'\xFF\x4C\x45' + bytes(ARDUINO_ROWS * ARDUINO_COLS * 3))
                time.sleep(0.1)
                arduino_ser.close()
            except: pass
        sound_manager.cleanup()

if __name__ == "__main__":
    main()
