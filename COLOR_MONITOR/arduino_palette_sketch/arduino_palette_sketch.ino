/*
 * ARDUINO VIDEO ONLY - Ricezione frame seriale per matrice LED 32x32
 * ==================================================================
 * Sketch DEDICATO alla ricezione video da Python via seriale.
 *
 * A differenza di arduino_palette_sketch.ino, questo sketch:
 *   - NON esegue animazioni in background
 *   - NON chiama FastLED.show() se non dopo aver ricevuto un frame
 *   - NON disabilita gli interrupt inutilmente
 *
 * Questo evita la perdita di dati seriali causata da FastLED.show()
 * che disabilita gli interrupt per ~31ms su 1024 LED WS2812B.
 *
 * Protocollo: Magic Header (0xFF 0x4C 0x45) + 5376 byte RGB
 * Risposta: 'K' dopo frame mostrato con successo
 *
 * Usa questo sketch con: test_arduino.py e minimalv2.py
 */

#include <FastLED.h>

// --- CONFIGURAZIONE MATRICE ---
#define LED_PIN 6
#define NUM_LEDS 1792 // 7x Matrice 8x32 = 56x32
#define BRIGHTNESS 40 // Sicurezza alimentazione
#define LED_TYPE WS2812B
#define COLOR_ORDER GRB

CRGB leds[NUM_LEDS];

// Contatore frame per debug
unsigned long frameCount = 0;
unsigned long errorCount = 0;
unsigned long lastStatusTime = 0;

void setup() {
  Serial.begin(500000);
  Serial.setTimeout(1000); // 1 secondo di timeout per readBytes
  pinMode(13, OUTPUT);

  FastLED.addLeds<LED_TYPE, LED_PIN, COLOR_ORDER>(leds, NUM_LEDS)
      .setCorrection(TypicalLEDStrip);
  FastLED.setBrightness(BRIGHTNESS);
  FastLED.clear();
  FastLED.show();

  // Segnale di avvio: 3 lampeggi bianchi
  for (int i = 0; i < 3; i++) {
    fill_solid(leds, NUM_LEDS, CRGB(30, 30, 30));
    FastLED.show();
    delay(150);
    FastLED.clear();
    FastLED.show();
    delay(150);
  }

  // Mostra una riga verde in alto come segnale "pronto"
  for (int x = 0; x < 32; x++) {
    leds[x] = CRGB(0, 40, 0);
  }
  FastLED.show();

  Serial.println("VIDEO_READY");
  Serial.println("In attesa di frame video...");
}

void loop() {
  // ============================================
  // SOLO ricezione frame video - nient'altro!
  // ============================================

  // Cerca il magic header: 0xFF 0x4C 0x45
  if (Serial.available() > 0) {
    uint8_t firstByte = Serial.peek();

    if (firstByte == 0xFF) {
      // Potrebbe essere l'inizio del magic header
      if (Serial.available() >= 3) {
        uint8_t hdr[3];
        Serial.readBytes((char *)hdr, 3);

        if (hdr[0] == 0xFF && hdr[1] == 0x4C && hdr[2] == 0x45) {
          // ===== MAGIC HEADER VALIDO =====
          // Leggi i 3072 byte del frame (1024 LED × 3 byte RGB)
          int bytesRead = Serial.readBytes((char *)leds, NUM_LEDS * 3);

          if (bytesRead == NUM_LEDS * 3) {
            // Frame completo: mostra e conferma
            FastLED.show();
            Serial.write('K');

            frameCount++;
            digitalWrite(13, frameCount & 1); // LED 13 lampeggia ad ogni frame
          } else {
            // Frame incompleto: flush e segnala errore
            errorCount++;
            while (Serial.available() > 0)
              Serial.read();

            // LED 13 acceso fisso = errore
            digitalWrite(13, HIGH);
          }
        } else {
          // Header falso: scarta e continua
          // (i 3 byte sono già stati consumati)
        }
      }
      // Se meno di 3 byte disponibili, aspetta il prossimo loop
    } else {
      // Non è 0xFF: scarta il byte spazzatura
      Serial.read();
    }
  }

  // === STATUS periodico (ogni 5 secondi) ===
  if (millis() - lastStatusTime > 5000) {
    lastStatusTime = millis();
    if (frameCount > 0 || errorCount > 0) {
      Serial.print("STAT: frame=");
      Serial.print(frameCount);
      Serial.print(" err=");
      Serial.println(errorCount);
    }
  }

  // NESSUN FastLED.show() qui!
  // NESSUN delay() qui!
  // Il loop gira il più veloce possibile per non perdere byte seriali.
}
