/*
 * gvms_final.ino
 * ==============
 * Ground Vibration Monitoring System (GVMS)
 *
 * Your original ESP32 code + ML inference additions:
 *   + #include HTTPClient.h and ArduinoJson.h
 *   + ML server IP/port constants
 *   + ML_DOWNSAMPLE=3  =>  100Hz sensor / 3 = ~33Hz into ML buffer
 *     (model trained at ~33Hz, so 50-sample window = ~1.5s as intended)
 *   + Sample ring buffer + sendToMLServer()
 *   + readSensor() pushes every 3rd sample into ML buffer
 *   + OLED shows ML label (startup splash changed, status says ALERT not EARTHQUAKE)
 *   + /data JSON adds mlLabel, mlConfidence, mlReady, mlServerOK
 *   + /events filter changed from "EARTHQUAKE" to "ALERT" (matches CSV)
 *   + Dashboard HTML adds ML prediction panel + class pills
 *   + Everything else (pins, thresholds, full rich dashboard) unchanged
 *
 * BEFORE FLASHING:
 *   1. Set ML_SERVER_IP_FALLBACK to your PC LAN IP  (run ipconfig on Windows)
 *   2. Start the Flask server:  python dashboard/app.py
 *   3. Install ArduinoJson v6 via Arduino Library Manager
 */

#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <MPU6050.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <ArduinoJson.h>
#include <ESPmDNS.h>

// =====================================================
// WIFI
// =====================================================

const char* WIFI_SSID = "PT";
const char* WIFI_PASSWORD = "zaidzaid";

// =====================================================
// ML SERVER  -- change ML_SERVER_IP_FALLBACK to your PC LAN IP
// =====================================================

const char* ML_SERVER_HOST = "gvms.local";   // mDNS hostname -- no IP needed ever again
const char* ML_SERVER_IP_FALLBACK = "10.83.76.117"; // fallback if mDNS fails

const int   ML_SERVER_PORT = 5000;

// 10-second batch mode
// Sensor at 100 Hz, downsample by 3 -> ~33 Hz -> 330 samples per 10s batch
// Python slides 50-sample windows over all 330 samples and majority-votes
const int ML_WINDOW_SIZE  = 50;    // must match Python WINDOW_SIZE
const int ML_DOWNSAMPLE   = 3;     // 100Hz / 3 = ~33Hz effective
const int ML_BATCH_SECS   = 10;    // record for this many seconds per batch
const int ML_BATCH_SIZE   = 330;   // ML_BATCH_SECS * (100/ML_DOWNSAMPLE) = 10*33

// =====================================================
// PIN DEFINITIONS
// =====================================================

#define SDA_PIN 21
#define SCL_PIN 22

#define SCREEN_WIDTH 128
#define SCREEN_HEIGHT 64
#define OLED_ADDRESS 0x3C

#define MPU_ADDRESS 0x68

#define BUZZER_PIN 27
#define GREEN_LED 25
#define RED_LED 26

#define SD_SCK 18
#define SD_MISO 19
#define SD_MOSI 23
#define SD_CS 5

// =====================================================
// OBJECTS
// =====================================================

Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &Wire, -1);
MPU6050 mpu(MPU_ADDRESS);
WebServer server(80);

// =====================================================
// STATUS
// =====================================================

bool oledOK = false;
bool mpuOK = false;
bool sdOK = false;
bool earthquake = false;
bool prevEarthquakeState = false;

// =====================================================
// MPU DATA
// =====================================================

int16_t ax = 0, ay = 0, az = 0;
int16_t prevAx = 0, prevAy = 0, prevAz = 0;
long vibration = 0;

// =====================================================
// EARTHQUAKE SETTINGS
// =====================================================

long EARTHQUAKE_THRESHOLD = 3000;
const long NOISE_THRESHOLD = 300;
const long MIN_THRESHOLD = 200;
const long MAX_THRESHOLD = 20000;

// =====================================================
// SESSION STATE
// =====================================================

long sessionAlertCount = 0;

bool demoMode = false;
unsigned long demoModeUntil = 0;

#define MAX_EVENTS 20
#define MAX_THRESHOLD_HISTORY 10

struct ThresholdChange {
  long value;
  unsigned long time;
};

ThresholdChange thresholdHistory[MAX_THRESHOLD_HISTORY];
int thresholdHistoryCount = 0;

// =====================================================
// TIMERS
// =====================================================

unsigned long lastSensorRead = 0;
unsigned long lastLog = 0;

const unsigned long SENSOR_INTERVAL = 10;    // 100 Hz
const unsigned long LOG_INTERVAL = 1000;

// =====================================================
// ML STATE  (10-second batch mode)
// =====================================================

struct Sample { int16_t ax, ay, az; long vs; };

// Batch buffer: holds one full 10-second recording
Sample batchBuffer[ML_BATCH_SIZE];
volatile int batchCount    = 0;     // how many samples collected so far
int    downsampleCtr = 0;     // counts raw reads for downsampling

// Batch timing
unsigned long batchStartMs = 0;    // millis() when current batch started
volatile bool batchRecording = true; // true = collecting, false = waiting for result

// ML result
String mlLabel      = "Recording...";
float  mlConfidence = 0.0f;
bool   mlReady      = false;
bool   mlServerOK   = false;
int    mlBatchNum   = 0;      // how many batches completed

// Ring buffer: last 10 completed batch results
#define RECENT_BATCH_MAX 10
String recentBatchLabels[RECENT_BATCH_MAX];
float  recentBatchConf[RECENT_BATCH_MAX];
int    recentBatchHead  = 0;
int    recentBatchCount = 0;

// Non-blocking HTTP: snapshot sent to Core 0 task
Sample mlSnapshot[ML_BATCH_SIZE];
volatile bool mlSendPending = false;
volatile bool mlTaskBusy    = false;
TaskHandle_t  mlTaskHandle  = NULL;

// =====================================================
// MPU6050 TEST
// =====================================================

bool testMPUReading() {

  int16_t testAX, testAY, testAZ;

  mpu.getAcceleration(&testAX, &testAY, &testAZ);

  Serial.print("MPU Test -> AX: ");
  Serial.print(testAX);
  Serial.print(" | AY: ");
  Serial.print(testAY);
  Serial.print(" | AZ: ");
  Serial.println(testAZ);

  if (testAX != 0 || testAY != 0 || testAZ != 0) {
    return true;
  }

  return false;
}

// =====================================================
// MPU SETUP
// =====================================================

void setupMPU() {

  Serial.println();
  Serial.println("Starting MPU6050...");

  mpu.initialize();
  delay(500);

  if (testMPUReading()) {

    mpuOK = true;
    Serial.println("MPU6050 CONNECTED!");
    Serial.println("MPU6050 READY!");

    mpu.getAcceleration(&ax, &ay, &az);
    prevAx = ax; prevAy = ay; prevAz = az;
    vibration = 0;

  } else {

    mpuOK = false;
    Serial.println("MPU6050 READ FAILED!");
    Serial.println("Check MPU wiring.");
  }
}

// =====================================================
// OLED UPDATE  (shows ML label; no "EARTHQUAKE" text)
// =====================================================

void updateOLED() {

  if (!oledOK) return;

  display.clearDisplay();
  display.setTextColor(SSD1306_WHITE);
  display.setTextSize(1);

  // Lines 1-2: sensor values
  display.setCursor(0, 0);
  display.print("AX:"); display.print(ax);
  display.print(" AY:"); display.println(ay);
  display.print("AZ:"); display.print(az);
  display.print(" VS:"); display.println(vibration);

  // Line 3: threshold status
  display.setCursor(0, 18);
  if (!mpuOK)          display.println("MPU ERROR");
  else if (earthquake) display.println("STATUS: ALERT");
  else                 display.println("STATUS: NORMAL");

  // Line 4: ML label or batch progress (sample-driven, not time-driven)
  display.setCursor(0, 28);
  if (batchRecording) {
    int pct = (int)((batchCount * 100L) / ML_BATCH_SIZE);
    display.print("REC ");
    display.print(pct);
    display.println("% ...");
  } else {
    display.print("ML:");
    String lbl = mlLabel;
    if (lbl.length() > 13) lbl = lbl.substring(0, 13);
    display.println(lbl);
  }

  // Line 5: confidence (batch count removed)
  display.setCursor(0, 40);
  if (!batchRecording && mlReady) {
    display.print("Conf:");
    display.print((int)(mlConfidence * 100));
    display.print("%");
    if (!mlServerOK) display.print(" [offline]");
  } else if (!batchRecording) {
    display.print("Sending...");
  }

  // Line 6: GROUND VIBRATION branding
  display.setCursor(0, 54);
  display.println("GROUND VIBRATION");

  display.display();
}

// =====================================================
// SEND TO ML SERVER  (runs on Core 0 via FreeRTOS task)
// The sensor loop on Core 1 is NEVER blocked by HTTP.
// =====================================================

// -- mlHttpTask: runs on Core 0, sends 10-second batch, gets label back --------
// -- Resolve ML server address via mDNS, fall back to hardcoded IP -----------
String resolveMLServer() {
  // ESP32 core v3.x: queryHost returns IPAddress directly (no output param)
  IPAddress ip = MDNS.queryHost(ML_SERVER_HOST, 2000);
  if (ip != IPAddress(0,0,0,0)) {
    Serial.print("[mDNS] Resolved gvms.local -> "); Serial.println(ip.toString());
    return ip.toString();
  }
  Serial.println("[mDNS] Resolve failed, using fallback IP");
  return String(ML_SERVER_IP_FALLBACK);
}

void mlHttpTask(void* param) {
  for (;;) {
    if (mlSendPending && !mlTaskBusy) {
      mlTaskBusy    = true;
      mlSendPending = false;

      Serial.print("[ML] Sending batch of ");
      Serial.print(ML_BATCH_SIZE);
      Serial.println(" samples to server...");

      if (WiFi.status() == WL_CONNECTED) {
        // Compact array format [[ax,ay,az,vs],...] ~6KB vs ~20KB named-key.
        // Built as plain String to avoid large ArduinoJson heap allocation.
        String body;
        body.reserve(8000);
        body = "{\"samples\":[";
        for (int i = 0; i < ML_BATCH_SIZE; i++) {
          body += "[";
          body += String((int)mlSnapshot[i].ax); body += ",";
          body += String((int)mlSnapshot[i].ay); body += ",";
          body += String((int)mlSnapshot[i].az); body += ",";
          body += String((long)mlSnapshot[i].vs);
          body += "]";
          if (i < ML_BATCH_SIZE - 1) body += ",";
        }
        body += "]}";

        String mlHost = resolveMLServer();
        String url = String("http://") + mlHost + ":" + ML_SERVER_PORT + "/api/infer/batch";

        HTTPClient http;
        http.begin(url);
        http.addHeader("Content-Type", "application/json");
        http.setTimeout(30000);  // 30s -- Flask feature extraction can be slow on first batch

        int code = http.POST(body);

        if (code == 200) {
          String resp = http.getString();
          // Flask returns batch_label + batch_conf at the top level.
          // StaticJsonDocument<512> is enough - we only read 4 scalar fields.
          StaticJsonDocument<512> res;
          DeserializationError jerr = deserializeJson(res, resp);
          if (!jerr) {
            const char* lbl  = res["batch_label"] | "";
            float        conf = res["batch_conf"]  | 0.0f;
            int          nproc = res["n_processed"] | 0;
            int          ems   = res["elapsed_ms"]  | 0;
            if (strlen(lbl) > 0) {
              mlLabel      = String(lbl);
              mlConfidence = conf;
              mlReady      = true;
              mlServerOK   = true;
              mlBatchNum++;
              recentBatchLabels[recentBatchHead] = String(lbl);
              recentBatchConf[recentBatchHead]   = conf;
              recentBatchHead  = (recentBatchHead + 1) % RECENT_BATCH_MAX;
              if (recentBatchCount < RECENT_BATCH_MAX) recentBatchCount++;
              Serial.print("[ML] Batch #"); Serial.print(mlBatchNum);
              Serial.print(" -> "); Serial.print(mlLabel);
              Serial.print("  conf="); Serial.print((int)(mlConfidence * 100));
              Serial.print("%  samples="); Serial.print(nproc);
              Serial.print("  server_ms="); Serial.println(ems);
            } else {
              Serial.println("[ML] batch_label missing in response");
              mlServerOK = false;
            }
          } else {
            Serial.print("[ML] JSON parse error: "); Serial.println(jerr.c_str());
            mlServerOK = false;
          }
        } else {
          mlServerOK = false;
          Serial.print("[ML] HTTP error: "); Serial.println(code);
        }
        http.end();

      mlTaskBusy    = false;
      batchRecording = true;           // start the next 10-second batch
      batchCount    = 0;
      batchStartMs  = millis();
    }
    vTaskDelay(20 / portTICK_PERIOD_MS);
  }
}

// Called from sensor loop when batch is full ? copies buffer, signals task
void triggerMLSend() {
  if (mlTaskBusy) {
    // Task still busy from last batch ? skip, restart recording
    batchRecording = true;
    batchCount     = 0;
    batchStartMs   = millis();
    return;
  }
  // Copy batch buffer into snapshot for the HTTP task
  for (int i = 0; i < ML_BATCH_SIZE; i++) {
    mlSnapshot[i] = batchBuffer[i];
  }
  mlSendPending  = true;   // wake up Core 0 task
  batchRecording = false;  // stop recording until result comes back
  mlLabel        = "Analyzing...";
  mlReady        = false;
}

// =====================================================
// READ MPU6050
// Runs at 100 Hz. Every ML_DOWNSAMPLE reads, one sample
// is pushed into the ML ring buffer (~33 Hz effective).
// =====================================================

void readSensor() {

  if (!mpuOK) return;

  mpu.getAcceleration(&ax, &ay, &az);

  long deltaX = abs((long)ax - (long)prevAx);
  long deltaY = abs((long)ay - (long)prevAy);
  long deltaZ = abs((long)az - (long)prevAz);

  vibration = deltaX + deltaY + deltaZ;

  if (vibration < NOISE_THRESHOLD) {
    vibration = 0;
  }

  earthquake = (vibration >= EARTHQUAKE_THRESHOLD);

  prevAx = ax; prevAy = ay; prevAz = az;

  // DEMO / TEST MODE OVERRIDE

  if (demoMode) {
    if (millis() < demoModeUntil) {
      earthquake = true;
      vibration = EARTHQUAKE_THRESHOLD + 500;
    } else {
      demoMode = false;
    }
  }

  // SESSION ALERT COUNT (rising edge only)

  if (earthquake && !prevEarthquakeState) {
    sessionAlertCount++;
  }
  prevEarthquakeState = earthquake;

  // OUTPUTS

  if (earthquake) {
    digitalWrite(RED_LED, HIGH);
    digitalWrite(GREEN_LED, LOW);
    digitalWrite(BUZZER_PIN, HIGH);
  } else {
    digitalWrite(RED_LED, LOW);
    digitalWrite(GREEN_LED, HIGH);
    digitalWrite(BUZZER_PIN, LOW);
  }

  // -- 10-second batch collection (downsampled to ~33 Hz) ----------------
  if (batchRecording) {
    downsampleCtr++;
    if (downsampleCtr >= ML_DOWNSAMPLE) {
      downsampleCtr = 0;

      if (batchCount < ML_BATCH_SIZE) {
        batchBuffer[batchCount] = { ax, ay, az, vibration };
        batchCount++;
      }

      // Batch full -> send it
      if (batchCount >= ML_BATCH_SIZE) {
        Serial.println("[ML] Batch full, triggering send...");
        triggerMLSend();
      }
    }
  }

  updateOLED();

  Serial.print("AX: "); Serial.print(ax);
  Serial.print(" | AY: "); Serial.print(ay);
  Serial.print(" | AZ: "); Serial.print(az);
  Serial.print(" | VIB: "); Serial.print(vibration);
  Serial.print(" | STATUS: "); Serial.print(earthquake ? "ALERT" : "NORMAL");
  Serial.print(" | ML: "); Serial.println(mlLabel);
}

// =====================================================
// CREATE CSV
// =====================================================

void createCSV() {

  if (!sdOK) return;

  if (!SD.exists("/earthquake.csv")) {

    File file = SD.open("/earthquake.csv", FILE_WRITE);

    if (file) {
      file.println("Time_ms,AX_raw,AY_raw,AZ_raw,Vibration,Status");
      file.close();
      Serial.println("earthquake.csv created");
    } else {
      Serial.println("CSV CREATE FAILED!");
    }

  } else {
    Serial.println("earthquake.csv already exists");
  }
}

// =====================================================
// SAVE DATA TO SD
// =====================================================

void saveData() {

  if (!sdOK || !mpuOK) return;

  File file = SD.open("/earthquake.csv", FILE_APPEND);

  if (file) {

    file.print(millis());
    file.print(",");
    file.print(ax);
    file.print(",");
    file.print(ay);
    file.print(",");
    file.print(az);
    file.print(",");
    file.print(vibration);
    file.print(",");
    file.println(earthquake ? "ALERT" : "NORMAL");

    file.close();

  } else {
    Serial.println("SD WRITE FAILED!");
  }
}

// =====================================================
// WEB PAGE  (your original rich dashboard + ML panel)
// =====================================================

// =====================================================
// WEB PAGE  -  full dashboard
// =====================================================
// =====================================================
// WEB PAGE  -  full GVMS dashboard
// =====================================================
String webpage() {
  String statusText;
  String statusClass;
  if (!mpuOK)          { statusText = "MPU6050 ERROR";   statusClass = "danger"; }
  else if (earthquake) { statusText = "VIBRATION ALERT"; statusClass = "danger"; }
  else                 { statusText = "SYSTEM NORMAL";    statusClass = "normal"; }

  String html = R"rawliteral(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>GVMS Ground Vibration Monitor</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Space+Grotesk:wght@400;700;800&display=swap');
*{box-sizing:border-box;margin:0;padding:0;}
:root{--blk:#000;--wht:#fff;--yel:#ffe600;--red:#ff2b2b;--grn:#00ff6a;--blu:#0057ff;--b:3px solid #000;}
body{font-family:'Space Grotesk',system-ui,sans-serif;background:var(--wht);color:var(--blk);padding-bottom:60px;}

/* -- HEADER -- */
header{background:var(--blk);border-bottom:5px solid var(--yel);padding:14px 24px;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;}
header h1{color:var(--yel);font-size:1.15rem;font-weight:800;letter-spacing:3px;text-transform:uppercase;font-family:'Space Grotesk',sans-serif;}
header .sub{color:#aaa;font-family:'Space Mono',monospace;font-size:.7rem;margin-top:3px;letter-spacing:1px;}
.hright{display:flex;align-items:center;gap:8px;flex-wrap:wrap;}
.badge{padding:4px 10px;font-size:.68rem;font-weight:700;font-family:'Space Mono',monospace;text-transform:uppercase;border:2px solid var(--blk);box-shadow:2px 2px 0 var(--blk);}
.bg{background:var(--grn);color:var(--blk);}
.br{background:var(--red);color:var(--wht);}
.bb{background:var(--yel);color:var(--blk);}
.hbtn{background:var(--wht);border:2px solid var(--blk);color:var(--blk);padding:5px 12px;cursor:pointer;font-size:.72rem;font-family:'Space Mono',monospace;font-weight:700;text-transform:uppercase;box-shadow:2px 2px 0 var(--blk);}
.hbtn:hover{background:var(--yel);}

/* -- TICKER -- */
.ticker{background:var(--yel);border-top:3px solid var(--blk);border-bottom:3px solid var(--blk);padding:5px 0;overflow:hidden;white-space:nowrap;}
.ticker-inner{display:inline-block;animation:tick 24s linear infinite;font-family:'Space Mono',monospace;font-size:.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--blk);}
@keyframes tick{0%{transform:translateX(110vw)}100%{transform:translateX(-100%)}}

/* -- CONNECTION WARN -- */
.conn-warn{background:var(--red);color:var(--wht);text-align:center;padding:7px;font-size:.8rem;font-weight:700;display:none;font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;border-bottom:3px solid var(--blk);}
.conn-warn.show{display:block;}

/* -- LAYOUT -- */
.container{max-width:1180px;margin:auto;padding:20px 18px;}
.tabs{display:flex;gap:0;margin-bottom:20px;overflow-x:auto;border:3px solid var(--blk);}
.tab{padding:10px 22px;background:var(--wht);color:#666;cursor:pointer;font-size:.78rem;font-weight:700;white-space:nowrap;font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;border-right:3px solid var(--blk);transition:background .1s;}
.tab:last-child{border-right:none;}
.tab:hover{background:#f0f0f0;}
.tab.active{background:var(--blk);color:var(--yel);}
.pane{display:none;}
.pane.active{display:block;}

/* -- GRIDS -- */
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:0;margin-bottom:20px;border:3px solid var(--blk);}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:0;margin-bottom:20px;border:3px solid var(--blk);}
@media(max-width:700px){.g4{grid-template-columns:1fr 1fr;}.g2{grid-template-columns:1fr;}}
@media(max-width:400px){.g4{grid-template-columns:1fr;}}

/* -- CARDS -- */
.card{background:var(--wht);border:3px solid var(--blk);padding:16px;margin:-1px;}
.g4 .card,.g2 .card{border:none;border-right:3px solid var(--blk);border-bottom:3px solid var(--blk);}
.g4 .card:nth-child(4n),.g2 .card:nth-child(2n){border-right:none;}
.card h3{font-family:'Space Mono',monospace;font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:8px;color:#555;}
.bv{font-size:1.7rem;font-weight:800;font-family:'Space Grotesk',sans-serif;color:var(--blk);}

/* -- STATUS BAR -- */
.statusbar{padding:14px 18px;text-align:center;font-size:1rem;font-weight:800;margin-bottom:20px;border:3px solid var(--blk);font-family:'Space Mono',monospace;letter-spacing:3px;text-transform:uppercase;box-shadow:5px 5px 0 var(--blk);}
.statusbar.normal{background:var(--grn);color:var(--blk);}
.statusbar.danger{background:var(--red);color:var(--wht);animation:blink .8s infinite;}
@keyframes blink{50%{opacity:.55;}}

/* -- ML PANEL -- */
.ml-panel{background:var(--blk);border:3px solid var(--blk);padding:20px;margin-bottom:20px;box-shadow:6px 6px 0 #555;}
.ml-panel .mph{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:14px;border-bottom:2px solid #333;padding-bottom:10px;}
.ml-panel .mph h2{color:var(--yel);font-size:.82rem;font-weight:800;letter-spacing:3px;text-transform:uppercase;font-family:'Space Mono',monospace;}
.ml-panel .mph .algo{color:#888;font-size:.67rem;font-family:'Space Mono',monospace;}
.ml-label{font-size:2.4rem;font-weight:800;color:var(--yel);text-align:center;padding:8px 0 4px;font-family:'Space Grotesk',sans-serif;letter-spacing:2px;}
.ml-sub{text-align:center;color:#aaa;font-size:.75rem;margin-bottom:10px;font-family:'Space Mono',monospace;}
.ml-bar-bg{background:#222;border:2px solid #555;height:12px;overflow:hidden;}
.ml-bar{height:12px;background:var(--yel);transition:width .5s ease;}
.pills{display:flex;flex-wrap:wrap;gap:6px;justify-content:center;margin-top:12px;}
.pill{padding:4px 12px;font-size:.68rem;font-weight:700;background:#1a1a1a;color:#666;border:2px solid #444;font-family:'Space Mono',monospace;text-transform:uppercase;transition:all .15s;}
.pill.hot{background:var(--yel);color:var(--blk);border-color:var(--yel);box-shadow:2px 2px 0 #888;}
.algo-box{background:#111;border-left:4px solid var(--yel);padding:8px 12px;font-size:.7rem;color:#888;margin-top:12px;line-height:1.7;font-family:'Space Mono',monospace;}

/* -- WAVEFORM -- */
.wave-card{background:var(--wht);border:3px solid var(--blk);padding:16px;margin-bottom:20px;box-shadow:4px 4px 0 #aaa;}
.whead{display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:6px;margin-bottom:10px;padding-bottom:8px;border-bottom:2px solid var(--blk);}
.whead h3{font-family:'Space Mono',monospace;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;}
.leg{display:flex;gap:12px;font-size:.68rem;font-family:'Space Mono',monospace;}
.ld{width:10px;height:10px;display:inline-block;margin-right:3px;vertical-align:middle;border:1px solid var(--blk);}
canvas{width:100%!important;display:block;}

/* -- SIGNAL ANALYSIS -- */
.sig-box{background:var(--wht);border:3px solid var(--blk);padding:16px;margin-bottom:20px;}
.sig-box h3{font-family:'Space Mono',monospace;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:2px;margin-bottom:12px;padding-bottom:8px;border-bottom:2px solid var(--blk);}
.srow{display:flex;align-items:center;gap:10px;margin:8px 0;}
.sl{width:120px;font-size:.7rem;color:#555;font-family:'Space Mono',monospace;text-transform:uppercase;}
.sb-bg{flex:1;background:#eee;border:2px solid var(--blk);height:10px;}
.sb{height:100%;transition:width .4s;}
.sv{width:50px;text-align:right;font-size:.72rem;font-weight:700;font-family:'Space Mono',monospace;}
.sig-note{margin-top:10px;font-size:.7rem;color:#555;font-family:'Space Mono',monospace;padding:8px;background:#f5f5f5;border-left:3px solid var(--blk);}

/* -- HISTORY TABLE -- */
.htable{width:100%;border-collapse:collapse;font-size:.76rem;font-family:'Space Mono',monospace;}
.htable th{padding:9px 10px;text-align:left;background:var(--blk);color:var(--yel);font-size:.65rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;}
.htable td{padding:7px 10px;border-bottom:2px solid #e0e0e0;vertical-align:middle;}
.htable tr:hover td{background:#fffde0;}
.obs{display:inline-block;padding:2px 8px;font-size:.65rem;font-weight:700;font-family:'Space Mono',monospace;border:2px solid var(--blk);}
.o-gt{background:#00ff6a;color:#000;}.o-ht{background:#0057ff;color:#fff;}
.o-mt{background:#a371f7;color:#fff;}.o-mht{background:#ffe600;color:#000;}
.o-w{background:#ff69b4;color:#000;}.o-al{background:#ff2b2b;color:#fff;}
.o-nr{background:#e0e0e0;color:#000;}.o-uk{background:#ccc;color:#000;}
.pg{display:flex;gap:7px;align-items:center;justify-content:flex-end;margin-top:10px;font-size:.72rem;font-family:'Space Mono',monospace;}

/* -- SETTINGS FORM -- */
.sform label{display:block;font-family:'Space Mono',monospace;font-size:.68rem;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:4px;margin-top:14px;color:#555;}
.sform input[type="number"]{background:var(--wht);border:3px solid var(--blk);color:var(--blk);padding:8px 12px;font-size:.88rem;font-family:'Space Mono',monospace;font-weight:700;width:160px;}
.sform input[type="number"]:focus{outline:none;box-shadow:4px 4px 0 var(--blk);}
.sbtn{background:var(--blk);color:var(--yel);border:3px solid var(--blk);padding:9px 20px;cursor:pointer;font-weight:700;font-size:.75rem;font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;box-shadow:4px 4px 0 #555;margin-top:10px;}
.sbtn:hover{background:#222;}
.sbtn:active{box-shadow:none;transform:translate(4px,4px);}
.sbtn.red{background:var(--red);color:var(--wht);}
.msg{font-size:.72rem;margin-top:7px;min-height:16px;font-family:'Space Mono',monospace;}
.msg.ok{color:#006600;font-weight:700;}.msg.err{color:var(--red);font-weight:700;}

/* -- LOG BOX -- */
.logbox{background:var(--blk);border:3px solid var(--blk);padding:10px;height:190px;overflow-y:auto;font-family:'Space Mono',monospace;font-size:.7rem;line-height:1.8;}
.logbox::-webkit-scrollbar{width:5px;}.logbox::-webkit-scrollbar-thumb{background:var(--yel);}
.logbox .le{color:#888;}
.logbox .le .ts{color:var(--grn);}.logbox .le .lb{color:var(--yel);font-weight:700;}

/* -- TOAST -- */
.toast{position:fixed;top:16px;left:50%;transform:translateX(-50%) translateY(-80px);background:var(--blk);color:var(--yel);padding:12px 22px;font-weight:700;box-shadow:5px 5px 0 #555;z-index:999;transition:transform .25s;font-family:'Space Mono',monospace;text-transform:uppercase;font-size:.8rem;border:3px solid var(--blk);}
.toast.show{transform:translateX(-50%) translateY(0);}

/* -- KV ROWS -- */
.kv{display:flex;justify-content:space-between;align-items:center;padding:9px 0;border-bottom:2px solid #e0e0e0;font-size:.78rem;font-family:'Space Mono',monospace;}
.kv:last-child{border-bottom:none;}.kv .k{color:#555;text-transform:uppercase;font-size:.68rem;letter-spacing:1px;}
.row-full{grid-column:1/-1;}

/* -- STANDALONE CARD (outside grids) -- */
.s-card{background:var(--wht);border:3px solid var(--blk);padding:18px;margin-bottom:20px;}
.s-card h3{font-family:'Space Mono',monospace;font-size:.72rem;font-weight:800;letter-spacing:2px;text-transform:uppercase;color:var(--blk);margin-bottom:10px;padding-bottom:8px;border-bottom:2px solid var(--blk);}
</style>
</head>
<body>
<div id="toast" class="toast"></div>
<div class="ticker"><span class="ticker-inner">&nbsp;&nbsp;&nbsp;SYSTEM ONLINE &nbsp;&#9670;&nbsp; ESP32 ACTIVE &nbsp;&#9670;&nbsp; MPU-6050 READY &nbsp;&#9670;&nbsp; ML BATCH MODE &nbsp;&#9670;&nbsp; RANDOM FOREST CLASSIFIER &nbsp;&#9670;&nbsp; GENTLE_TAP &nbsp;&middot;&nbsp; HAND_TAP &nbsp;&middot;&nbsp; MULTI_TAP &nbsp;&middot;&nbsp; MULTIHAND_TAP &nbsp;&middot;&nbsp; WALK1 &nbsp;&#9670;&nbsp; SYSTEM ONLINE &nbsp;&#9670;&nbsp; ESP32 ACTIVE &nbsp;&#9670;&nbsp; ML BATCH MODE &nbsp;&#9670;&nbsp;</span></div>
<div id="connWarn" class="conn-warn">&#9888; CONNECTION TO ESP32 LOST &mdash; RETRYING...</div>
<header>
  <div>
    <h1>/// GROUND VIBRATION MONITOR</h1>
    <div class="sub">ESP32 + MPU-6050 &nbsp;&middot;&nbsp; RANDOM FOREST ML &nbsp;&middot;&nbsp; 10S BATCH CLASSIFICATION</div>
  </div>
  <div class="hright">
    <span id="hAlerts" class="badge br">ALERTS: 0</span>
    <span id="hSD"     class="badge bb">SD: --</span>
    <span id="hML"     class="badge bg">ML: --</span>
    <span id="hClock"  class="badge bb">--:--:--</span>
    <button class="hbtn" onclick="toggleTheme()">THEME</button>
  </div>
</header>
<div class="container">
<div class="tabs">
  <div class="tab active" id="tb-dash"     onclick="showTab('dash')">// DASHBOARD</div>
  <div class="tab"        id="tb-history"  onclick="showTab('history')">// HISTORY</div>
  <div class="tab"        id="tb-settings" onclick="showTab('settings')">// SETTINGS</div>
</div>

<!-- DASHBOARD -->
<div class="pane active" id="pane-dash">
  <div id="statusBar" class="statusbar normal"><span id="statusTxt">SYSTEM NORMAL</span></div>
  <div class="g4">
    <div class="card"><h3>Accel X</h3><div class="bv" id="vAX">--</div></div>
    <div class="card"><h3>Accel Y</h3><div class="bv" id="vAY">--</div></div>
    <div class="card"><h3>Accel Z</h3><div class="bv" id="vAZ">--</div></div>
    <div class="card"><h3>Vibration Score</h3><div class="bv" id="vVIB">--</div></div>
    <div class="card"><h3>Session Min</h3><div class="bv" id="vMIN" style="color:#006600">--</div></div>
    <div class="card"><h3>Session Max</h3><div class="bv" id="vMAX" style="color:var(--red)">--</div></div>
    <div class="card"><h3>Session Avg</h3><div class="bv" id="vAVG" style="color:#996600">--</div></div>
    <div class="card"><h3>Wi-Fi RSSI</h3><div class="bv" id="vRSSI" style="font-size:1.1rem">--</div></div>
  </div>
  <div class="ml-panel">
    <div class="mph"><h2>// ML VIBRATION CLASSIFICATION</h2><span class="algo">Random Forest &middot; 40 features &middot; 50-sample windows &middot; 10s batch</span></div>
    <div class="ml-label" id="mlLabel">WAITING...</div>
    <div class="ml-sub"   id="mlConf">RECORDING 10S BATCH...</div>
    <div class="ml-bar-bg"><div class="ml-bar" id="mlBar" style="width:0%"></div></div>
    <div class="pills" id="mlPills"></div>
    <div class="algo-box">
      <b style="color:var(--yel)">HOW IT WORKS:</b> ESP32 records 330 samples (~10s at 33Hz).
      Each 50-sample window extracts 40 features (mean, std, peaks, zero-crossings, energy).
      A Random Forest votes across all windows &mdash; majority label is the final classification.<br>
      <b style="color:#f97316">CLASSES:</b> Gentle_Tap &bull; Hand_Tap &bull; Multi_Tap &bull; MultiHand_Tap &bull; Walk1
    </div>
  </div>
  <div class="wave-card">
    <div class="whead">
      <h3>// Live Signal Waveform</h3>
      <div style="display:flex;gap:9px;align-items:center;">
        <div class="leg">
          <span><span class="ld" style="background:#0057ff"></span>AX</span>
          <span><span class="ld" style="background:#00ff6a"></span>AY</span>
          <span><span class="ld" style="background:#ff2b2b"></span>AZ</span>
          <span><span class="ld" style="background:#ffe600"></span>VIB</span>
        </div>
        <button class="hbtn" onclick="togglePause()"><span id="pauseLbl">PAUSE</span></button>
      </div>
    </div>
    <canvas id="waveCanvas" height="180"></canvas>
  </div>
  <div class="sig-box">
    <h3>// Signal Character Analysis</h3>
    <div class="srow"><span class="sl">Energy</span><div class="sb-bg"><div class="sb" id="sbE" style="background:#0057ff;width:0%"></div></div><span class="sv" id="svE">--</span></div>
    <div class="srow"><span class="sl">Impulsiveness</span><div class="sb-bg"><div class="sb" id="sbI" style="background:#ff2b2b;width:0%"></div></div><span class="sv" id="svI">--</span></div>
    <div class="srow"><span class="sl">Smoothness</span><div class="sb-bg"><div class="sb" id="sbS" style="background:#00ff6a;width:0%"></div></div><span class="sv" id="svS">--</span></div>
    <div class="srow"><span class="sl">Linearity</span><div class="sb-bg"><div class="sb" id="sbL" style="background:#ffe600;width:0%"></div></div><span class="sv" id="svL">--</span></div>
    <div class="sig-note" id="sigNote">Collecting data...</div>
  </div>
  <div class="g2">
    <div class="card"><h3>// Vibration Gauge</h3><canvas id="gaugeCanvas" height="120"></canvas></div>
    <div class="card"><h3>// Tilt Direction</h3><canvas id="tiltCanvas" height="120"></canvas></div>
  </div>
  <div class="s-card">
    <h3>// System Information</h3>
    <div class="kv"><span class="k">ESP32 IP</span><span>%IP%</span></div>
    <div class="kv"><span class="k">Uptime</span><span id="siUp">--</span></div>
    <div class="kv"><span class="k">Threshold</span><span id="siThr">--</span></div>
    <div class="kv"><span class="k">MPU-6050</span><span id="siMPU">--</span></div>
    <div class="kv"><span class="k">SD Card</span><span id="siSD">--</span></div>
    <div class="kv"><span class="k">OLED</span><span id="siOLED">--</span></div>
    <div class="kv"><span class="k">ML Server</span><span id="siMLSrv">--</span></div>
    <div class="kv"><span class="k">Total Alerts</span><span id="siAlerts">0</span></div>
  </div>
</div>

<!-- HISTORY -->
<div class="pane" id="pane-history">
  <div class="s-card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:14px;">
      <h3 style="border-bottom:none;margin-bottom:0;padding-bottom:0;">// Vibration Event Log (SD Card)</h3>
      <div style="display:flex;gap:6px;flex-wrap:wrap;">
        <button class="hbtn" onclick="loadHistory()">REFRESH</button>
        <a href="/download" style="text-decoration:none"><button class="hbtn">DOWNLOAD CSV</button></a>
        <button class="hbtn" onclick="exportJSON()">EXPORT JSON</button>
        <button class="hbtn" style="background:var(--red);color:var(--wht);" onclick="clearLog()">CLEAR LOG</button>
      </div>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;align-items:center;margin-bottom:12px;font-size:.7rem;font-family:'Space Mono',monospace;text-transform:uppercase;letter-spacing:1px;color:#555;">
      Filter:
      <select id="fClass" onchange="applyFilter()" style="background:var(--wht);border:2px solid var(--blk);color:var(--blk);padding:4px 8px;font-size:.7rem;font-family:'Space Mono',monospace;">
        <option value="">All</option>
        <option>Gentle_Tap</option><option>Hand_Tap</option><option>Multi_Tap</option>
        <option>MultiHand_Tap</option><option>Walk1</option><option>ALERT</option><option>NORMAL</option>
      </select>
      Min vib:
      <input type="number" id="fMinVib" value="0" onchange="applyFilter()" style="background:var(--wht);border:2px solid var(--blk);color:var(--blk);padding:4px 7px;font-size:.7rem;width:70px;font-family:'Space Mono',monospace;">
    </div>
    <table class="htable">
      <thead><tr><th>Time</th><th>AX</th><th>AY</th><th>AZ</th><th>Vibration</th><th>Status</th><th>Observation</th></tr></thead>
      <tbody id="histTbody"><tr><td colspan="7" style="text-align:center;padding:18px;color:#888;font-family:'Space Mono',monospace;">// Click REFRESH to load SD card data</td></tr></tbody>
    </table>
    <div class="pg"><span id="histPg">Page 1/1</span><button class="hbtn" onclick="hPrev()">PREV</button><button class="hbtn" onclick="hNext()">NEXT</button></div>
  </div>
  <div class="wave-card">
    <div class="whead"><h3>// Vibration History Chart (last 100 events)</h3></div>
    <canvas id="histChart" height="150"></canvas>
  </div>
  <!-- LAST 10 BATCHES SUMMARY -->
  <div class="s-card">
    <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;margin-bottom:14px;">
      <h3 style="border-bottom:none;margin-bottom:0;padding-bottom:0;">// Last 10 Batch Summary</h3>
      <button class="hbtn" onclick="loadRecentBatches()">REFRESH</button>
    </div>
    <div id="batchStrip" style="display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px;"></div>
    <div id="batchBreakdown" style="font-size:.75rem;color:#555;font-family:'Space Mono',monospace;">// Click REFRESH to load.</div>
  </div>
</div>

<!-- SETTINGS -->
<div class="pane" id="pane-settings">
  <div class="g2">
    <div class="card sform" style="border:3px solid var(--blk);margin:0;">
      <h3 style="font-family:'Space Mono',monospace;font-size:.72rem;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">// Detection Threshold</h3>
      <p style="font-size:.7rem;color:#555;font-family:'Space Mono',monospace;">Vibration score that triggers ALERT. Resets on reboot.</p>
      <label>New value (200-20000)</label>
      <input type="number" id="threshInput" value="%THRESHOLD%" min="200" max="20000">
      <br><button class="sbtn" onclick="setThreshold()">UPDATE</button>
      <div class="msg" id="threshMsg"></div>
    </div>
    <div class="card sform" style="border:3px solid var(--blk);border-left:none;margin:0;">
      <h3 style="font-family:'Space Mono',monospace;font-size:.72rem;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">// Demo / Test Mode</h3>
      <p style="font-size:.7rem;color:#555;font-family:'Space Mono',monospace;">Triggers a fake ALERT for 5 seconds.</p>
      <button class="sbtn" onclick="runDemo()" style="margin-top:14px;">RUN DEMO ALERT</button>
      <h3 style="font-family:'Space Mono',monospace;font-size:.72rem;font-weight:800;letter-spacing:2px;text-transform:uppercase;margin-top:20px;margin-bottom:6px;">// SD Card Log</h3>
      <p style="font-size:.7rem;color:#555;font-family:'Space Mono',monospace;">Permanently deletes the log file from SD card.</p>
      <button class="sbtn red" onclick="clearLog()" style="margin-top:10px;">CLEAR LOG FILE</button>
      <div class="msg" id="sdMsg"></div>
    </div>
  </div>
  <div class="s-card" style="margin-top:20px;">
    <h3>// Threshold Change History</h3>
    <div id="threshHistory" style="font-size:.75rem;color:#555;font-family:'Space Mono',monospace;">// Load settings to see history.</div>
  </div>
  <div class="s-card">
    <h3>// ML Inference Log</h3>
    <div class="logbox" id="mlLog"></div>
  </div>
</div>
</div>

<script>
const ML_CLASSES=['Gentle_Tap','Hand_Tap','Multi_Tap','MultiHand_Tap','Walk1'];
const OBS_CLS={'Gentle_Tap':'o-gt','Hand_Tap':'o-ht','Multi_Tap':'o-mt','MultiHand_Tap':'o-mht','Walk1':'o-w','ALERT':'o-al','NORMAL':'o-nr'};
const MAX_PTS=80,POLL_MS=400;
let axH=[],ayH=[],azH=[],vibH=[],paused=false;
let sMin=Infinity,sMax=-Infinity,sSum=0,sN=0;
let darkMode=true,failStreak=0;
let allHist=[],filtHist=[],hPage=0;
const HPP=20;
let mlLogLines=[];

function toggleTheme(){
  darkMode=!darkMode;
  document.body.style.filter=darkMode?'':'invert(1) hue-rotate(180deg)';
}

function showTab(t){
  ['dash','history','settings'].forEach(id=>{
    document.getElementById('pane-'+id).classList.toggle('active',id===t);
    document.getElementById('tb-'+id).classList.toggle('active',id===t);
  });
  if(t==='history'){loadHistory();loadRecentBatches();}
  if(t==='settings') loadThreshHist();
}

function toast(msg,ok){
  const el=document.getElementById('toast');
  el.textContent=msg;
  el.style.background=ok?'#000':'#ff2b2b';
  el.style.color=ok?'#ffe600':'#fff';
  el.classList.add('show');
  setTimeout(()=>el.classList.remove('show'),3500);
}

function fmtUp(ms){const s=Math.floor(ms/1e3),h=Math.floor(s/3600),m=Math.floor((s%3600)/60),sc=s%60;return h+'h '+m+'m '+sc+'s';}

setInterval(()=>document.getElementById('hClock').textContent=new Date().toLocaleTimeString(),1000);

(function initPills(){
  const c=document.getElementById('mlPills');
  ML_CLASSES.forEach(cls=>{const d=document.createElement('div');d.className='pill';d.id='pill-'+cls;d.textContent=cls;c.appendChild(d);});
})();

function updateML(label,conf,ready,serverOK){
  const lEl=document.getElementById('mlLabel'),cEl=document.getElementById('mlConf'),bar=document.getElementById('mlBar');
  const hML=document.getElementById('hML');
  if(ready){
    lEl.textContent=label; lEl.style.color='var(--green)';
    cEl.textContent='Confidence: '+Math.round(conf*100)+'%  (10s batch)';
    bar.style.width=Math.round(conf*100)+'%';
    hML.textContent='ML: '+label; hML.className='badge bg';
    ML_CLASSES.forEach(c=>{const el=document.getElementById('pill-'+c);if(el)el.classList.toggle('hot',c===label);});
    addMLLog(label,conf);
  } else {
    lEl.textContent=label||'Recording...'; lEl.style.color='var(--muted)';
    cEl.textContent='Recording 10s batch...'; bar.style.width='0%';
    hML.textContent='ML: '+label; hML.className='badge bb';
    ML_CLASSES.forEach(c=>{const el=document.getElementById('pill-'+c);if(el)el.classList.remove('hot');});
  }
}

function addMLLog(label,conf){
  const ts=new Date().toLocaleTimeString();
  mlLogLines.unshift('<span class="ts">['+ts+']</span> <span class="lb">'+label+'</span> conf='+Math.round(conf*100)+'%');
  if(mlLogLines.length>60) mlLogLines.pop();
  const box=document.getElementById('mlLog');
  if(box) box.innerHTML=mlLogLines.map(l=>'<div class="le">'+l+'</div>').join('');
}

function updateSig(vib,thr){
  if(vibH.length<5) return;
  const arr=vibH.slice(-30);
  const mean=arr.reduce((a,b)=>a+b,0)/arr.length;
  const maxV=Math.max(...arr,1);
  const energy=Math.min(mean/Math.max(thr,1),1);
  const impulse=maxV>0?Math.min((maxV-mean)/maxV,1):0;
  const std=Math.sqrt(arr.reduce((a,v)=>a+(v-mean)**2,0)/arr.length);
  const smooth=Math.max(0,1-std/maxV);
  const n=arr.length,sumX=n*(n-1)/2,sumY=arr.reduce((a,b)=>a+b,0);
  const sumXY=arr.reduce((a,v,i)=>a+i*v,0),sumX2=n*(n-1)*(2*n-1)/6;
  const slope=(n*sumXY-sumX*sumY)/(n*sumX2-sumX*sumX+1e-6);
  const intercept=(sumY-slope*sumX)/n;
  const range=Math.max(maxV-Math.min(...arr),1);
  const resid=arr.reduce((a,v,i)=>a+Math.abs(v-(slope*i+intercept)),0)/(n*range);
  const linear=Math.max(0,Math.min(1,1-resid));
  function sb(id,val,txt){document.getElementById('sb'+id).style.width=Math.round(val*100)+'%';document.getElementById('sv'+id).textContent=txt;}
  sb('E',energy,Math.round(energy*100)+'%');
  sb('I',impulse,impulse>0.65?'Spiky':'Smooth');
  sb('S',smooth,Math.round(smooth*100)+'%');
  sb('L',linear,Math.round(linear*100)+'%');
  let note='Collecting...';
  if(energy<0.08) note='Near-baseline. Very little activity.';
  else if(impulse>0.7) note='High impulsiveness -- sharp impact (tap-like). Non-linear signal.';
  else if(smooth>0.7&&linear>0.5) note='Smooth linear signal -- sustained vibration (walk-like).';
  else if(linear<0.3) note='Non-linear/chaotic pattern -- erratic multi-source vibration.';
  else note='Mixed pattern -- moderate irregular vibration.';
  document.getElementById('sigNote').textContent=note;
}

let wasAlert=false;
function pollData(){
  fetch('/data').then(r=>r.json()).then(d=>{
    failStreak=0; document.getElementById('connWarn').classList.remove('show');
    document.getElementById('vAX').textContent=d.ax;
    document.getElementById('vAY').textContent=d.ay;
    document.getElementById('vAZ').textContent=d.az;
    document.getElementById('vVIB').textContent=d.vibration;
    document.getElementById('vRSSI').textContent=d.rssi+' dBm';
    sN++; sSum+=d.vibration;
    if(d.vibration<sMin) sMin=d.vibration;
    if(d.vibration>sMax) sMax=d.vibration;
    document.getElementById('vMIN').textContent=sMin===Infinity?'--':sMin;
    document.getElementById('vMAX').textContent=sMax===-Infinity?'--':sMax;
    document.getElementById('vAVG').textContent=sN?Math.round(sSum/sN):'--';
    const isAlert=d.earthquake||!d.mpuOK;
    document.getElementById('statusBar').className='statusbar '+(isAlert?'danger':'normal');
    document.getElementById('statusTxt').textContent=d.status||(isAlert?'VIBRATION ALERT':'SYSTEM NORMAL');
    document.getElementById('hAlerts').textContent='Alerts: '+d.alertCount;
    document.getElementById('hAlerts').className='badge '+(d.alertCount>0?'br':'bb');
    document.getElementById('hSD').textContent='SD: '+(d.sdOK?'OK':'FAIL');
    document.getElementById('hSD').className='badge '+(d.sdOK?'bg':'br');
    document.getElementById('siUp').textContent=fmtUp(d.uptime);
    document.getElementById('siThr').textContent=d.threshold;
    document.getElementById('siMPU').textContent=d.mpuOK?'Connected':'ERROR';
    document.getElementById('siSD').textContent=d.sdOK?'Connected (writing)':'Not found';
    document.getElementById('siOLED').textContent=d.oledOK?'Connected':'Not found';
    document.getElementById('siMLSrv').textContent=d.mlServerOK?'Online':'Offline';
    document.getElementById('siAlerts').textContent=d.alertCount;
    if(d.mlLabel!==undefined) updateML(d.mlLabel,d.mlConfidence,d.mlReady,d.mlServerOK);
    if(!paused){
      axH.push(d.ax);ayH.push(d.ay);azH.push(d.az);vibH.push(d.vibration);
      if(axH.length>MAX_PTS){axH.shift();ayH.shift();azH.shift();vibH.shift();}
      drawWave(); drawGauge(d.vibration,d.threshold); drawTilt(d.ax,d.ay);
    }
    updateSig(d.vibration,d.threshold);
    if(isAlert&&!wasAlert) playBeep();
    wasAlert=isAlert;
  }).catch(()=>{if(++failStreak>=4) document.getElementById('connWarn').classList.add('show');});
}
setInterval(pollData,POLL_MS); pollData();

function drawWave(){
  const cv=document.getElementById('waveCanvas');if(!cv)return;
  const dpr=devicePixelRatio||1,W=cv.clientWidth,H=180;
  cv.width=W*dpr;cv.height=H*dpr;
  const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,W,H);
  ctx.strokeStyle='#ddd';ctx.lineWidth=1;
  for(let i=0;i<=4;i++){const y=(i/4)*H;ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();}
  ctx.strokeStyle='#bbb';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(0,H/2);ctx.lineTo(W,H/2);ctx.stroke();
  const n=axH.length;if(n<2)return;
  const peak=Math.max(...[...axH,...ayH,...azH].map(Math.abs),500);
  function line(data,col,lw){
    ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.beginPath();
    data.forEach((v,i)=>{const x=(i/(MAX_PTS-1))*W,y=H/2-(v/peak)*(H/2-8);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
    ctx.stroke();
  }
  line(axH,'#0057ff',1.4);line(ayH,'#00cc55',1.4);line(azH,'#ff2b2b',1.4);
  if(vibH.length>1){
    const vmax=Math.max(...vibH,1);
    ctx.strokeStyle='#d29922';ctx.lineWidth=2;ctx.beginPath();
    vibH.forEach((v,i)=>{const x=(i/(MAX_PTS-1))*W,y=H-(v/vmax)*(H*0.3);i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);});
    ctx.stroke();
    ctx.lineTo((vibH.length-1)/(MAX_PTS-1)*W,H);ctx.lineTo(0,H);ctx.closePath();
    ctx.fillStyle='rgba(210,153,34,0.15)';ctx.fill();
  }
}

function drawGauge(vib,thr){
  const cv=document.getElementById('gaugeCanvas');if(!cv)return;
  const dpr=devicePixelRatio||1,W=cv.clientWidth||200,H=120;
  cv.width=W*dpr;cv.height=H*dpr;
  const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,W,H);
  const cx=W/2,cy=H-6,r=Math.min(W/2-10,H-14);
  const ratio=Math.min(vib/Math.max(thr*1.5,1),1);
  ctx.lineWidth=13;ctx.lineCap='round';
  ctx.strokeStyle='#ddd';ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,2*Math.PI);ctx.stroke();
  const g=ctx.createLinearGradient(0,0,W,0);
  g.addColorStop(0,'#3fb950');g.addColorStop(.6,'#d29922');g.addColorStop(1,'#f85149');
  ctx.strokeStyle=g;ctx.beginPath();ctx.arc(cx,cy,r,Math.PI,Math.PI+ratio*Math.PI);ctx.stroke();
  const a=Math.PI+ratio*Math.PI,nx=cx+Math.cos(a)*(r-16),ny=cy+Math.sin(a)*(r-16);
  ctx.strokeStyle='#333';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(nx,ny);ctx.stroke();
  ctx.fillStyle='#333';ctx.font='bold 14px Space Mono,monospace';ctx.textAlign='center';ctx.fillText(vib,cx,cy-r/2+4);
}

function drawTilt(axv,ayv){
  const cv=document.getElementById('tiltCanvas');if(!cv)return;
  const dpr=devicePixelRatio||1,W=cv.clientWidth||200,H=120;
  cv.width=W*dpr;cv.height=H*dpr;
  const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,W,H);
  const cx=W/2,cy=H/2,r=Math.min(W,H)/2-8;
  [r,r*0.66,r*0.33].forEach(rr=>{ctx.strokeStyle='#ddd';ctx.lineWidth=1;ctx.beginPath();ctx.arc(cx,cy,rr,0,2*Math.PI);ctx.stroke();});
  const angle=Math.atan2(ayv,axv),mag=Math.min(Math.sqrt(axv*axv+ayv*ayv)/8000,1);
  const nx=cx+Math.cos(angle)*r*mag,ny=cy+Math.sin(angle)*r*mag;
  ctx.strokeStyle='#0057ff';ctx.lineWidth=2.5;ctx.beginPath();ctx.moveTo(cx,cy);ctx.lineTo(nx,ny);ctx.stroke();
  ctx.fillStyle='#0057ff';ctx.beginPath();ctx.arc(nx,ny,5,0,2*Math.PI);ctx.fill();
  ctx.fillStyle='#000';ctx.beginPath();ctx.arc(cx,cy,3,0,2*Math.PI);ctx.fill();
}

function obsLabel(r){if(r.mlLabel&&r.mlLabel.length>2&&r.mlLabel!=='--'&&r.mlLabel!=='Recording...'&&r.mlLabel!=='Analyzing...')return r.mlLabel;return r.status==='ALERT'?'ALERT':'NORMAL';}

function loadHistory(){
  const tb=document.getElementById('histTbody');
  tb.innerHTML='<tr><td colspan="7" style="text-align:center;padding:18px;color:var(--muted)">Loading...</td></tr>';
  fetch('/events').then(r=>r.json()).then(d=>{allHist=d.events||[];hPage=0;applyFilter();drawHistChart();})
  .catch(()=>{tb.innerHTML='<tr><td colspan="7" style="text-align:center;padding:18px;color:var(--red)">Failed. Is SD card connected?</td></tr>';});
}

function applyFilter(){
  const cls=document.getElementById('fClass').value,minV=parseInt(document.getElementById('fMinVib').value)||0;
  filtHist=allHist.filter(r=>{const ob=obsLabel(r);if(cls&&ob!==cls)return false;if(r.vibration<minV)return false;return true;});
  hPage=0;renderHist();
}

function renderHist(){
  const tb=document.getElementById('histTbody'),total=Math.max(1,Math.ceil(filtHist.length/HPP));
  if(hPage>=total)hPage=total-1;
  const sl=filtHist.slice(hPage*HPP,(hPage+1)*HPP);
  document.getElementById('histPg').textContent='Page '+(hPage+1)+'/'+total+' ('+filtHist.length+')';
  if(!sl.length){tb.innerHTML='<tr><td colspan="7" style="text-align:center;padding:18px;color:var(--muted)">No matching records.</td></tr>';return;}
  tb.innerHTML=sl.map(r=>{const ob=obsLabel(r),oc=OBS_CLS[ob]||'o-uk';
    return '<tr><td>'+fmtUp(r.time)+'</td><td>'+r.ax+'</td><td>'+r.ay+'</td><td>'+r.az+'</td>'
      +'<td>'+r.vibration+'</td>'
      +'<td><span class="obs '+(r.status==='ALERT'?'o-al':'o-nr')+'">'+r.status+'</span></td>'
      +'<td><span class="obs '+oc+'">'+ob+'</span></td></tr>';}).join('');
}
function hPrev(){if(hPage>0){hPage--;renderHist();}}
function hNext(){hPage++;renderHist();}

function drawHistChart(){
  const cv=document.getElementById('histChart');if(!cv)return;
  const dpr=devicePixelRatio||1,W=cv.clientWidth||600,H=150;
  cv.width=W*dpr;cv.height=H*dpr;
  const ctx=cv.getContext('2d');ctx.scale(dpr,dpr);
  ctx.clearRect(0,0,W,H);
  const data=allHist.slice(-100);if(data.length<2)return;
  const maxV=Math.max(...data.map(r=>r.vibration),1),bw=W/data.length;
  data.forEach((r,i)=>{const h=(r.vibration/maxV)*(H-8);ctx.fillStyle=r.status==='ALERT'?'#ff2b2b':'#00cc55';ctx.fillRect(i*bw+1,H-h,bw-2,h);});
}

// -- LAST 10 BATCHES SUMMARY -----------------------------------
const BATCH_COLORS={
  'Gentle_Tap':'#006600','Hand_Tap':'#0057ff','Multi_Tap':'#6600cc',
  'MultiHand_Tap':'#996600','Walk1':'#cc0066','ALERT':'#ff2b2b','NORMAL':'#006600','Unknown':'#555'
};

function loadRecentBatches(){
  fetch("/recentbatches").then(r=>r.json()).then(d=>{
    const batches=d.batches||[];
    const strip=document.getElementById("batchStrip");
    const breakdown=document.getElementById("batchBreakdown");

    if(!batches.length){
      strip.innerHTML="";
      breakdown.textContent="No batches completed yet. Run the system for 10 seconds.";
      return;
    }

    // -- Timeline strip: one pill per batch --------------------
    strip.innerHTML=batches.map((b,i)=>{
      const col=BATCH_COLORS[b.label]||"#555";
      return `<div style="background:#fff;border:3px solid ${col};padding:5px 10px;font-size:.7rem;font-weight:700;color:${col};text-align:center;min-width:70px;font-family:'Space Mono',monospace;text-transform:uppercase;box-shadow:2px 2px 0 ${col};">
        <div style="font-size:.58rem;color:#aaa;margin-bottom:2px;">BATCH ${i+1}</div>
        ${b.label}<br><span style="color:#888;font-weight:400;">${Math.round(b.conf*100)}%</span>
      </div>`;
    }).join("");

    // -- Count occurrences -------------------------------------
    const counts={};
    batches.forEach(b=>{counts[b.label]=(counts[b.label]||0)+1;});
    const total=batches.length;
    const sorted=Object.entries(counts).sort((a,b)=>b[1]-a[1]);

    // -- Breakdown bars ----------------------------------------
    breakdown.innerHTML=`
      <div style="margin-bottom:8px;font-size:.72rem;color:#888;">${total} batch${total>1?"es":""} recorded &mdash; ${new Set(batches.map(b=>b.label)).size} unique vibration type${new Set(batches.map(b=>b.label)).size>1?"s":""}</div>
    `+sorted.map(([lbl,cnt])=>{
      const col=BATCH_COLORS[lbl]||"#555";
      const pct=Math.round(cnt/total*100);
      return `<div style="display:flex;align-items:center;gap:10px;margin:6px 0;">
        <div style="width:120px;font-size:.7rem;color:${col};font-weight:700;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-family:'Space Mono',monospace;text-transform:uppercase;">${lbl}</div>
        <div style="flex:1;background:#eee;border:2px solid #000;height:12px;overflow:hidden;">
          <div style="height:12px;background:${col};width:${pct}%;transition:width .4s;"></div>
        </div>
        <div style="width:26px;text-align:right;font-size:.72rem;font-weight:700;font-family:'Space Mono',monospace;">${cnt}</div>
        <div style="width:34px;text-align:right;font-size:.7rem;color:#888;font-family:'Space Mono',monospace;">${pct}%</div>
      </div>`;
    }).join("");
  }).catch(()=>{
    document.getElementById("batchBreakdown").textContent="Failed to load. Flash firmware and check connection.";
  });
}

function exportJSON(){const b=new Blob([JSON.stringify(allHist,null,2)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='gvms_history.json';a.click();}

function clearLog(){
  if(!confirm('Delete SD card log? Cannot be undone.'))return;
  fetch('/clearLog').then(r=>r.json()).then(d=>{
    document.getElementById('sdMsg').textContent=d.success?'Log cleared.':'Failed.';
    document.getElementById('sdMsg').className='msg '+(d.success?'ok':'err');
    if(d.success){allHist=[];filtHist=[];renderHist();}
    toast(d.success?'Log cleared':'Clear failed',d.success);
  });
}

function setThreshold(){
  const v=document.getElementById('threshInput').value;
  fetch('/setThreshold?value='+v).then(r=>r.json()).then(d=>{
    const m=document.getElementById('threshMsg');
    m.textContent=d.success?'Updated to '+d.threshold:'Invalid (200-20000).';
    m.className='msg '+(d.success?'ok':'err');
    if(d.success)loadThreshHist();
  });
}

function loadThreshHist(){
  fetch('/thresholdHistory').then(r=>r.json()).then(d=>{
    const el=document.getElementById('threshHistory');
    if(!d.history||!d.history.length){el.textContent='No changes this session.';return;}
    el.innerHTML=d.history.slice().reverse().map(h=>'<div class="kv"><span class="k">'+fmtUp(h.time)+'</span><span>'+h.value+'</span></div>').join('');
  });
}

function runDemo(){fetch('/simulate').then(r=>r.json()).then(()=>toast('Demo alert active for 5s',false));}
function togglePause(){paused=!paused;document.getElementById('pauseLbl').textContent=paused?'Resume':'Pause';}
function playBeep(){
  try{const ac=new(window.AudioContext||window.webkitAudioContext)();
    [0,280,560].forEach(d=>setTimeout(()=>{const o=ac.createOscillator(),g=ac.createGain();o.connect(g);g.connect(ac.destination);o.type='square';o.frequency.value=880;g.gain.setValueAtTime(0.22,ac.currentTime);o.start();o.stop(ac.currentTime+0.18);},d));
  }catch(e){}
}
</script>
</body>
</html>
)rawliteral";

  html.replace("%THRESHOLD%", String(EARTHQUAKE_THRESHOLD));
  html.replace("%IP%", WiFi.localIP().toString());
  return html;
}

void handleData() {

  String statusText;

  if (!mpuOK) statusText = "MPU6050 ERROR";
  else if (earthquake) statusText = "VIBRATION ALERT DETECTED";
  else statusText = "SYSTEM NORMAL";

  String json = "{";
  json += "\"ax\":"           + String(ax)                      + ",";
  json += "\"ay\":"           + String(ay)                      + ",";
  json += "\"az\":"           + String(az)                      + ",";
  json += "\"vibration\":"    + String(vibration)               + ",";
  json += "\"earthquake\":"   + String(earthquake ? "true" : "false") + ",";
  json += "\"mpuOK\":"        + String(mpuOK   ? "true" : "false") + ",";
  json += "\"sdOK\":"         + String(sdOK    ? "true" : "false") + ",";
  json += "\"oledOK\":"       + String(oledOK  ? "true" : "false") + ",";
  json += "\"threshold\":"    + String(EARTHQUAKE_THRESHOLD)    + ",";
  json += "\"rssi\":"         + String(WiFi.RSSI())             + ",";
  json += "\"uptime\":"       + String(millis())                + ",";
  json += "\"alertCount\":"   + String(sessionAlertCount)       + ",";
  json += "\"demo\":"         + String(demoMode ? "true" : "false") + ",";
  // ML fields (NEW)
  json += "\"mlLabel\":\""    + mlLabel                         + "\",";
  json += "\"mlConfidence\":" + String(mlConfidence, 3)         + ",";
  json += "\"mlReady\":"      + String(mlReady    ? "true" : "false") + ",";
  json += "\"mlServerOK\":"   + String(mlServerOK ? "true" : "false") + ",";
  json += "\"status\":\""     + statusText                      + "\"";
  json += "}";

  server.send(200, "application/json", json);
}

// =====================================================
// SET THRESHOLD
// =====================================================

void recordThresholdChange(long value) {

  if (thresholdHistoryCount < MAX_THRESHOLD_HISTORY) {
    thresholdHistory[thresholdHistoryCount].value = value;
    thresholdHistory[thresholdHistoryCount].time = millis();
    thresholdHistoryCount++;
  } else {
    for (int i = 0; i < MAX_THRESHOLD_HISTORY - 1; i++) {
      thresholdHistory[i] = thresholdHistory[i + 1];
    }
    thresholdHistory[MAX_THRESHOLD_HISTORY - 1].value = value;
    thresholdHistory[MAX_THRESHOLD_HISTORY - 1].time = millis();
  }
}

void handleSetThreshold() {

  if (server.hasArg("value")) {

    long newThreshold = server.arg("value").toInt();

    if (newThreshold >= MIN_THRESHOLD && newThreshold <= MAX_THRESHOLD) {

      EARTHQUAKE_THRESHOLD = newThreshold;
      recordThresholdChange(EARTHQUAKE_THRESHOLD);

      Serial.print("Threshold updated: ");
      Serial.println(EARTHQUAKE_THRESHOLD);

      String json = "{\"success\":true,\"threshold\":" + String(EARTHQUAKE_THRESHOLD) + "}";
      server.send(200, "application/json", json);
      return;
    }
  }

  server.send(400, "application/json", "{\"success\":false}");
}

void handleThresholdHistory() {

  String json = "{\"history\":[";

  for (int i = 0; i < thresholdHistoryCount; i++) {
    json += "{\"value\":" + String(thresholdHistory[i].value) + ",\"time\":" + String(thresholdHistory[i].time) + "}";
    if (i < thresholdHistoryCount - 1) json += ",";
  }

  json += "]}";

  server.send(200, "application/json", json);
}

// =====================================================
// EVENT HISTORY
// Filter "ALERT" (matches CSV Status column written by saveData)
// =====================================================

void handleEvents() {

  if (!sdOK) {
    server.send(200, "application/json", "{\"events\":[]}");
    return;
  }

  File file = SD.open("/earthquake.csv", FILE_READ);

  if (!file) {
    server.send(200, "application/json", "{\"events\":[]}");
    return;
  }

  String eventLines[MAX_EVENTS];
  int count = 0;

  if (file.available()) {
    file.readStringUntil('\n');  // skip header
  }

  while (file.available()) {

    String line = file.readStringUntil('\n');
    line.trim();

    if (line.length() == 0) continue;

    if (line.indexOf("ALERT") >= 0) {   // CSV writes "ALERT"

      if (count < MAX_EVENTS) {
        eventLines[count] = line;
        count++;
      } else {
        for (int i = 0; i < MAX_EVENTS - 1; i++) {
          eventLines[i] = eventLines[i + 1];
        }
        eventLines[MAX_EVENTS - 1] = line;
      }
    }
  }

  file.close();

  String json = "{\"events\":[";

  for (int i = 0; i < count; i++) {

    String line = eventLines[i];

    int p1 = line.indexOf(',');
    int p2 = line.indexOf(',', p1 + 1);
    int p3 = line.indexOf(',', p2 + 1);
    int p4 = line.indexOf(',', p3 + 1);
    int p5 = line.indexOf(',', p4 + 1);

    if (p1 < 0 || p2 < 0 || p3 < 0 || p4 < 0) continue;

    String t    = line.substring(0, p1);
    String vax  = line.substring(p1 + 1, p2);
    String vay  = line.substring(p2 + 1, p3);
    String vaz  = line.substring(p3 + 1, p4);
    String vvib = (p5 > 0) ? line.substring(p4 + 1, p5) : line.substring(p4 + 1);

    json += "{\"time\":"      + t    +
            ",\"ax\":"        + vax  +
            ",\"ay\":"        + vay  +
            ",\"az\":"        + vaz  +
            ",\"vibration\":" + vvib +
            ",\"mlLabel\":\"" + mlLabel + "\"}";

    if (i < count - 1) json += ",";
  }

  json += "]}";

  server.send(200, "application/json", json);
}

// =====================================================
// DOWNLOAD / CLEAR LOG
// =====================================================

void handleDownload() {

  if (!sdOK) {
    server.send(404, "text/plain", "SD card not available");
    return;
  }

  File file = SD.open("/earthquake.csv", FILE_READ);

  if (!file) {
    server.send(404, "text/plain", "Log file not found");
    return;
  }

  server.sendHeader("Content-Disposition", "attachment; filename=gvms_log.csv");
  server.streamFile(file, "text/csv");
  file.close();
}

// =====================================================
// RECENT BATCHES API  -- last 10 batch results
// =====================================================

void handleRecentBatches() {
  String json = "{\"batches\":[";
  // Walk from oldest to newest
  int start = (recentBatchCount < RECENT_BATCH_MAX)
              ? 0
              : recentBatchHead;  // oldest slot when buffer full
  for (int i = 0; i < recentBatchCount; i++) {
    int idx = (start + i) % RECENT_BATCH_MAX;
    json += "{\"label\":\"" + recentBatchLabels[idx] + "\"";
    json += ",\"conf\":"  + String(recentBatchConf[idx], 2) + "}";
    if (i < recentBatchCount - 1) json += ",";
  }
  json += "],\"count\":" + String(recentBatchCount) + "}";
  server.send(200, "application/json", json);
}

void handleClearLog() {

  if (!sdOK) {
    server.send(200, "application/json", "{\"success\":false}");
    return;
  }

  SD.remove("/earthquake.csv");
  createCSV();
  sessionAlertCount = 0;

  server.send(200, "application/json", "{\"success\":true}");
}

// =====================================================
// DEMO / TEST MODE
// =====================================================

void handleSimulate() {

  demoMode = true;
  demoModeUntil = millis() + 5000;

  server.send(200, "application/json", "{\"success\":true}");
}

// =====================================================
// HOME
// =====================================================

void handleRoot() {
  server.send(200, "text/html", webpage());
}

// =====================================================
// SETUP
// =====================================================

void setup() {

  Serial.begin(115200);
  delay(1000);

  Serial.println();
  Serial.println("================================");
  Serial.println(" GROUND VIBRATION MONITOR (GVMS)");
  Serial.println("================================");

  // OUTPUT PINS

  pinMode(BUZZER_PIN, OUTPUT);
  pinMode(GREEN_LED, OUTPUT);
  pinMode(RED_LED, OUTPUT);

  digitalWrite(BUZZER_PIN, LOW);
  digitalWrite(GREEN_LED, LOW);
  digitalWrite(RED_LED, LOW);

  // I2C

  Serial.println();
  Serial.println("Starting I2C...");
  Wire.begin(SDA_PIN, SCL_PIN);
  delay(300);
  Serial.println("I2C started: SDA=21 SCL=22");

  // OLED

  Serial.println();
  Serial.println("Starting OLED...");

  if (display.begin(SSD1306_SWITCHCAPVCC, OLED_ADDRESS)) {

    oledOK = true;

    display.clearDisplay();
    display.setTextColor(SSD1306_WHITE);
    display.setTextSize(1);
    display.setCursor(10, 10);
    display.println("GROUND VIBRATION");
    display.setCursor(20, 28);
    display.println("MONITOR (GVMS)");
    display.setCursor(10, 46);
    display.println("Starting...");
    display.display();

    delay(2000);

    Serial.println("OLED CONNECTED!");

  } else {
    Serial.println("OLED FAILED!");
  }

  // MPU6050

  setupMPU();

  // SD CARD

  Serial.println();
  Serial.println("Starting SD Card...");

  SPI.begin(SD_SCK, SD_MISO, SD_MOSI, SD_CS);
  delay(300);

  if (SD.begin(SD_CS, SPI)) {

    sdOK = true;

    Serial.println("SD CARD OK!");
    Serial.print("SD Size: ");
    Serial.print(SD.cardSize() / (1024 * 1024));
    Serial.println(" MB");
    Serial.print("Used Space: ");
    Serial.print(SD.usedBytes() / (1024 * 1024));
    Serial.println(" MB");

    createCSV();

  } else {
    sdOK = false;
    Serial.println("SD CARD FAILED!");
  }

  // WIFI

  Serial.println();
  Serial.println("Connecting WiFi...");

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting");

  int attempts = 0;

  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  Serial.println();

  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("WiFi Connected!");
    // Start mDNS -- ESP32 will resolve "gvms.local" without needing a fixed IP
    if (MDNS.begin("esp32-gvms")) {
      Serial.println("mDNS started: esp32-gvms.local");
    }
    MDNS.addService("http", "tcp", 80);
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("ML Server:  http://");
    Serial.print(ML_SERVER_IP_FALLBACK);
    Serial.print(":");
    Serial.println(ML_SERVER_PORT);
  } else {
    Serial.println("WiFi FAILED! (ML inference disabled)");
  }

  // WEB SERVER

  // Start ML HTTP task on Core 0 (sensor loop runs on Core 1)
  xTaskCreatePinnedToCore(
    mlHttpTask,    // function
    "mlHttpTask",  // name
    16384,         // stack size (bytes) ? needs room for JSON + HTTPClient
    NULL,          // parameter
    1,             // priority
    &mlTaskHandle, // handle
    0              // Core 0
  );
  Serial.println("ML task: started on Core 0");
  batchStartMs  = millis();
  batchRecording = true;
  Serial.println("ML batch: recording started (10s batches)");

  server.on("/", handleRoot);
  server.on("/data", handleData);
  server.on("/setThreshold", handleSetThreshold);
  server.on("/thresholdHistory", handleThresholdHistory);
  server.on("/events", handleEvents);
  server.on("/download", HTTP_GET, handleDownload);
  server.on("/recentbatches",   handleRecentBatches);
  server.on("/clearLog", handleClearLog);
  server.on("/simulate", handleSimulate);

  server.begin();

  Serial.println("Web Server Started!");

  if (mpuOK) {
    digitalWrite(GREEN_LED, HIGH);
  }

  Serial.println();
  Serial.println("================================");
  Serial.println(" SYSTEM READY");
  Serial.println("================================");
  Serial.println();
  Serial.println("Open the IP address in your browser.");
}

// =====================================================
// LOOP
// =====================================================

void loop() {

  server.handleClient();

  unsigned long currentMillis = millis();

  // Sensor at 100 Hz; ML buffer fills at ~33 Hz via ML_DOWNSAMPLE
  if (currentMillis - lastSensorRead >= SENSOR_INTERVAL) {
    lastSensorRead = currentMillis;
    readSensor();
  }

  if (currentMillis - lastLog >= LOG_INTERVAL) {
    lastLog = currentMillis;
    saveData();
  }
}

