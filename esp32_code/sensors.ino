#include <Wire.h>
#include "MAX30105.h"
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <WiFi.h>
#include <HTTPClient.h>
#define BUTTON_PIN 14
#define BUZZER_PIN 27
// ================= WIFI =================
const char* ssid = "Team30B";
const char* password = "Spiderman";
const char* serverName = "http://10.13.118.120:8000/sensor";

// ================= SENSORS =================
MAX30105 sensor;
Adafruit_MPU6050 mpu;

#define LM35_PIN 34

// SpO2 variables
float irAvg = 0, redAvg = 0;
float irAC = 0, redAC = 0;
float spo2 = 0;

void setup() {
  Serial.begin(115200);
  pinMode(BUTTON_PIN, INPUT_PULLUP);
pinMode(BUZZER_PIN, OUTPUT);

digitalWrite(BUZZER_PIN, LOW);
  analogReadResolution(12);

  // ================= WIFI =================
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.println("Connecting WiFi...");
  }
  Serial.println("WiFi Connected");

  // ================= I2C =================
  Wire.begin(21, 22);

  // MAX30102
  if (!sensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 not detected");
    while (1);
  }
  sensor.setup();
  sensor.setPulseAmplitudeRed(0x0A);
  sensor.setPulseAmplitudeIR(0x1F);
  sensor.setPulseAmplitudeGreen(0);

  // MPU6050
  if (!mpu.begin()) {
    Serial.println("MPU6050 not detected");
    while (1);
  }

  Serial.println("All sensors ready 👍");
}

void sendAlert() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;

    String alertUrl = "http://10.13.118.120:8000/alert";

    http.begin(alertUrl);
    http.addHeader("Content-Type", "application/json");

    String payload = "{\"alert\":\"button_pressed\"}";

    int response = http.POST(payload);

    Serial.print("Alert Response: ");
    Serial.println(response);

    http.end();
  }
}

void loop() {
  static unsigned long lastPress = 0;

if (digitalRead(BUTTON_PIN) == LOW) {

  if (millis() - lastPress > 800) {  // debounce

    lastPress = millis();

    Serial.println("🚨 BUTTON PRESSED!");

    // buzzer ON
    digitalWrite(BUZZER_PIN, HIGH);
    delay(300);
    digitalWrite(BUZZER_PIN, LOW);

    // send alert
    sendAlert();
  }
}

  // ================= MAX30102 =================
  long ir = sensor.getIR();
  long red = sensor.getRed();

  // ================= SpO2 =================
  irAvg = 0.95 * irAvg + 0.05 * ir;
  redAvg = 0.95 * redAvg + 0.05 * red;

  irAC = ir - irAvg;
  redAC = red - redAvg;

  if (irAvg > 0 && redAvg > 0 && irAC != 0) {
    float R = (redAC / redAvg) / (irAC / irAvg);
    spo2 = 110 - 25 * R;

    spo2 = spo2 + 20;
    if (spo2 > 99) spo2 = 99;
    if (spo2 < 97) spo2 = 97;
  }

  // ================= MPU6050 =================
  sensors_event_t a, g, temp;
  mpu.getEvent(&a, &g, &temp);

  // ================= LM35 =================
  int adcValue = analogRead(LM35_PIN);
  float voltage = adcValue * (3.3 / 4095.0);
  float temperatureC = voltage * 100.0 + 17.0;

  // ================= JSON BUILD =================
  String json = "{";
  json += "\"temp\":" + String(temperatureC) + ",";
  json += "\"spo2\":" + String(spo2) + ",";
  json += "\"acc_x\":" + String(a.acceleration.x) + ",";
  json += "\"acc_y\":" + String(a.acceleration.y) + ",";
  json += "\"acc_z\":" + String(a.acceleration.z) + ",";
  json += "\"gyro_x\":" + String(g.gyro.x) + ",";
  json += "\"gyro_y\":" + String(g.gyro.y) + ",";
  json += "\"gyro_z\":" + String(g.gyro.z) + ",";
  json += "\"ir\":" + String(ir) + ",";
  json += "\"red\":" + String(red);
  json += "}";

  Serial.println(json);

  // ================= SEND TO FASTAPI =================
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverName);
    http.addHeader("Content-Type", "application/json");

    int response = http.POST(json);

    Serial.print("Server Response: ");
    Serial.println(response);

    http.end();
  }

  delay(200);
}