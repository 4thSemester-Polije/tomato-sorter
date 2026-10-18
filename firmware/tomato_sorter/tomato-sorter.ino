#include <ESP8266WiFi.h>
#include <SPI.h>
#include <MFRC522.h>
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <Firebase_ESP_Client.h>

#include "addons/RTDBHelper.h"

// --- 1. KONFIGURASI WIFI ---
#define WIFI_SSID "Teknologi Informasi"
#define WIFI_PASSWORD ""

// --- 2. KONFIGURASI FIREBASE ---
#define DATABASE_URL "https://rfid-39416-default-rtdb.asia-southeast1.firebasedatabase.app/"
#define DATABASE_SECRET "iIz6v3VeuSqPHiUYl3PkrymWmmZAOoUlhWdByQLC"

// --- 3. DEFINISI PIN RFID (WEMOS D1 MINI) ---
#define SS_PIN  D8  // Pin SDA RFID
#define RST_PIN D4  // Pin Reset RFID

MFRC522 rfid(SS_PIN, RST_PIN);
LiquidCrystal_I2C lcd(0x27, 16, 2);

FirebaseData fbdo;
FirebaseAuth auth;
FirebaseConfig config;

String strID;

void setup() {
  Serial.begin(115200);

  // Inisialisasi Bus I2C LCD (SDA = D2, SCL = D1)
  Wire.begin(D2, D1);
  lcd.init();
  lcd.backlight();

  // Tampilan Teks Menghubungkan WiFi
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi.");
  
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  
  Serial.println("\n[CONNECTED] Terhubung ke WiFi!");
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("WiFi Connected!");
  delay(1000);

  // Inisialisasi Bus SPI & Modul RFID
  SPI.begin();
  rfid.PCD_Init();

  // Inisialisasi Firebase dengan URL & Database Secret (Legacy Token)
  config.database_url = DATABASE_URL;
  config.signer.tokens.legacy_token = DATABASE_SECRET;

  Firebase.begin(&config, &auth);
  Firebase.reconnectWiFi(true);

  // Tampilan Standby Siap Baca Kartu
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("  RFID Reader   ");
  lcd.setCursor(0, 1);
  lcd.print("Tempelkan Kartu ");

  Serial.println("=========================================");
  Serial.println("Sistem RFID Firebase Ready! Tempelkan kartu...");
  Serial.println("=========================================");
}

void loop() {
  // Cek jika tidak ada kartu baru atau gagal membaca serial kartu
  if (!rfid.PICC_IsNewCardPresent() || !rfid.PICC_ReadCardSerial()) {
    return;
  }

  // Formatting UID Kartu ke bentuk String HEX
  strID = "";
  for (byte i = 0; i < rfid.uid.size; i++) {
    strID += (rfid.uid.uidByte[i] < 0x10 ? "0" : "") +
             String(rfid.uid.uidByte[i], HEX) +
             (i != rfid.uid.size - 1 ? ":" : "");
  }
  strID.toUpperCase();

  Serial.print("[RFID DETECTED] Kartu ID: ");
  Serial.println(strID);

  // Update Tampilan LCD
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("ID:" + strID);
  lcd.setCursor(0, 1);
  lcd.print("Kirim Firebase..");

  // 1. Update UID Kartu Terakhir ke Node "/Last_Card/uid"
  if (Firebase.RTDB.setString(&fbdo, "/Last_Card/uid", strID)) {
    Serial.println("-> [FIREBASE SUCCESS] UID Terakhir Diperbarui!");
  } else {
    Serial.print("-> [FIREBASE ERROR] ");
    Serial.println(fbdo.errorReason());
  }

  // 2. Tambah Log Riwayat Akses ke Node "/RFID_Logs"
  FirebaseJson json;
  json.set("uid", strID);
  json.set("status", "SUCCESS");

  if (Firebase.RTDB.pushJSON(&fbdo, "/RFID_Logs", &json)) {
    Serial.println("-> [FIREBASE SUCCESS] Riwayat Log Ditambahkan!");
    lcd.setCursor(0, 1);
    lcd.print("Status: SUCCESS ");
  } else {
    lcd.setCursor(0, 1);
    lcd.print("Status: FAILED  ");
  }

  // Hentikan komunikasi sementara dengan kartu
  rfid.PICC_HaltA();
  rfid.PCD_StopCrypto1();

  delay(2000); // Jeda 2 detik

  // Kembali ke status standby
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("  RFID Reader   ");
  lcd.setCursor(0, 1);
  lcd.print("Tempelkan Kartu ");
}