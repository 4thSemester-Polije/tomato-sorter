#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP32Servo.h>

// ================= PIN =================
#define SDA_PIN 21
#define SCL_PIN 22

#define IR_PIN 16

#define L298N_IN1 4
#define L298N_IN2 27
#define L298N_ENA 14

#define SERVO_PUSHER_PIN   18
#define SERVO_SETENGAH_PIN 19
#define SERVO_MATANG_PIN   23

// ================= PENGATURAN KONVEYOR & SENSOR =================
#define KECEPATAN_KONVEYOR 255   // 0 - 255
#define IR_AKTIF_LEVEL     LOW   // Ganti ke HIGH jika sensor IR aktif HIGH saat ada objek
#define TES_MOTOR_AWAL     true  // true = motor berputar 2 detik saat startup untuk tes wiring
#define IR_DEBOUNCE_MS     50    // Sensor harus aktif minimal sekian ms agar dianggap buah
#define IR_COOLDOWN_MS     500   // Setelah siklus selesai, abaikan IR sebentar

// ================= PENGATURAN SERVO (derajat) =================
#define SUDUT_TUTUP 0
#define SUDUT_BUKA  90

// ================= PENGATURAN WAKTU (ms) - SESUAIKAN DENGAN ALAT =================
#define WAKTU_SODOK           1000   // Lama servo 1 menyodok buah ke konveyor
#define WAKTU_TEMPUH_BUSUK    4000   // Konveyor jalan sampai buah busuk lewat jalur lurus
#define WAKTU_TEMPUH_SETENGAH 4000   // Servo 2 terbuka sampai buah masuk jalur setengah matang
#define WAKTU_TEMPUH_MATANG   5000   // Servo 3 terbuka sampai buah masuk jalur matang

// ================= PWM MOTOR =================
#define PWM_FREQ 1000
#define PWM_RES  8
#define PWM_CHANNEL 15               // dipakai hanya untuk core ESP32 versi 2.x

LiquidCrystal_I2C lcd(0x27, 16, 2);  // jika LCD kosong, coba alamat 0x3F

Servo servoPusher;
Servo servoSetengah;
Servo servoMatang;

// ================= STATUS SISTEM =================
enum Status {
  CARI_BUAH,        // konveyor jalan, menunggu IR deteksi buah
  MENUNGGU_GRADE,   // konveyor berhenti, menunggu input grade
  MENYODOK,         // servo 1 menyodok buah ke konveyor
  MENGANTAR         // konveyor jalan mengantar buah ke jalur grade
};

Status status = CARI_BUAH;
int gradeDipilih = 0;                // 1 = busuk, 2 = setengah, 3 = matang
unsigned long tMulai = 0;            // penanda waktu tahap berjalan
unsigned long tSelesai = 0;          // waktu siklus terakhir selesai
unsigned long tIRMulai = 0;          // untuk debounce IR

int jumlahBusuk = 0, jumlahSetengah = 0, jumlahMatang = 0;

// ================= KONTROL MOTOR =================
void motorPwmInit() {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(L298N_ENA, PWM_FREQ, PWM_RES);
#else
  ledcSetup(PWM_CHANNEL, PWM_FREQ, PWM_RES);
  ledcAttachPin(L298N_ENA, PWM_CHANNEL);
#endif
}

void motorPwmWrite(int duty) {
#if ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(L298N_ENA, duty);
#else
  ledcWrite(PWM_CHANNEL, duty);
#endif
}

void motorMaju() {
  digitalWrite(L298N_IN1, HIGH);
  digitalWrite(L298N_IN2, LOW);
  motorPwmWrite(KECEPATAN_KONVEYOR);
}

void motorStop() {
  digitalWrite(L298N_IN1, LOW);
  digitalWrite(L298N_IN2, LOW);
  motorPwmWrite(0);
}

// ================= SENSOR IR (dengan debounce) =================
bool irTerdeteksi() {
  if (digitalRead(IR_PIN) == IR_AKTIF_LEVEL) {
    if (tIRMulai == 0) tIRMulai = millis();
    return (millis() - tIRMulai >= IR_DEBOUNCE_MS);
  }
  tIRMulai = 0;
  return false;
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);
  Serial.setTimeout(50);

  pinMode(IR_PIN, INPUT);
  pinMode(L298N_IN1, OUTPUT);
  pinMode(L298N_IN2, OUTPUT);
  digitalWrite(L298N_IN1, LOW);
  digitalWrite(L298N_IN2, LOW);

  // LCD
  Wire.begin(SDA_PIN, SCL_PIN);
  lcd.init();
  lcd.backlight();

  // Servo
  ESP32PWM::allocateTimer(0);
  ESP32PWM::allocateTimer(1);
  ESP32PWM::allocateTimer(2);

  servoPusher.setPeriodHertz(50);
  servoSetengah.setPeriodHertz(50);
  servoMatang.setPeriodHertz(50);

  servoPusher.attach(SERVO_PUSHER_PIN, 500, 2400);
  servoSetengah.attach(SERVO_SETENGAH_PIN, 500, 2400);
  servoMatang.attach(SERVO_MATANG_PIN, 500, 2400);

  servoPusher.write(SUDUT_TUTUP);
  servoSetengah.write(SUDUT_TUTUP);
  servoMatang.write(SUDUT_TUTUP);

  // PWM motor
  motorPwmInit();
  motorPwmWrite(0);

  lcd.setCursor(0, 0);
  lcd.print(" SYSTEM CONVEYOR");
  lcd.setCursor(0, 1);
  lcd.print(" L298N + 3 SERVO");
  delay(2000);

  // Tes motor
  if (TES_MOTOR_AWAL) {
    Serial.println("[TEST] Motor berputar 2 detik...");
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("TES MOTOR...");
    motorMaju();
    delay(2000);
    motorStop();
    delay(500);
  }

  Serial.print("[IR] Nilai awal sensor: ");
  Serial.println(digitalRead(IR_PIN));

  mulaiCariBuah();
}

// ================= LOOP (STATE MACHINE) =================
void loop() {
  switch (status) {

    // ---- Konveyor jalan, tunggu IR deteksi buah ----
    case CARI_BUAH:
      if (millis() - tSelesai >= IR_COOLDOWN_MS && irTerdeteksi()) {
        motorStop();
        status = MENUNGGU_GRADE;
        while (Serial.available() > 0) Serial.read();   // buang sisa input lama
        tampilkanMenuGrade();
      }
      break;

    // ---- Konveyor berhenti, tunggu pilihan grade ----
    case MENUNGGU_GRADE:
      if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        input.toLowerCase();

        if (input == "busuk")         mulaiSodok(1);
        else if (input == "setengah") mulaiSodok(2);
        else if (input == "matang")   mulaiSodok(3);
        else if (input.length() > 0) {
          Serial.println("[ERROR] Perintah salah! Ketik: busuk / setengah / matang");
        }
      }
      break;

    // ---- Servo 1 menyodok buah ke konveyor ----
    case MENYODOK:
      if (millis() - tMulai >= WAKTU_SODOK) {
        servoPusher.write(SUDUT_TUTUP);   // pusher kembali
        mulaiMengantar();
      }
      break;

    // ---- Konveyor jalan mengantar buah ke grade ----
    case MENGANTAR:
      if (millis() - tMulai >= waktuTempuh(gradeDipilih)) {
        selesaiMengantar();
      }
      break;
  }
}

// ================= FUNGSI TAHAPAN =================
void mulaiCariBuah() {
  status = CARI_BUAH;
  tSelesai = millis();
  tIRMulai = 0;
  motorMaju();

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Konveyor: JALAN ");
  lcd.setCursor(0, 1);
  lcd.print("Mencari Buah... ");
  Serial.println("[SYSTEM] Konveyor jalan, mencari buah...");
}

void tampilkanMenuGrade() {
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("BUAH TERDETEKSI!");
  lcd.setCursor(0, 1);
  lcd.print("Ketik di Serial ");

  Serial.println("\n-------------------------------------------");
  Serial.println("[IR SENSOR] Buah terdeteksi! Konveyor BERHENTI.");
  Serial.println("Ketik salah satu di Serial Monitor:");
  Serial.println("1. busuk    (jalur lurus)");
  Serial.println("2. setengah (Servo 2 buka)");
  Serial.println("3. matang   (Servo 3 buka)");
  Serial.println("-------------------------------------------");
}

void mulaiSodok(int grade) {
  gradeDipilih = grade;

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("Buah  : " + namaGrade(grade));
  lcd.setCursor(0, 1);
  lcd.print("Action: Sodok   ");

  Serial.println("-> [GRADE] " + namaGrade(grade));
  Serial.println("-> [SERVO 1] Menyodok buah ke konveyor...");

  servoPusher.write(SUDUT_BUKA);
  tMulai = millis();
  status = MENYODOK;
}

void mulaiMengantar() {
  // Konveyor jalan lagi dan buka gate sesuai grade
  motorMaju();

  lcd.setCursor(0, 1);
  if (gradeDipilih == 1) {
    lcd.print("Jalur : LURUS   ");
    Serial.println("-> [KONVEYOR] Jalan, buah busuk lewat jalur lurus...");
  } else if (gradeDipilih == 2) {
    servoSetengah.write(SUDUT_BUKA);
    lcd.print("Jalur : SERVO 2 ");
    Serial.println("-> [SERVO 2] Terbuka, konveyor jalan mengantar buah...");
  } else if (gradeDipilih == 3) {
    servoMatang.write(SUDUT_BUKA);
    lcd.print("Jalur : SERVO 3 ");
    Serial.println("-> [SERVO 3] Terbuka, konveyor jalan mengantar buah...");
  }

  tMulai = millis();
  status = MENGANTAR;
}

void selesaiMengantar() {
  // Tutup gate, hitung, lanjut cari buah berikutnya
  servoSetengah.write(SUDUT_TUTUP);
  servoMatang.write(SUDUT_TUTUP);

  if (gradeDipilih == 1) jumlahBusuk++;
  else if (gradeDipilih == 2) jumlahSetengah++;
  else if (gradeDipilih == 3) jumlahMatang++;

  Serial.println("-> [SELESAI] Buah masuk jalur " + namaGrade(gradeDipilih));
  Serial.print("   Total  Busuk: ");
  Serial.print(jumlahBusuk);
  Serial.print(" | Setengah: ");
  Serial.print(jumlahSetengah);
  Serial.print(" | Matang: ");
  Serial.println(jumlahMatang);

  mulaiCariBuah();
}

unsigned long waktuTempuh(int grade) {
  if (grade == 1) return WAKTU_TEMPUH_BUSUK;
  if (grade == 2) return WAKTU_TEMPUH_SETENGAH;
  return WAKTU_TEMPUH_MATANG;
}

String namaGrade(int grade) {
  if (grade == 1) return "BUSUK";
  if (grade == 2) return "SETENGAH";
  return "MATANG";
}
