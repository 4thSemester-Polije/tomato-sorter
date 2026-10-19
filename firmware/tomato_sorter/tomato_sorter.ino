// Firmware sortir tomat (ESP32). Protokol serial: docs/PROTOCOL.md v0.3
// Pemetaan jalur: A = gate servo 3 (dulu "matang"), B = gate servo 2 (dulu "setengah"), C = jalur lurus (dulu "busuk").
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP32Servo.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

// ================= PIN =================
#define SDA_PIN 21
#define SCL_PIN 22

#define IR_PIN 16

#define L298N_IN1 4
#define L298N_IN2 27
#define L298N_ENA 14

#define SERVO_PUSHER_PIN   18
#define SERVO_SETENGAH_PIN 19   // gate grade B (servo 2)
#define SERVO_MATANG_PIN   23   // gate grade A (servo 3)

// ================= PENGATURAN KONVEYOR & SENSOR =================
#define KECEPATAN_KONVEYOR 255   // 0 - 255
#define IR_AKTIF_LEVEL     LOW   // Ganti ke HIGH jika sensor IR aktif HIGH saat ada objek
#define TES_MOTOR_AWAL     true  // true = motor berputar 2 detik saat startup untuk tes wiring
#define IR_DEBOUNCE_MS     50    // Sensor harus aktif minimal sekian ms agar dianggap buah
#define IR_COOLDOWN_MS     500   // Setelah siklus selesai, abaikan IR sebentar
#define KONVEYOR_JALAN_SAAT_IDLE false  // false = konveyor mati saat menunggu tomat (stasiun foto manual). true = perilaku lama

// ================= PENGATURAN SERVO (derajat) =================
#define SUDUT_TUTUP 0
#define SUDUT_BUKA  90

// ================= PENGATURAN WAKTU (ms) - SESUAIKAN DENGAN ALAT =================
#define WAKTU_SODOK           1000   // Lama servo 1 menyodok buah ke konveyor
#define WAKTU_PUSHER_KEMBALI  400    // BARU, PERKIRAAN: waktu pusher kembali sebelum konveyor jalan. Ukur di alat
#define WAKTU_TEMPUH_C_LURUS  4000   // Konveyor jalan sampai buah grade C lewat jalur lurus
#define WAKTU_TEMPUH_B_SERVO2 4000   // Servo 2 terbuka sampai buah masuk jalur grade B
#define WAKTU_TEMPUH_A_SERVO3 5000   // Servo 3 terbuka sampai buah masuk jalur grade A
#define WAKTU_TUNGGU_GRADE    15000  // BARU, PERKIRAAN: batas menunggu GRADE dari PC setelah DETECT
#define WAKTU_BARIS_IDLE      50     // Baris tanpa newline dianggap selesai setelah sekian ms (sama seperti Serial.setTimeout(50) lama)

// ================= PWM MOTOR =================
#define PWM_FREQ 1000
#define PWM_RES  8
#define PWM_CHANNEL 15               // dipakai hanya untuk core ESP32 versi 2.x

LiquidCrystal_I2C lcd(0x27, 16, 2);  // jika LCD kosong, coba alamat 0x3F

Servo servoPusher;
Servo servoSetengah;   // gate B
Servo servoMatang;     // gate A

// ================= STATUS SISTEM =================
enum Status {
  CARI_BUAH,        // menunggu IR deteksi buah
  MENUNGGU_GRADE,   // konveyor berhenti, menunggu GRADE dari PC (atau huruf A/B/C manual)
  MENYODOK,         // servo 1 menyodok buah ke konveyor
  PUSHER_KEMBALI,   // menunggu pusher kembali
  MENGANTAR,        // konveyor jalan mengantar buah ke jalur grade (berbasis waktu)
  TUNGGU_BEBAS      // setelah GRADE_TIMEOUT: tunggu tomat diambil (IR bebas)
};

Status status = CARI_BUAH;
char gradeDipilih = 0;               // 'A', 'B', atau 'C'
unsigned long idTomat = 0;           // penghitung ID, naik setiap DETECT
unsigned long idSaatIni = 0;
unsigned long tMulai = 0;            // penanda waktu tahap berjalan
unsigned long tSelesai = 0;          // waktu siklus terakhir selesai
unsigned long tIRMulai = 0;          // debounce IR aktif
unsigned long tBebasMulai = 0;       // debounce IR bebas

// Penghitung = jumlah yang DIPERINTAHKAN, bukan terkonfirmasi masuk wadah (pengantaran berbasis waktu).
unsigned long jumlahA = 0, jumlahB = 0, jumlahC = 0;

// Buffer baris serial
char rxBuf[24];
uint8_t rxLen = 0;
bool rxOverflow = false;
unsigned long rxTerakhir = 0;

// Prototipe (agar aman di PlatformIO)
void motorPwmInit();
void motorPwmWrite(int duty);
void motorMaju();
void motorStop();
bool irTerdeteksi();
bool irBebas();
void dbg(const char* teks);
void kirimReady();
void kirimDetect(unsigned long id);
void kirimAck(unsigned long id);
void kirimDone(unsigned long id, char g);
void kirimErr(unsigned long id, const char* kode);
bool selesaiBaris(char* out, size_t ukuran);
bool bacaBaris(char* out, size_t ukuran);
void tanganiBaris(const char* baris);
void mulaiCariBuah();
void mulaiMenungguGrade();
void mulaiSodok(char grade);
void mulaiMengantar();
void selesaiMengantar();
unsigned long waktuTempuh(char grade);

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

bool irBebas() {
  if (digitalRead(IR_PIN) != IR_AKTIF_LEVEL) {
    if (tBebasMulai == 0) tBebasMulai = millis();
    return (millis() - tBebasMulai >= IR_DEBOUNCE_MS);
  }
  tBebasMulai = 0;
  return false;
}

// ================= KELUARAN SERIAL =================
// Semua log debug WAJIB diawali '#'. Selain itu hanya pesan protokol.
void dbg(const char* teks) { Serial.print("# "); Serial.println(teks); }
void kirimReady() { Serial.println("READY"); }
void kirimDetect(unsigned long id) { Serial.print("DETECT,"); Serial.println(id); }
void kirimAck(unsigned long id) { Serial.print("ACK,"); Serial.println(id); }
void kirimDone(unsigned long id, char g) {
  Serial.print("DONE,"); Serial.print(id); Serial.print(','); Serial.println(g);
}
void kirimErr(unsigned long id, const char* kode) {
  Serial.print("ERR,"); Serial.print(id); Serial.print(','); Serial.println(kode);
}

// ================= MASUKAN SERIAL (non-blocking, tanpa String) =================
bool selesaiBaris(char* out, size_t ukuran) {
  rxBuf[rxLen] = 0;
  strncpy(out, rxOverflow ? "?" : rxBuf, ukuran - 1);
  out[ukuran - 1] = 0;
  rxLen = 0;
  rxOverflow = false;
  return true;
}

bool bacaBaris(char* out, size_t ukuran) {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    rxTerakhir = millis();
    if (c == '\r') continue;
    if (c == '\n') return selesaiBaris(out, ukuran);
    if (rxLen < sizeof(rxBuf) - 1) rxBuf[rxLen++] = c; else rxOverflow = true;
  }
  // Serial Monitor "No line ending": anggap selesai setelah hening sebentar
  if (rxLen > 0 && millis() - rxTerakhir >= WAKTU_BARIS_IDLE) return selesaiBaris(out, ukuran);
  return false;
}

// Mode PC: "GRADE,<id>,<A|B|C>". Mode debug manual: huruf tunggal A/B/C (memakai ID saat ini).
void tanganiBaris(const char* baris) {
  if (baris[0] == 0) return;
  char g = 0;
  unsigned long id = idSaatIni;
  char sisa[8];

  if (strlen(baris) == 1 && (toupper(baris[0]) >= 'A' && toupper(baris[0]) <= 'C')) {
    g = (char)toupper(baris[0]);
  } else if (sscanf(baris, "GRADE,%lu,%7s", &id, sisa) == 2 && strlen(sisa) == 1 &&
             (toupper(sisa[0]) >= 'A' && toupper(sisa[0]) <= 'C')) {
    g = (char)toupper(sisa[0]);
  } else {
    kirimErr(idSaatIni, "BAD_CMD");
    return;
  }

  if (status != MENUNGGU_GRADE || id != idSaatIni) {
    kirimErr(id, "BUSY");
    return;
  }
  kirimAck(id);
  mulaiSodok(g);
}

// ================= SETUP =================
void setup() {
  Serial.begin(115200);

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
    dbg("TEST motor berputar 2 detik");
    lcd.clear();
    lcd.setCursor(0, 0);
    lcd.print("TES MOTOR...");
    motorMaju();
    delay(2000);
    motorStop();
    delay(500);
  }

  char buf[24];
  snprintf(buf, sizeof(buf), "IR nilai awal: %d", digitalRead(IR_PIN));
  dbg(buf);

  mulaiCariBuah();
}

// ================= LOOP (STATE MACHINE) =================
void loop() {
  // Baca baris serial di semua status; balasan ditentukan oleh tanganiBaris()
  char baris[24];
  if (bacaBaris(baris, sizeof(baris))) tanganiBaris(baris);

  switch (status) {

    // ---- Tunggu IR deteksi buah ----
    case CARI_BUAH:
      if (millis() - tSelesai >= IR_COOLDOWN_MS && irTerdeteksi()) {
        mulaiMenungguGrade();
      }
      break;

    // ---- Konveyor berhenti, tunggu GRADE. Ada batas waktu agar tidak macet selamanya ----
    case MENUNGGU_GRADE:
      if (millis() - tMulai >= WAKTU_TUNGGU_GRADE) {
        kirimErr(idSaatIni, "GRADE_TIMEOUT");
        lcd.clear();
        lcd.setCursor(0, 0);
        lcd.print("GRADE TIMEOUT   ");
        lcd.setCursor(0, 1);
        lcd.print("Ambil tomat...  ");
        tBebasMulai = 0;
        status = TUNGGU_BEBAS;
      }
      break;

    // ---- Servo 1 menyodok buah ke konveyor ----
    case MENYODOK:
      if (millis() - tMulai >= WAKTU_SODOK) {
        servoPusher.write(SUDUT_TUTUP);   // pusher kembali
        tMulai = millis();
        status = PUSHER_KEMBALI;
      }
      break;

    // ---- Beri waktu pusher kembali sebelum konveyor jalan ----
    case PUSHER_KEMBALI:
      if (millis() - tMulai >= WAKTU_PUSHER_KEMBALI) {
        mulaiMengantar();
      }
      break;

    // ---- Konveyor jalan mengantar buah (berbasis waktu, tanpa konfirmasi) ----
    case MENGANTAR:
      if (millis() - tMulai >= waktuTempuh(gradeDipilih)) {
        selesaiMengantar();
      }
      break;

    // ---- Tomat belum diambil setelah timeout: tunggu IR bebas ----
    case TUNGGU_BEBAS:
      if (irBebas()) {
        mulaiCariBuah();
      }
      break;
  }
}

// ================= FUNGSI TAHAPAN =================
void mulaiCariBuah() {
  status = CARI_BUAH;
  tSelesai = millis();
  tIRMulai = 0;
  if (KONVEYOR_JALAN_SAAT_IDLE) motorMaju(); else motorStop();

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(KONVEYOR_JALAN_SAAT_IDLE ? "Konveyor: JALAN " : "Konveyor: STOP  ");
  lcd.setCursor(0, 1);
  lcd.print("Mencari Buah... ");
  kirimReady();
}

void mulaiMenungguGrade() {
  motorStop();
  idSaatIni = ++idTomat;
  tMulai = millis();
  status = MENUNGGU_GRADE;
  while (Serial.available() > 0) Serial.read();   // buang sisa input lama
  rxLen = 0;
  rxOverflow = false;

  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("BUAH TERDETEKSI!");
  lcd.setCursor(0, 1);
  lcd.print("Menunggu grade  ");
  kirimDetect(idSaatIni);
  dbg("Manual: ketik A, B, atau C (A=servo 3, B=servo 2, C=lurus)");
}

void mulaiSodok(char grade) {
  gradeDipilih = grade;

  char baris[17];
  snprintf(baris, sizeof(baris), "Grade : %c       ", grade);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print(baris);
  lcd.setCursor(0, 1);
  lcd.print("Action: Sodok   ");

  dbg("SERVO 1 menyodok buah ke konveyor");
  servoPusher.write(SUDUT_BUKA);
  tMulai = millis();
  status = MENYODOK;
}

void mulaiMengantar() {
  // Konveyor jalan dan buka gate sesuai grade
  motorMaju();

  lcd.setCursor(0, 1);
  if (gradeDipilih == 'A') {
    servoMatang.write(SUDUT_BUKA);
    lcd.print("Jalur : SERVO 3 ");
    dbg("SERVO 3 terbuka (grade A)");
  } else if (gradeDipilih == 'B') {
    servoSetengah.write(SUDUT_BUKA);
    lcd.print("Jalur : SERVO 2 ");
    dbg("SERVO 2 terbuka (grade B)");
  } else {
    lcd.print("Jalur : LURUS   ");
    dbg("Jalur lurus (grade C)");
  }

  tMulai = millis();
  status = MENGANTAR;
}

void selesaiMengantar() {
  // Tutup gate, hitung, lapor, siap tomat berikutnya
  servoSetengah.write(SUDUT_TUTUP);
  servoMatang.write(SUDUT_TUTUP);

  if (gradeDipilih == 'A') jumlahA++;
  else if (gradeDipilih == 'B') jumlahB++;
  else jumlahC++;

  kirimDone(idSaatIni, gradeDipilih);
  char buf[48];
  snprintf(buf, sizeof(buf), "Diperintahkan A:%lu B:%lu C:%lu", jumlahA, jumlahB, jumlahC);
  dbg(buf);

  mulaiCariBuah();
}

unsigned long waktuTempuh(char grade) {
  if (grade == 'A') return WAKTU_TEMPUH_A_SERVO3;
  if (grade == 'B') return WAKTU_TEMPUH_B_SERVO2;
  return WAKTU_TEMPUH_C_LURUS;   // nilai tak dikenal: jalur lurus, tanpa membuka gate
}
