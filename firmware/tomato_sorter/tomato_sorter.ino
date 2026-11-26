// Firmware sortir tomat (ESP32). Protokol serial: docs/PROTOCOL.md v0.3
// Pemetaan jalur: A = gate servo 3 (dulu "matang"), B = gate servo 2 (dulu "setengah"), C = jalur lurus (dulu "busuk").
#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <ESP32Servo.h>
#include <esp_arduino_version.h>   // wajib: dipakai oleh #if ESP_ARDUINO_VERSION_MAJOR
#include <stdio.h>
#include <string.h>
#include <ctype.h>

// ================= PIN =================
#define SDA_PIN 21
#define SCL_PIN 22

#define IR_PIN 16
#define BUZZER_PIN 26

#define L298N_IN1 4
#define L298N_IN2 27
#define L298N_ENA 14

#define SERVO_PUSHER_PIN   18
#define SERVO_SETENGAH_PIN 19   // gate grade B (servo 2)
#define SERVO_MATANG_PIN   23   // gate grade A (servo 3)

// ================= PENGATURAN KONVEYOR & SENSOR =================
#define KECEPATAN_KONVEYOR 200   // 0 - 255. Waktu tempuh jalur berbasis waktu, jadi kecepatan harus sama dengan saat WAKTU_TEMPUH_* dikalibrasi
#define IR_AKTIF_LEVEL     LOW   // Ganti ke HIGH jika sensor IR aktif HIGH saat ada objek
#define TES_MOTOR_AWAL     true  // true = motor berputar 2 detik saat startup untuk tes wiring
#define IR_DEBOUNCE_MS     50    // Sensor harus aktif minimal sekian ms agar dianggap buah
#define IR_COOLDOWN_MS     500   // Setelah siklus selesai, abaikan IR sebentar
#define KONVEYOR_JALAN_SAAT_IDLE false  // false = konveyor mati saat menunggu tomat (stasiun foto manual). true = perilaku lama

// ================= PENGATURAN MOTOR (kick-start) =================
// Motor DC diam + beban + PWM 200/255 lewat L298N sering tidak punya torsi awal cukup (stiction),
// akibatnya "harus dipancing tangan". Solusi standar: beri duty penuh sebentar, lalu turun ke kecepatan kerja.
#define KICK_DUTY    255   // 0 - 255, duty saat motor mulai berputar dari diam
#define WAKTU_KICK_MS 300  // lama duty penuh. Naikkan (mis. 500) jika masih berat; 0 = matikan kick

// ================= PENGATURAN BUZZER =================
#define BUZZER_JUMLAH_BIP 3    // jumlah bunyi saat buah terdeteksi (sama seperti kode lama)
#define BUZZER_BIP_MS     120  // lama tiap bunyi
#define BUZZER_JEDA_MS    120  // jeda antar bunyi

// ================= MODE UJI SERVO (khusus testing) =================
// true  = perintah uji lewat Serial Monitor aktif (lihat bagian bawah komentar ini)
// false = perintah uji dimatikan total. WAJIB false untuk operasi normal / demo / produksi.
// Perintah (hanya diterima saat sistem idle/CARI_BUAH; semua balasan berawalan '#', jadi diabaikan PC):
//   S1 / S2 / S3 : buka servo 1 (pusher) / servo 2 (gate B) / servo 3 (gate A) sebentar lalu tutup
//   TA / TB / TC : simulasi siklus penuh grade A / B / C (sodok -> konveyor -> gate) tanpa PC
//   HELP         : tampilkan daftar perintah
// Simulasi TIDAK mengirim DONE/ACK dan TIDAK menambah penghitung, jadi dashboard tidak tercemar.
#define MODE_TEST_SERVO true
#define WAKTU_TES_SERVO 1500   // ms servo tetap terbuka pada perintah S1/S2/S3

// ================= PENGATURAN SERVO (derajat) =================
#define SUDUT_TUTUP 0
#define SUDUT_BUKA  60

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

// Motor kick-start
bool kickAktif = false;
unsigned long tKick = 0;

// Buzzer non-blocking
uint8_t buzzerSisa = 0;        // jumlah bunyi yang belum dimainkan
bool buzzerNyala = false;
unsigned long tBuzzer = 0;

// Mode uji servo
bool modeSimulasi = false;     // true selama siklus simulasi TA/TB/TC
Servo* servoUji = nullptr;     // servo yang sedang diuji (S1/S2/S3)
unsigned long tServoUji = 0;

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
void motorUpdate();
void bunyiDeteksi();
void buzzerUpdate();
bool tanganiTest(const char* baris);
void ujiServoMulai(Servo* sv, const char* nama);
void ujiServoUpdate();
void simulasiMulai(char grade);
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
  if (WAKTU_KICK_MS > 0 && KICK_DUTY > KECEPATAN_KONVEYOR) {
    motorPwmWrite(KICK_DUTY);       // dorong dulu agar lepas dari diam
    kickAktif = true;
    tKick = millis();
  } else {
    motorPwmWrite(KECEPATAN_KONVEYOR);
    kickAktif = false;
  }
}

void motorStop() {
  kickAktif = false;
  digitalWrite(L298N_IN1, LOW);
  digitalWrite(L298N_IN2, LOW);
  motorPwmWrite(0);
}

// Dipanggil tiap loop(): turunkan ke kecepatan kerja setelah kick selesai
void motorUpdate() {
  if (kickAktif && millis() - tKick >= WAKTU_KICK_MS) {
    motorPwmWrite(KECEPATAN_KONVEYOR);
    kickAktif = false;
  }
}

// ================= BUZZER (non-blocking, tidak menunda state machine) =================
void bunyiDeteksi() {
  buzzerSisa = BUZZER_JUMLAH_BIP;
  buzzerNyala = false;
  tBuzzer = millis() - BUZZER_JEDA_MS;   // mulai segera
}

void buzzerUpdate() {
  if (buzzerSisa == 0 && !buzzerNyala) return;
  unsigned long now = millis();
  if (buzzerNyala) {
    if (now - tBuzzer >= BUZZER_BIP_MS) {
      digitalWrite(BUZZER_PIN, LOW);
      buzzerNyala = false;
      tBuzzer = now;
    }
  } else if (buzzerSisa > 0 && now - tBuzzer >= BUZZER_JEDA_MS) {
    digitalWrite(BUZZER_PIN, HIGH);
    buzzerNyala = true;
    buzzerSisa--;
    tBuzzer = now;
  }
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
#if MODE_TEST_SERVO
  if (tanganiTest(baris)) return;
#endif
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

  pinMode(IR_PIN, INPUT_PULLUP);
  pinMode(L298N_IN1, OUTPUT);
  pinMode(L298N_IN2, OUTPUT);
  pinMode(L298N_ENA, OUTPUT);           // sama seperti kode lama; harus SEBELUM motorPwmInit()
  pinMode(BUZZER_PIN, OUTPUT);
  digitalWrite(BUZZER_PIN, LOW);
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
    if (kickAktif) {                       // delay() memblokir loop, jadi kick diselesaikan manual di sini
      delay(WAKTU_KICK_MS);
      motorPwmWrite(KECEPATAN_KONVEYOR);
      kickAktif = false;
      delay(2000 - WAKTU_KICK_MS);
    } else {
      delay(2000);
    }
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
  motorUpdate();
  buzzerUpdate();
#if MODE_TEST_SERVO
  ujiServoUpdate();
#endif

  static int irNilaiTerakhir = -1;
  int irNilai = digitalRead(IR_PIN);
  if (irNilaiTerakhir >= 0 && irNilai != irNilaiTerakhir) {
    char buf[32];
    snprintf(buf, sizeof(buf), "IR raw berubah: %d", irNilai);
    dbg(buf);
  }
  irNilaiTerakhir = irNilai;

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
  bunyiDeteksi();   // setelah DETECT terkirim, tidak menunda protokol
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

  if (grade == 'A') {
    servoMatang.write(SUDUT_BUKA);
    dbg("SERVO 3 grade A bergerak bersama pusher");
  } else if (grade == 'B') {
    servoSetengah.write(SUDUT_BUKA);
    dbg("SERVO 2 grade B bergerak bersama pusher");
  }
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
    lcd.print("Jalur : SERVO 3 ");
    dbg("Jalur SERVO 3 grade A (gate tetap terbuka)");
  } else if (gradeDipilih == 'B') {
    lcd.print("Jalur : SERVO 2 ");
    dbg("Jalur SERVO 2 grade B (gate tetap terbuka)");
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

  if (modeSimulasi) {
    // Simulasi uji: tidak ada DONE, tidak ada penghitung. Dashboard tidak tahu apa-apa.
    modeSimulasi = false;
    dbg("TEST simulasi selesai");
    mulaiCariBuah();
    return;
  }

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


// ================= MODE UJI SERVO (hanya aktif bila MODE_TEST_SERVO true) =================
#if MODE_TEST_SERVO

void ujiServoMulai(Servo* sv, const char* nama) {
  char buf[40];
  snprintf(buf, sizeof(buf), "TEST %s buka", nama);
  dbg(buf);
  sv->write(SUDUT_BUKA);
  servoUji = sv;
  tServoUji = millis();
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("TEST SERVO");
  lcd.setCursor(0, 1);
  lcd.print(nama);
}

void ujiServoUpdate() {
  if (servoUji != nullptr && millis() - tServoUji >= WAKTU_TES_SERVO) {
    servoUji->write(SUDUT_TUTUP);
    servoUji = nullptr;
    dbg("TEST servo tutup");
    // Pulihkan LCD hanya bila sistem masih idle; jangan menimpa siklus nyata bila IR sempat terpicu
    if (status == CARI_BUAH) mulaiCariBuah();
  }
}

void simulasiMulai(char grade) {
  modeSimulasi = true;
  idSaatIni = idTomat;   // tidak menaikkan idTomat; ID asli tetap berurutan
  dbg("TEST simulasi siklus penuh dimulai");
  mulaiSodok(grade);     // sodok -> pusher kembali -> konveyor + gate -> selesaiMengantar()
}

// true bila baris adalah perintah uji (ditangani di sini, tidak diteruskan ke protokol)
bool tanganiTest(const char* baris) {
  char cmd[8];
  size_t n = strlen(baris);
  if (n == 0 || n > 4) return false;
  for (size_t i = 0; i < n; i++) cmd[i] = (char)toupper(baris[i]);
  cmd[n] = 0;

  bool kenal = (strcmp(cmd, "S1") == 0 || strcmp(cmd, "S2") == 0 || strcmp(cmd, "S3") == 0 ||
                strcmp(cmd, "TA") == 0 || strcmp(cmd, "TB") == 0 || strcmp(cmd, "TC") == 0 ||
                strcmp(cmd, "HELP") == 0);
  if (!kenal) return false;

  if (strcmp(cmd, "HELP") == 0) {
    dbg("S1/S2/S3 = tes servo 1(pusher)/2(gate B)/3(gate A)");
    dbg("TA/TB/TC = simulasi siklus penuh grade A/B/C");
    dbg("Hanya diterima saat idle (Mencari Buah)");
    return true;
  }

  if (status != CARI_BUAH || servoUji != nullptr) {
    dbg("TEST ditolak: sistem sedang bekerja");
    return true;
  }

  if (strcmp(cmd, "S1") == 0) ujiServoMulai(&servoPusher, "SERVO 1 pusher");
  else if (strcmp(cmd, "S2") == 0) ujiServoMulai(&servoSetengah, "SERVO 2 gate B");
  else if (strcmp(cmd, "S3") == 0) ujiServoMulai(&servoMatang, "SERVO 3 gate A");
  else simulasiMulai(cmd[1]);
  return true;
}

#endif
