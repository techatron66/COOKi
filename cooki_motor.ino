// ─────────────────────────────────────────────────────────────────────────────
// COOKi Motor Controller  (Arduino Uno / Nano)
// ─────────────────────────────────────────────────────────────────────────────
//
// Serial input:  "speed,direction\n"   e.g.  "200,1\n"
//   speed:       0-255   (PWM duty cycle)
//   direction:   0=CCW   1=CW
//
// Serial reply:  "OK spd=200 dir=1"
//
// Wiring:
//   PWM_PIN (D9)  → motor driver IN1 / PWM input
//   DIR_PIN (D8)  → motor driver IN2 / direction input
//   Motor driver GND → Arduino GND
//   Motor driver VM  → external power (6-12V for the motor)
//
// Compatible motor drivers: L298N, L9110S, DRV8833, TB6612FNG
// ─────────────────────────────────────────────────────────────────────────────

const int PWM_PIN = 9;
const int DIR_PIN = 8;
const int LED_PIN = 13;   // built-in LED blinks on each command

void setup() {
  Serial.begin(9600);
  pinMode(PWM_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(LED_PIN, OUTPUT);
  analogWrite(PWM_PIN, 0);
  digitalWrite(DIR_PIN, LOW);
  Serial.println("COOKi motor ready");
}

void loop() {
  if (!Serial.available()) return;

  String cmd = Serial.readStringUntil('\n');
  cmd.trim();

  int comma = cmd.indexOf(',');
  if (comma == -1) {
    Serial.println("ERR bad format — expected speed,direction");
    return;
  }

  int spd = cmd.substring(0, comma).toInt();
  int dir = cmd.substring(comma + 1).toInt();

  spd = constrain(spd, 0, 255);
  dir = constrain(dir, 0, 1);

  digitalWrite(DIR_PIN, dir);
  analogWrite(PWM_PIN, spd);

  // Blink LED on command received
  digitalWrite(LED_PIN, HIGH);
  delay(20);
  digitalWrite(LED_PIN, LOW);

  Serial.print("OK spd=");
  Serial.print(spd);
  Serial.print(" dir=");
  Serial.println(dir);
}
