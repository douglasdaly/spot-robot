/*
  Main Squad Robot Arduino controller code.
*/
#include <Servo.h>
#include <math.h>


// CONSTANTS
#define BAUD_RATE 115200
#define SERIAL_TIMEOUT 1
#define LOOP_INTERVAL 20
#define ZERO_INTERVAL 1000

// - Servos
#define MAX_SERVOS 12
#define MAX_PULSE 2500
#define MIN_PULSE 560

// - Servo bounds
#define MIN_HIP -45
#define MAX_HIP 45
#define MIN_FEMUR -90
#define MAX_FEMUR 90
#define MIN_LEG -55
#define MAX_LEG 33

// DEFINITIONS
#define SERVO_MAX_ANGLE 270
#define SERVO_PWM_MIN 500
#define SERVO_PWM_MAX 2500

#define ANG_MAX SERVO_MAX_ANGLE / 2
#define ANG_MIN -ANG_MAX


// VARIABLES
bool zeroed = false;

// - Servo
Servo servos[MAX_SERVOS];
int pulses[MAX_SERVOS];

// - Timing
unsigned long startTime;
unsigned long loopTime;
unsigned long previousLoopTime = 0;

// Constants
const float kAngleToMicroseconds = (SERVO_PWM_MAX - SERVO_PWM_MIN) / SERVO_MAX_ANGLE;


// FUNCTIONS

void connectServos() {
  // - Front Left (FL)
  servos[0].attach(42);  // Hip
  servos[1].attach(44);  // Femur
  servos[2].attach(46);  // Leg

  // - Front Right (FR)
  servos[3].attach(40);  // Hip
  servos[4].attach(38);  // Femur
  servos[5].attach(36);  // Leg

  // - Back Left (BL)
  servos[6].attach(48);  // Hip
  servos[7].attach(50);  // Femur
  servos[8].attach(52);  // Leg

  // - Back Right (BR)
  servos[9].attach(34);  // Hip
  servos[10].attach(32); // Femur
  servos[11].attach(30); // Leg
}

void moveServos(int pulses[MAX_SERVOS]) {
  for (int i=0; i<MAX_SERVOS; ++i) {
    servos[i].writeMicroseconds(pulses[i]);
  }
}

int getAngleMs(float angle, float minAngle, float maxAngle) {
  angle = (angle >= 0.0) ? min(angle, minAngle) : max(angle, maxAngle);
  return round(((SERVO_MAX_ANGLE / 2.0) + angle) * kAngleToMicroseconds);
}

void initPulses(float hipAngle, float femurAngle, float legAngle) {
    int hipPulse = getAngleMs(hipAngle, MIN_HIP, MAX_HIP);
    int femurPulse = getAngleMs(femurAngle, MIN_FEMUR, MAX_FEMUR);
    int legPulse = getAngleMs(legAngle, MIN_LEG, MAX_LEG);

    int sType;
    for (int i=0; i<MAX_SERVOS; ++i) {
        sType = i % 3;
        switch (sType) {
            case 0:
                pulses[i] = hipPulse;
                break;
            case 1:
                pulses[i] = femurPulse;
                break;
            case 2:
                pulses[i] = legPulse;
                break;
            default:
                pulses[i] = getAngleMs(0.0, ANG_MIN, ANG_MAX);
        }
    }
}


// ARDUINO
void setup() {
    Serial.begin(BAUD_RATE);
    Serial.setTimeout(SERIAL_TIMEOUT);

    // Setup servos
    connectServos();
    initPulses(0.0, 0.0, 0.0);

    // Wrap-up setup
    startTime = millis();
}

// Main Loop
void loop() {
    static char buffer_in[36];
    static size_t buffer_in_pos;

    loopTime = millis();

    if (!zeroed) {
        if (loopTime - startTime >= ZERO_INTERVAL) {
            zeroed = true;
        }
    } else {
        // - Serial IO
        if (Serial.available()) {
            char c_in = Serial.read();
            buffer_in[buffer_in_pos] = c_in;
            if (buffer_in_pos == sizeof(buffer_in) - 1) {
                // - Received all angles, update pulses
                float anglesIn[12];
                memcpy(&anglesIn, &buffer_in, sizeof(anglesIn));
                buffer_in_pos = 0;

                int sType;
                float aMin;
                float aMax;
                for (int i=0; i < MAX_SERVOS; i++) {
                    sType = i % 3;
                    switch (sType) {
                        case 0:
                            aMin = MIN_HIP;
                            aMax = MAX_HIP;
                            break;
                        case 1:
                            aMin = MIN_FEMUR;
                            aMax = MAX_FEMUR;
                            break;
                        case 2:
                            aMin = MIN_LEG;
                            aMax = MAX_LEG;
                            break;
                        default:
                            aMin = ANG_MIN;
                            aMax = ANG_MAX;
                    }
                    pulses[i] = getAngleMs(anglesIn[i], aMin, aMax);
                }

                // - Write out IMU data
            } else {
                buffer_in_pos++;
            }
        }
    }

    // - Write pulses
    if (loopTime - previousLoopTime >= LOOP_INTERVAL) {
        previousLoopTime = loopTime;
        moveServos(pulses);
    }
}
