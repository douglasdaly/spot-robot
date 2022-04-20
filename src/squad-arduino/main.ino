/*
  Main Squad Robot Arduino controller code.
*/
#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Servo.h>
#include <math.h>
#include <Wire.h>


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

// - Board Pins
#define PIN_FL_HIP 42
#define PIN_FL_FEMUR 44
#define PIN_FL_LEG 46

#define PIN_FR_HIP 40
#define PIN_FR_FEMUR 38
#define PIN_FR_LEG 36

#define PIN_BL_HIP 48
#define PIN_BL_FEMUR 50
#define PIN_BL_LEG 52

#define PIN_BR_HIP 34
#define PIN_BR_FEMUR 32
#define PIN_BR_LEG 30


// VARIABLES
bool zeroed = false;

// - Servo
Servo servos[MAX_SERVOS];
int pulses[MAX_SERVOS];

// - IMU
Adafruit_MPU6050 mpu6050;
sensors_event_t mpuAcc, mpuGyro, mpuTemp;
float imuZero[3];
float imuOut[6];
long imuPreInterval;
float imuInterval = 0.0;

// - Timing
unsigned long startTime;
unsigned long loopTime;
unsigned long previousLoopTime = 0;

// Constants
const float kAngleToMicroseconds = (SERVO_PWM_MAX - SERVO_PWM_MIN) / SERVO_MAX_ANGLE;


// FUNCTIONS

void connectServos() {
  servos[0].attach(PIN_FL_HIP);
  servos[1].attach(PIN_FL_FEMUR);
  servos[2].attach(PIN_FL_LEG);
  servos[3].attach(PIN_FR_HIP);
  servos[4].attach(PIN_FR_FEMUR);
  servos[5].attach(PIN_FR_LEG);
  servos[6].attach(PIN_BL_HIP);
  servos[7].attach(PIN_BL_FEMUR);
  servos[8].attach(PIN_BL_LEG);
  servos[9].attach(PIN_BR_HIP);
  servos[10].attach(PIN_BR_FEMUR);
  servos[11].attach(PIN_BR_LEG);
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

void calibrateIMU(unsigned int n) {
    for (int i = 0; i < n; ++i) {
        mpu6050.getEvent(&mpuAcc, &mpuGyro, &mpuTemp);

        imuZero[0] += mpuGyro.gyro.x;
        imuZero[1] += mpuGyro.gyro.y;
        imuZero[2] += mpuGyro.gyro.z;

        delay(10);
    }

    for (int i = 0; i < 3; ++i) {
        imuZero[i] /= ((float)n);
    }

    imuPreInterval = millis();
}

void updateIMU(bool output, bool clearAfter) {
    // - Update data from IMU
    mpu6050.getEvent(&mpuAcc, &mpuGyro, &mpuTemp);

    imuInterval = (millis() - imuPreInterval) * 0.001;

    imuOut[0] = mpuAcc.acceleration.x;
    imuOut[1] = mpuAcc.acceleration.y;
    imuOut[2] = mpuAcc.acceleration.z;
    imuOut[3] += (mpuGyro.gyro.x - imuZero[0]) * imuInterval;
    imuOut[4] -= (mpuGyro.gyro.y - imuZero[1]) * imuInterval;
    imuOut[5] += (mpuGyro.gyro.z - imuZero[2]) * imuInterval;

    imuPreInterval = millis();

    if (output) {
        // - Write out the data
        Serial.write((byte *)imuOut, 24);
        Serial.flush();
    }

    if (clearAfter) {
        // - Zero out accumulated angles
        for (int i = 0; i < 3; ++i) {
            imuOut[i+3] = 0.0;
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

    // Initialize IMU
    mpu6050.begin();
    mpu6050.setAccelerometerRange(MPU6050_RANGE_2_G);
    mpu6050.setGyroRange(MPU6050_RANGE_250_DEG);
    mpu6050.setFilterBandwidth(MPU6050_BAND_21_HZ);

    for (int i = 0; i < 3; ++i) {
        imuZero[i] = 0.0;
    }

    imuPreInterval = millis();

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
            // - Calibrate IMU
            calibrateIMU(100);
            updateIMU(false, true);
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
                updateIMU(true, true);
            } else {
                buffer_in_pos++;
                updateIMU(false, false);
            }
        } else {
            updateIMU(false, false);
        }
    }

    // - Write pulses
    if (loopTime - previousLoopTime >= LOOP_INTERVAL) {
        previousLoopTime = loopTime;
        moveServos(pulses);
    }
}
