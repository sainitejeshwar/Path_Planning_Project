int nos = 5;
int pins[] = {3, 4, 5, 6, 7};
bool stable[] = {1, 1, 0, 1, 1};
bool ir_array[nos];
bool head = 0;
int pin_head = 8;
int weight = 0;

void sharp_turn_right();
void turn_right();
void turn_left();
void sharp_turn_left();
void forward();

void setup() {
  // put your setup code here, to run once:
  // test();
}

void loop() {
  // put your main code here, to run repeatedly:
  for (int i = 0; i < 5; i++)
  {
    ir_array[i] = digitalRead(pins[i]);
  }
  weight = -1 *(ir_array[0] + ir_array[1]) + ir_array[3] + ir_array[4];
  switch(weight)
  {
    case 2 : sharp_turn_right(); break;
    case 1 : turn_right(); break;
    case 0 : forward();
    case -1 : turn_left(); break;
    case -2 : sharp_turn_left(); break;
  }
}
void forward() {
  digitalWrite(left_motor, HIGH);
  digitalWrite(right_motor, HIGH);
}

void turn_right() {
  digitalWrite(left_motor, HIGH);
  digitalWrite(right_motor, LOW);
}

void turn_left() {
  digitalWrite(left_motor, LOW);
  digitalWrite(right_motor, HIGH);
}

void sharp_turn_left() {
  digitalWrite(left_motor, LOW);
  digitalWrite(right_motor, HIGH);
}

void sharp_turn_right() {
  digitalWrite(left_motor, HIGH);
  digitalWrite(right_motor, LOW);
}

