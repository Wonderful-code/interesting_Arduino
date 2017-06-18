#include <Servo.h>  
  
#define PIN_SERVO 10  
Servo myservo;  
int val;
void setup()  
{  
  myservo.attach(PIN_SERVO);
  Serial.begin(9600);//设置波特率为9600
  //Serial.println("servo=o_seral_simple ready" ); 
  myservo.write(90);//初始角度
}  
  
void loop() 
{  
  
  val = Serial.read(); //读取串口收到的数据
  if (val > '0' && val <= '9') //判断收到数据值是否符合范围
  {
     val = val - '0';
     val = val * (180/9); //将数字转化为角度
     Serial.print("moving servo to ");
     Serial.print(val, DEC);
     Serial.println();
    myservo.write(val);
    delay(1000);
  }
  
}
