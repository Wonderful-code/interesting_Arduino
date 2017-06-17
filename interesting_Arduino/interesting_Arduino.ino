#include <Servo.h>  

#define PIN_N 9 
#define PIN_SERVO 10
  
Servo myservo0;
Servo myservo1;

int val0,val1;
int val;
String comdata = "";
//numdata是分拆之后的数字数组
int numdata[2] = {0};
int mark = 0;
void setup()  
{  
  myservo0.attach(PIN_SERVO);  // attaches the servo on pin 9 to the servo object 
  myservo1.attach(PIN_N);
  
  Serial.begin(9600);//设置波特率为9600
  //Serial.println("servo=o_seral_simple ready" ); 
  myservo0.write(90);//初始角度
  myservo1.write(160);//160~180
}  
  
void loop() 
{  
  //j是分拆之后数字数组的位置记数
  int j = 0;
  //不断循环检测串口缓存，一个个读入字符串，
  while (Serial.available() > 0)
  {
    //读入之后将字符串，串接到comdata上面。
    comdata += char(Serial.read());//读取串口收到的数据]
    //延时一会，让串口缓存准备好下一个数字，不延时会导致数据丢失，
     delay(2);
    //标记串口读过数据，如果没有数据的话，直接不执行这个while了。
    mark = 1;
   }
   
if(mark == 1){ //如果接收到数据则执行comdata分析操作，否则什么都不做。
  /*******************下面是重点*******************/
  //以串口读取字符串长度循环，
  for(int i = 0; i < comdata.length() ; i++){
  //逐个分析comdata[i]字符串的文字，如果碰到文字是分隔符（这里选择逗号分割）则将结果数组位置下移一位
  //即比如1,2开始的1记到numdata[0];碰到逗号就j等于1了，
  //再转换就转换到numdata[1];再碰到逗号就记到numdata[2];以此类推，直到字符串结束
  if(comdata[i] == ','){
    j++;
  }else{
  //(comdata[i] - '0')就是将字符'0'的ASCII码转换成数字0（下面不再叙述此问题，直接视作数字0）。
     numdata[j] = comdata[i] - '0';
    }
  }
  //comdata的字符串已经全部转换到numdata了，清空comdata以便下一次使用，
  //如果不请空的话，本次结果极有可能干扰下一次。
  comdata = String("");
  //输出numdata的内容，并且载入舵机
  Serial.println(numdata[0]);
  val0 = numdata[0];
  val1 = numdata[1];
    if (val0 > '0' && val0 <= '9' && val1 > '0' && val1 <= '9') //判断收到数据值是否符合范围
    {
      val0 = val0 * (180 / 9); //将数字转化为角度，例9*（180/9）=180
      val1 = val1 * (180 / 9);
      myservo0.write(val0);
      //myservo1.write(val1);
      delay(2000);
    }
}
  numdata[0] = 0;
  numdata[1] = 0;
  //输出之后必须将读到数据的mark置0，不置0下次循环就不能使用了。
  mark = 0;
}

