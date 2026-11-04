using System;
using System.IO;
using System.Collections.Generic;
using System.Globalization;
public static class KernelExport {
 sealed class State {
  public int Key,N; public int[] M=new int[100], C=new int[3];

 }
 static Dictionary<int,State> states=new Dictionary<int,State>();
 static int[] used=new int[10]; static int paths=0;
 static bool DrawB(int b,int p3) {
  if(p3<0)return b<=5;
  if(b<=2)return true;
  if(b==3)return p3!=8;
  if(b==4)return p3>=2&&p3<=7;
  if(b==5)return p3>=4&&p3<=7;
  return b==6&&(p3==6||p3==7);
 }
 static void Offer(int p,int b,int n) {
  int key=0;for(int i=0;i<10;i++)key|=used[i]<<(i*3);
  State s;if(!states.TryGetValue(key,out s)){s=new State();s.Key=key;s.N=n;states[key]=s;}
  s.M[p*10+b]++;s.C[b>p?0:p>b?1:2]++;paths++;
 }
 static void Enumerate(){
  for(int a=0;a<10;a++)for(int b=0;b<10;b++)for(int c=0;c<10;c++)for(int d=0;d<10;d++){
   used[a]++;used[b]++;used[c]++;used[d]++;
   int p=(a+c)%10,q=(b+d)%10;
   if(p>=8||q>=8)Offer(p,q,4);
   else if(p>5){
    if(!DrawB(q,-1))Offer(p,q,4);
    else for(int y=0;y<10;y++){used[y]++;Offer(p,(q+y)%10,5);used[y]--;}
   }else for(int x=0;x<10;x++){
    used[x]++;int fp=(p+x)%10;
    if(!DrawB(q,x))Offer(fp,q,5);
    else for(int y=0;y<10;y++){used[y]++;Offer(fp,(q+y)%10,6);used[y]--;}
    used[x]--;
   }
   used[a]--;used[b]--;used[c]--;used[d]--;
  }
 }
 static double Denom(int total,int n){double x=1;for(int i=0;i<n;i++)x*=total-i;return x;}
 public static void Run(string output){
 states.Clear();Array.Clear(used,0,used.Length);paths=0;Enumerate();
 using(var f=new StreamWriter(output)){
 f.Write("(function(r){const a=[");bool first=true;
 foreach(var s in states.Values){if(!first)f.Write(",");first=false;
 f.Write("["+s.Key+","+s.N+","+s.C[0]+","+s.C[1]+","+s.C[2]+"]");}
 f.Write("];if(typeof module==='object'&&module.exports)module.exports=a;else r.BBBCardKernel=a;})(typeof self!=='undefined'?self:this);\n");
 }
 }
}
