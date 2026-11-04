using System;
using System.IO;
using System.Globalization;
public static class MillionScoreValidation {
 sealed class Metric {public long Bets,Wins,Losses,Ties;public double Profit,Profit2;}
 struct Hand {public int P,B;}
 // Separate validation implementation: draw physical cards with partial
 // Fisher-Yates, rather than use the enumerator's multiset polynomials.
 static int[] deck=new int[416];static int pos;static Random rng;
 static int Draw(){int at=rng.Next(pos,416);int v=deck[at];deck[at]=deck[pos];deck[pos]=v;pos++;return v;}
 static Hand Deal(){
  int p1=Draw(),b1=Draw(),p2=Draw(),b2=Draw();int p=(p1+p2)%10,b=(b1+b2)%10;
  if(p<8&&b<8){
   if(p<=5){
    int x=Draw();p=(p+x)%10;
    if(b<=2||(b==3&&x!=8)||(b==4&&x>=2&&x<=7)||(b==5&&x>=4&&x<=7)||(b==6&&(x==6||x==7)))b=(b+Draw())%10;
   }else if(b<=5)b=(b+Draw())%10;
  }
  return new Hand{P=p,B=b};
 }
 static double Record(Metric m,string direction,Hand h){
  if(direction=="SKIP")return 0;
  m.Bets++;double gain=0;
  if(h.B==h.P)m.Ties++;
  else if((direction=="B"&&h.B>h.P)||(direction=="P"&&h.P>h.B)){m.Wins++;gain=direction=="B"?.95:1;}
  else {m.Losses++;gain=-1;}
  m.Profit+=gain;m.Profit2+=gain*gain;return gain;
 }
 public static void Run(string comparison,string output,int samples,int seed){
  if(samples!=1000000)throw new Exception("release validation requires exactly 1000000 independent pairs");
  rng=new Random(seed);var lines=File.ReadAllLines(comparison);var old=new string[100];var exact=new string[100];
  for(int i=1;i<lines.Length;i++){var a=lines[i].Split('\t');int k=int.Parse(a[0])*10+int.Parse(a[1]);old[k]=a[2];exact[k]=a[3];}
  string[] names={"legacy_display","exact_v4_display","exact_relative_candidate","always_banker","always_player","exact_candidate_on_legacy_entries"};
  var ms=new Metric[6];for(int i=0;i<6;i++)ms[i]=new Metric();
  var bins=new long[100,3];var original=new int[416];int n=0;
  for(int i=0;i<128;i++)original[n++]=0;
  for(int r=1;r<10;r++)for(int i=0;i<32;i++)original[n++]=r;
  double diff=0,diff2=0;var trace=new byte[samples*2];
  for(int i=0;i<samples;i++){
   Array.Copy(original,deck,416);pos=0;
   Hand current=Deal();int k=current.P*10+current.B;
   // Freeze all choices BEFORE drawing the target next hand.
   string oldChoice=old[k],newChoice=exact[k];
   Hand target=Deal();bins[k,target.B>target.P?0:target.P>target.B?1:2]++;
   trace[2*i]=(byte)k;trace[2*i+1]=(byte)(target.B>target.P?0:target.P>target.B?1:2);
   double oldGain=Record(ms[0],oldChoice,target);
   Record(ms[1],newChoice,target);double newGain=Record(ms[2],"B",target);
   Record(ms[3],"B",target);Record(ms[4],"P",target);
   double matchedGain=Record(ms[5],oldChoice=="SKIP"?"SKIP":"B",target);
   double d=matchedGain-oldGain;diff+=d;diff2+=d*d;
  }
  using(var f=new StreamWriter(output)){
   f.WriteLine("policy\tbets\twins\tlosses\tties\tprofit\tprofit2");
   for(int i=0;i<6;i++){var m=ms[i];f.WriteLine(names[i]+"\t"+m.Bets+"\t"+m.Wins+"\t"+m.Losses+"\t"+m.Ties+"\t"+m.Profit.ToString("R",CultureInfo.InvariantCulture)+"\t"+m.Profit2.ToString("R",CultureInfo.InvariantCulture));}
   f.WriteLine("paired_matched_candidate_minus_legacy\t"+samples+"\t0\t0\t0\t"+diff.ToString("R",CultureInfo.InvariantCulture)+"\t"+diff2.ToString("R",CultureInfo.InvariantCulture));
   f.WriteLine("BIN\tB\tP\tT");
   for(int k=0;k<100;k++)f.WriteLine(k+"\t"+bins[k,0]+"\t"+bins[k,1]+"\t"+bins[k,2]);
  }
  File.WriteAllBytes(output+".cases.bin",trace);
  Console.WriteLine("Completed "+samples+" independent two-hand scenarios; seed="+seed);
 }
}
