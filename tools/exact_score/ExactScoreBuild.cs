using System;
using System.IO;
using System.Collections.Generic;
using System.Globalization;
public static class ExactScoreBuild {
 sealed class State {
  public int Key,N; public int[] M=new int[100], C=new int[3];
  public int[] R,K; public double BaseMass;
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
  states.Clear();Array.Clear(used,0,used.Length);paths=0;
  Enumerate();var ss=new List<State>(states.Values);ss.Sort((x,y)=>x.Key.CompareTo(y.Key));
  int[] original={128,32,32,32,32,32,32,32,32,32};
  foreach(var s in ss){
   var rs=new List<int>();var ks=new List<int>();double mass=1;
   for(int r=0;r<10;r++){int k=(s.Key>>(3*r))&7;if(k==0)continue;rs.Add(r);ks.Add(k);for(int j=0;j<k;j++)mass*=original[r]-j;}
   s.R=rs.ToArray();s.K=ks.ToArray();s.BaseMass=mass/Denom(416,s.N);
  }
  double[,] joint=new double[100,3],length=new double[100,3];double[,,] lengthJoint=new double[100,3,3];
  double[] evidence=new double[100],gap2=new double[100];
  int done=0;
  foreach(var current in ss){
   var falling=new double[10,7];
   for(int r=0;r<10;r++){int n=original[r]-((current.Key>>(3*r))&7);falling[r,0]=1;for(int k=1;k<=6;k++)falling[r,k]=falling[r,k-1]*(n-k+1);}
   double[] denominators={Denom(416-current.N,4),Denom(416-current.N,5),Denom(416-current.N,6)};
   double[] next=new double[3],nextLength=new double[3];
   foreach(var terminal in ss){
    double mass=1;for(int j=0;j<terminal.R.Length;j++)mass*=falling[terminal.R[j],terminal.K[j]];
    mass/=denominators[terminal.N-4];
    for(int o=0;o<3;o++)next[o]+=mass*terminal.C[o];
    nextLength[terminal.N-4]+=mass*(terminal.C[0]+terminal.C[1]+terminal.C[2]);
   }
   double total=next[0]+next[1]+next[2];if(Math.Abs(total-1)>1e-10)throw new Exception("next mass "+total);
   double gap=1.95*next[0]-2*next[1];
   for(int score=0;score<100;score++){
    if(current.M[score]==0)continue;
    double w=current.BaseMass*current.M[score];evidence[score]+=w;
    gap2[score]+=w*gap*gap;length[score,current.N-4]+=w;
    for(int o=0;o<3;o++){joint[score,o]+=w*next[o];lengthJoint[score,current.N-4,o]+=w*nextLength[o];}
   }
   done++;if(done%1000==0)Console.WriteLine("Exact states "+done+"/"+ss.Count);
  }
  using(var f=new StreamWriter(output)){
   f.WriteLine("player\tbanker\tevidence\tB\tP\tT\tposterior_gap_sd\tL4\tL5\tL6\tJ44\tJ45\tJ46\tJ54\tJ55\tJ56\tJ64\tJ65\tJ66");
   for(int score=0;score<100;score++){
    double e=evidence[score],b=joint[score,0]/e,p=joint[score,1]/e,gap=1.95*b-2*p;
    var values=new List<double>{score/10,score%10,e,b,p,joint[score,2]/e,Math.Sqrt(Math.Max(0,gap2[score]/e-gap*gap))};
    for(int k=0;k<3;k++)values.Add(length[score,k]/e);
    for(int k=0;k<3;k++)for(int j=0;j<3;j++)values.Add(lengthJoint[score,k,j]/e);
    f.WriteLine(string.Join("\t",values.ConvertAll(v=>v.ToString("R",CultureInfo.InvariantCulture))));
   }
  }
  Console.WriteLine("Finished exact legal paths="+paths+" merged states="+ss.Count);
 }
}
