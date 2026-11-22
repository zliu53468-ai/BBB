/**
 * Independent educational Monte Carlo demonstration.
 * 200 paths x 500 sampled hands = exactly 100,000 pseudo-random outcomes.
 * NOT actual historical hands, NOT BBB Soft Shoe results, NOT a universal ruin rate.
 * Run: node research/martingale-100k.mjs
 */
export const SETTINGS=Object.freeze({
  seed:20261122,paths:200,roundsPerPath:500,initialBankroll:100,
  baseStake:1,maxStake:64,bankerWinProbability:0.458597,
  playerWinProbability:0.446247,tieProbability:0.095156,
  bankerProfitMultiplier:0.95
});
export function runDemo(config=SETTINGS){
  const {seed,paths,roundsPerPath,initialBankroll,baseStake,maxStake,
    bankerWinProbability:pb,playerWinProbability:pp,bankerProfitMultiplier:profit}=config;
  let state=seed>>>0,stops=0,stopByLimit=0,stopByBankroll=0,
    belowBase=0,betsPlaced=0;
  const maxDrawdowns=[],endingBalances=[];
  for(let path=0;path<paths;path++){
    let bankroll=initialBankroll,peak=bankroll,required=baseStake,
      stopped=false,maxDrawdown=0;
    for(let hand=0;hand<roundsPerPath;hand++){
      // Draw an outcome even after strategy is stopped: constant 200*500 sample size.
      state=(Math.imul(1664525,state)+1013904223)>>>0;
      const random=state/4294967296;
      if(stopped)continue;
      if(required>maxStake||bankroll<required){
        stopped=true;stops++;
        if(required>maxStake)stopByLimit++;else stopByBankroll++;
        continue;
      }
      bankroll-=required;betsPlaced++;
      if(random<pb){
        bankroll+=required*(1+profit);
        required=baseStake;
      }else if(random<pb+pp){
        required*=2;
      }else{
        bankroll+=required; // Tie: banker wager is pushed.
      }
      peak=Math.max(peak,bankroll);
      maxDrawdown=Math.max(maxDrawdown,(peak-bankroll)/peak);
    }
    if(!stopped&&(required>maxStake||bankroll<required)){
      stops++;
      if(required>maxStake)stopByLimit++;else stopByBankroll++;
    }
    if(bankroll<baseStake)belowBase++;
    maxDrawdowns.push(maxDrawdown);
    endingBalances.push(bankroll);
  }
  maxDrawdowns.sort((a,b)=>a-b);
  endingBalances.sort((a,b)=>a-b);
  const median=arr=>(arr[(arr.length-1)>>1]+arr[arr.length>>1])/2;
  return Object.freeze({
    ...config,totalRandomOutcomes:paths*roundsPerPath,betsPlaced,
    strategyStops:stops,stopByLimit,stopByBankroll,
    strategyStopFraction:stops/paths,
    belowBase,belowBaseFraction:belowBase/paths,
    medianMaxDrawdown:median(maxDrawdowns),
    p90MaxDrawdown:maxDrawdowns[Math.ceil(paths*.9)-1],
    medianEndingBalance:median(endingBalances)
  });
}
if(process.argv[1]&&import.meta.url===new URL("file://"+process.argv[1]).href){
  console.log(JSON.stringify(runDemo(),null,2));
}
