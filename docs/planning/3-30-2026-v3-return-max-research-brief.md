# 3-30-2026 v3 return max research brief

## What we researched

- Higher-confidence crypto quant fundamentals
- Lower-confidence practitioner / X-style heuristics
- A concrete roadmap for a higher-return `v3` branch

## Main conclusion

The research does **not** support "more generic ML" as the best next step.

The strongest path is:
- keep trend / regime logic as the base
- port the strongest aggressive `v2` logic into `v3`
- add market-structure data next
- keep funding and premium
- optimize for return under a drawdown cap

## Important framing

- Current robust `v3` result of about `+1.02%` monthly is believable but not enough for the target
- A realistic ambitious next target is roughly `3%` to `5%` monthly unlevered
- The `10%` monthly target is more credible as a stretch goal under disciplined `2x` leverage than as an unlevered target

## Research artifacts added

- `research/08-quant-fundamentals.md`
- `research/09-practitioner-scan.md`
- `research/10-return-max-v3-roadmap.md`

## Next development step

Do not start with sentiment.

Start by:
1. porting the strongest aggressive `v2` hybrid / trend config into the strict `v3` walk-forward harness
2. keeping Deribit funding and Coinbase premium
3. reporting results at `1x` and `2x`
