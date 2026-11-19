# Recalibration study results

The simulation scores 2,784,000 distinct shots. The two window sizes reuse each stream. Each decoder sees the same shots. Independent trajectories are the units of inference.

The primary mean difference is -0.0150 percentage points. The pointwise 95% interval is [-0.0337, +0.0037]. The raw paired p-value is 0.11226, Holm-adjusted p is 0.33678, and the two-sided signed-rank sensitivity p is 0.13122. The predefined success rule returned **False**.

## Simulated logical errors

Fits count graph rebuilds for selective and periodic policies, respectively. The matched model knows the generating noise law.

|Distance|Basis|Window|Regime|Seeds|Shots|Factory|Raw|Shrink|Rolling|Periodic|Selective|Historical|Matched|Fits S/P|
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
|3|X|200|bias|10|120000|5136|8022|5257|5190|5265|5266|6253|4908|13/20|
|3|X|200|hotspot|10|120000|2382|4619|2228|2184|2318|2227|3202|2125|14/20|
|3|X|200|stationary|10|120000|1586|3605|1594|1576|1611|1586|1586|1586|0/20|
|3|X|200|uniform|10|120000|2302|4787|2394|2379|2392|2322|2541|2307|15/20|
|3|X|2000|bias|10|120000|5136|5234|5053|5104|5139|5090|5170|4908|10/20|
|3|X|2000|hotspot|10|120000|2382|2640|2245|2283|2304|2255|2562|2125|10/20|
|3|X|2000|stationary|10|120000|1586|1746|1576|1563|1596|1586|1586|1586|0/20|
|3|X|2000|uniform|10|120000|2302|2495|2302|2289|2308|2306|2375|2307|10/20|
|3|Z|200|bias|30|360000|4102|10786|4282|4170|4253|4170|5879|4102|36/60|
|3|Z|200|hotspot|30|360000|6750|14081|6461|6332|6657|6277|9553|6135|39/60|
|3|Z|200|stationary|30|360000|4042|10604|4209|4183|4188|4042|4042|4042|1/60|
|3|Z|200|uniform|30|360000|6126|13831|6390|6284|6412|6171|6753|6127|45/60|
|3|Z|2000|bias|30|360000|4102|4565|4118|4122|4103|4125|4379|4102|30/60|
|3|Z|2000|hotspot|30|360000|6750|7554|6428|6443|6460|6406|7281|6135|30/60|
|3|Z|2000|stationary|30|360000|4042|4426|4051|4027|4025|4042|4042|4042|1/60|
|3|Z|2000|uniform|30|360000|6126|6580|6134|6120|6112|6112|6235|6127|30/60|
|5|Z|200|bias|12|144000|1112|3459|1226|1176|1172|1125|1718|1112|18/24|
|5|Z|200|hotspot|12|144000|1431|3859|1419|1397|1494|1347|2355|1302|16/24|
|5|Z|200|stationary|12|144000|1077|3312|1155|1119|1147|1077|1077|1077|0/24|
|5|Z|200|uniform|12|144000|2123|5621|2456|2351|2407|2231|2577|2146|15/24|
|5|Z|2000|bias|12|144000|1112|1253|1113|1108|1113|1123|1211|1112|12/24|
|5|Z|2000|hotspot|12|144000|1431|1665|1378|1408|1421|1383|1570|1302|12/24|
|5|Z|2000|stationary|12|144000|1077|1235|1080|1079|1081|1078|1077|1077|1/24|
|5|Z|2000|uniform|12|144000|2123|2422|2162|2147|2150|2149|2268|2146|12/24|
|7|Z|200|bias|6|72000|312|1356|356|322|326|307|572|312|8/12|
|7|Z|200|hotspot|6|72000|382|1458|415|406|414|387|825|352|9/12|
|7|Z|200|stationary|6|72000|290|1315|334|317|332|290|290|290|1/12|
|7|Z|200|uniform|6|72000|1050|3082|1261|1183|1213|1190|1497|1060|10/12|
|7|Z|2000|bias|6|72000|312|359|298|311|306|306|335|312|6/12|
|7|Z|2000|hotspot|6|72000|382|469|388|381|378|376|435|352|6/12|
|7|Z|2000|stationary|6|72000|290|365|305|294|288|290|290|290|0/12|
|7|Z|2000|uniform|6|72000|1050|1267|1089|1061|1077|1079|1185|1060|6/12|

## Exploratory single-fit comparison

These reuse the primary hotspot trajectories. Each arm rebuilds once per seed. The independent fixed schedule fits at the midpoint. The alarm schedule uses the original causal alarm-entry decision. Restoring the cached graph has a cost and requires no new fit.

|Timing and retention|Total logical errors|
|---|---:|
|midpoint|6855|
|midpoint one window|6618|
|alarm hold|6437|
|alarm one window|6493|
|alarm with observed clearance|6406|

## Independent follow-up on one update

Both methods fit exactly once, use the same estimator and retain its graph for one window. The fixed update time was learned on old trajectories and frozen before new seeds ran. Distance three is primary and distance five is descriptive. The minimum useful absolute improvement was 0.060 percentage points. The follow-up success rule returned **False**.

|Distance|Seeds|Shots|Fixed errors|Alarm errors|Difference (pp)|95% CI (pp)|Paired p|
|---:|---:|---:|---:|---:|---:|---|---:|
|3|90|1,080,000|20164|20025|-0.01287|[-0.02167, -0.00407]|0.00460|
|5|45|540,000|5313|5257|-0.01037|[-0.01869, -0.00205]|0.01579|

## Offline experimental replay

Each basis contributes 44,000 held-out shots from one Google device placement. The acquisition timestamps are unknown. Alignment is a post hoc implementation of the first DGR stage. Publisher predictions are included in the released data.

|Basis|Window|Factory|Always shrink|Periodic|Selective|DGR alignment|Publisher correlated|
|---|---:|---:|---:|---:|---:|---:|---:|
|X|200|4856|4777|4783|4850|14376|4086|
|X|2000|4856|4555|4676|4680|4590|4086|
|Z|200|4046|4090|4140|4046|11724|3419|
|Z|2000|4046|3906|3927|4039|4143|3419|

## Additional same-shot physical decoders

|Implementation or released prediction|X errors|Z errors|
|---|---:|---:|
|local uncorrelated|4856|4046|
|local correlated|4205|3607|
|correlated matching decoder with rl optimized prior|3643|2960|
|correlated matching decoder with si1000 prior|4086|3419|
|harmony decoder with rl optimized prior|3517|2899|
|harmony decoder with si1000 prior|3889|3089|

The publisher reports tuning the optimized priors on 13-cycle calibration data. Their independence from these rows is unverified. Both local PyMatching modes were run directly.

Per-window errors, fit and reset decisions, host timing samples, and scored public files are in data/. The paper tables include the host timing distribution. The historical IBM summaries lack raw registers. The active study draws no IBM logical performance or feedback latency conclusion.
