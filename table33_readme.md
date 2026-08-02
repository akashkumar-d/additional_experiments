# The 33-configuration grid: complete results

**33 configurations × 50 seeds = 1,650 runs.** Every numeric column is the
**median over the 50 seeds**. The one exception is `cm_end`, which is written
`median ± sd` (standard deviation over seeds). Per-run values for all 1,650 runs
are released as CSV so any other statistic can be recomputed.

Setup: teacher–student, full-batch **exact-polar** matrix Muon, fixed learning
rate, no schedule. Teachers span ReLU (12 configurations), GELU (11) and SiLU
(10); input dimension $p \in \{100, 150, 200\}$, teacher rank
$r_t \in \{4,6,8,10,12,16,20\}$, student rank $r_s \in \{30,50,75,80,100\}$.

---

## Columns

| column | meaning |
|---|---|
| `id` | grid identifier (R = ReLU, G = GELU, S = SiLU) |
| `act`, `p`, `rt`, `rs`, `eta` | activation, input dim, teacher rank $r_t$, student rank $r_s$, learning rate |
| `sds` | seeds completed (50 everywhere) |
| `div` | seeds flagged by the amplitude check — **see the note below, this is not divergence** |
| `1-rho2` | $1-\rho_2$, where $\rho_2 = -\mathrm{Corr}(\Delta L_t, \Delta L_{t+1})$ on the dense per-step loss. **Smaller is a tighter period-2 lock**; $10^{-10}$ means successive loss increments alternate almost exactly |
| `ratio` | $\bar L / L_{V,\mathrm{opt}}$ at the end of training: achieved cycle-mean loss divided by what the *current* features already support after optimally refitting the readout. **1 = the features are fully exploited** |
| `t_plat` | plateau onset, first step after which the cycle-mean loss never again falls faster than 2% per 500 steps |
| `t_sat` | saturation onset for mean alignment |
| `gap_sat` | median $\bar L / L_{V,\mathrm{opt}}$ over $[t_{\mathrm{sat}}, T]$ — the same ratio, measured over the saturated window instead of at the endpoint |
| `dmean_pl` | change in mean $\cos^2$ **during the plateau window** $[t_{\mathrm{plat}}, T]$ |
| `dmin_pl` | change in $\cos^2_{\min}$ over the same window — the strict metric, requiring **every** teacher direction to be recovered |
| `cm_end` | final $\cos^2_{\min}$, median ± sd over seeds |
| `t50` | steps for $\cos^2_{\min}$ to reach 0.5; `nan` if it never does |
| `tauV2` | $\tau_V^2 = r_t - \lVert P_g P_V \rVert_F^2 = \lVert (I - P_g)P_V \rVert_F^2$, the part of the teacher subspace *not* captured by the polar update direction (Lemma quantity) |
| `verdict` | outcome class, assigned from `dmin_pl`, `ratio` and the plateau test |

$\cos^2_{\min}$ and mean $\cos^2$ are principal-angle alignments between the
student's top-$r_t$ AGOP eigenspace and the known teacher subspace. Chance level
for mean $\cos^2$ is $r_t/p$.

---

```
id   act    p  rt  rs   eta sds div    1-rho2    ratio  t_plat  t_sat  gap_sat  dmean_pl  dmin_pl           cm_end    t50  tauV2  verdict
-------------------------------------------------------------------------------------------------------------------------------------------
  R1  relu  100   4  50   2.0  50  50   4.2e-07   3027.5    2178    620   2699.0    -0.014   -0.065    0.658+/-0.027      320  0.154  saturates early, gap persists
  R2  relu  100   6  50   1.5  50   0   2.5e-08    218.8     501   4820    202.2    +0.389   +0.930    0.957+/-0.005     1310  0.129  clean plateau-with-learning
  R3  relu  100   8  50   1.5  50   0   1.2e-10     38.4     501   6790     39.4    +0.461   +0.896    0.908+/-0.012     2920  0.219  clean plateau-with-learning
  R4  relu  100   8  50   1.0  50   0   1.0e-10     13.7     505   7070     14.1    +0.343   +0.816    0.856+/-0.034     3540  0.187  clean plateau-with-learning
 R10  relu  100   8  50  1.25  50   0   1.0e-10     25.6     502   7110     26.1    +0.409   +0.841    0.862+/-0.024     3720  0.207  clean plateau-with-learning
  R5  relu  100   8  30   1.5  50   0   1.0e-10     16.0     559   7240     16.2    +0.401   +0.823    0.838+/-0.042     4350  0.263  clean plateau-with-learning
 R11  relu  100   8  80   1.5  50  50   1.4e-09    187.1     500   5150    176.7    +0.370   +0.928    0.944+/-0.007     1490  0.079  clean plateau-with-learning
 R12  relu  100  10  50   1.5  50   0   1.0e-10     20.1     503   6860     20.4    +0.289   +0.827    0.863+/-0.016     3060  0.282  clean plateau-with-learning
  R6  relu  100  12  50   1.5  50   0   1.0e-10     23.3     510   5450     23.3    +0.056   +0.309    0.826+/-0.083      470  0.329  clean plateau-with-learning
 R13  relu  150   8  50   1.5  50   0   2.6e-10     26.3     502   5600     26.5    +0.268   +0.686    0.924+/-0.007      890  0.382  clean plateau-with-learning
  R7  relu  100  16  50   1.5  50   0   1.9e-10     11.7     563   7700     12.0    +0.223   +0.023    0.024+/-0.062      nan  0.544  weakest direction not recovered
  R9  relu  100  20  50   2.0  50  50   1.6e-10     16.0     545   7720     16.3    +0.221   +0.021    0.023+/-0.057      nan  0.743  weakest direction not recovered
 G14  gelu  100   8  50   0.6  50   0   2.2e-09      5.1     644   5330      5.1    +0.201   +0.771    0.961+/-0.007     1400  0.128  clean plateau-with-learning
  G6  gelu  100   8  50   0.5  50   0   5.4e-09      4.1     813   4870      4.1    +0.124   +0.483    0.967+/-0.005     1120  0.121  clean plateau-with-learning
 G13  gelu  100  10  50   0.7  50   0   2.5e-10      4.3     546   6330      4.3    +0.207   +0.836    0.907+/-0.019     2360  0.193  clean plateau-with-learning
  G7  gelu  100  12  50   0.6  50   0   3.9e-10      3.3     776   6960      3.3    +0.241   +0.790    0.833+/-0.050     3670  0.262  clean plateau-with-learning
  G5  gelu  100  12  50   0.7  50   0   2.4e-10      4.0     550   6450      4.0    +0.145   +0.741    0.854+/-0.042     2730  0.248  clean plateau-with-learning
 G12  gelu  100  12  80   0.7  50   0   1.7e-10      8.4     520   7480      8.4    +0.246   +0.575    0.607+/-0.197     6460  0.103  clean plateau-with-learning
 G10  gelu  100  16  50  0.85  50   0   1.4e-10      4.0     635   7520      4.0    +0.230   +0.438    0.450+/-0.184     7040  0.440  clean plateau-with-learning
  G2  gelu  100   6  50   0.7  50   0   9.3e-08      7.5       0   2950      8.0    +0.932   +0.989    0.989+/-0.002       70  0.079  clean plateau-with-learning
 G15  gelu  200  12  50   0.7  50   0   3.4e-09      1.9    1980   5350      1.9    +0.042   +0.100    0.920+/-0.006      480  0.699  modest plateau (2-3x oracle)
  G4  gelu  100   8  30   0.7  50   0   3.1e-09      3.0    2455   3210      3.1    +0.013   +0.029    0.976+/-0.003      120  0.187  no gain in plateau window
 G16  gelu  100   8 100   0.7  50   0   1.0e-08     23.0       0   6260     22.8    +0.900   +0.965    0.966+/-0.007     1400 -0.000  clean plateau-with-learning
  S4  silu  100   8  30  0.85  50   0   8.2e-09      2.5    6436   3810      2.5    +0.003   +0.006    0.977+/-0.003      570  0.208  modest plateau (2-3x oracle)
 S13  silu  100  10  50  0.85  50   0   1.5e-09      4.9     504   5420      4.9    +0.215   +0.866    0.949+/-0.006     1580  0.203  clean plateau-with-learning
  S5  silu  100  12  50  0.85  50   0   5.6e-10      4.2     538   6590      4.2    +0.282   +0.872    0.891+/-0.020     3100  0.267  clean plateau-with-learning
  S8  silu  100  12  50   1.0  50   0   6.3e-10      5.3     518   6200      5.3    +0.284   +0.902    0.922+/-0.012     2520  0.262  clean plateau-with-learning
 S10  silu  100  16  50   1.0  50   0   1.4e-10      3.8     510   7060      3.9    +0.232   +0.770    0.787+/-0.034     4440  0.453  clean plateau-with-learning
 S11  silu  100  20  50  0.85  50   0   1.1e-10      3.3     655   7530      3.4    +0.198   +0.318    0.328+/-0.177     7540  0.620  clean plateau-with-learning
  S3  silu  100   8  50  0.85  50   0   1.3e-08      5.5       0   3860      5.7    +0.905   +0.975    0.976+/-0.003      390  0.159  clean plateau-with-learning
 S14  silu  200  12  80  0.85  50   0   1.1e-08      5.0       0   6440      4.9    +0.885   +0.911    0.914+/-0.005     1580  0.600  clean plateau-with-learning
 S15  silu  100  12  75  0.85  50   0   5.4e-10     10.1     503   6500     10.1    +0.258   +0.886    0.907+/-0.012     2720  0.141  clean plateau-with-learning
 S12  silu  100  12  80  0.85  50   0   5.2e-10     11.6     502   6720     11.5    +0.320   +0.888    0.901+/-0.011     3010  0.116  clean plateau-with-learning
```

---

## Reading the table

**The claim is `dmin_pl` > 0 while the loss is on a plateau.** Across the grid,
$\cos^2_{\min}$ gains a median of **+0.816** during the plateau window
(IQR +0.438 to +0.888); it improves in **32 of 33** configurations and by more
than 0.3 in **27 of 33**.

**`ratio` is the scale against which "the loss is flat" should be read.**
Median **7.5**, IQR 4.1–20.1, range 1.9–3028; only one configuration (G15) sits
below 2. The residual loss motion during the plateau therefore happens
several-fold above what the representation already permits, not in the final
tail of convergence.

**`1-rho2` confirms the mechanism is present everywhere**, spanning
$5.5\times10^{-11}$ to $4.2\times10^{-7}$ — the loss is in a locked period-2
cycle in every configuration, including those where alignment does not improve.
The orbit and the feature learning are separate facts.

### Outcome classes

| verdict | n | what it means |
|---|---|---|
| clean plateau-with-learning | 27 | loss plateaus well above the oracle, $\cos^2_{\min}$ keeps rising |
| modest plateau (2–3× oracle) | 2 | plateau present but close to the oracle, so little room to improve |
| no gain in plateau window | 1 | plateau present, alignment already complete before it |
| weakest direction not recovered | 2 | mean alignment rises but $\cos^2_{\min}$ stays near 0 |
| saturates early, gap persists | 1 | alignment peaks and then **declines** during the plateau |

---

## Rows that need explanation

**`div = 50` on R1, R9 and R11 is not divergence.** These are
**high-amplitude locked cycles**: the cycle-mean loss sits above the flag's
absolute threshold, but the orbit is intact. R11 has $1-\rho_2 = 1.4\times10^{-9}$
and gains $\Delta\cos^2_{\min} = +0.928$ — it is one of the strongest
plateau-with-learning cases in the grid. The column flags amplitude, not
instability, and will be renamed in the revision.

**`t_plat = 0` on G2, G16, S3, S14** means the plateau rule fires at the first
step: these configurations enter the oscillatory regime immediately, so the whole
run is the plateau window. Their `dmin_pl` values (+0.911 to +0.989) are
correspondingly large because the window covers all of training.

**`t50 = nan` on R7 and R9** means $\cos^2_{\min}$ never reaches 0.5. Both end at
0.024 and 0.023 — the weakest teacher direction is never recovered. These are the
two negative configurations and they are reported as such.

**`t_plat > t_sat` on R1 and S4** means alignment saturated *before* the loss
plateaued, so their plateau windows contain little remaining feature motion. This
is why S4 gains only +0.006 and why R1, whose alignment peaks at 0.769 around
step 540 and then falls to 0.658, is classified "saturates early, gap persists".

**Seed dispersion is concentrated, not uniform.** The median sd of `cm_end` is
**0.012**; three partial-recovery configurations — G12 (±0.197), G10 (±0.184),
S11 (±0.177) — carry almost all of the variance. Dispersion appears exactly
where the outcome is marginal, which is where it should be.

