# DE_Compare — több single-cell DE-módszer összehasonlítása

Egyetlen script (`run_de_comparison.py`), ami **ugyanazon az adaton**
(Reference vs pancreas) lefuttat több differenciális génexpressziós (DE)
módszert, **módszerenként egy harmonizált eredményfájlt** ír ki (azonos
oszlopfejléccel), majd **génenként összehasonlítja** őket: melyik algoritmus mit
adott az adott génre, és mennyire konzisztensek egymással.

> **Névváltozás (2026-07-21): `scPyDE` → `DUET`.**
> A `scPyDE` név „Python SCDE"-ként olvasódott, márpedig az
> [SCDE](https://bioconductor.org/packages/release/bioc/html/scde.html)
> (Kharchenko et al., Nat Methods 2014) egy létező, de **statisztikailag más**
> módszer — Bayes-féle dropout hibamodell, nem hurdle. Az algoritmus valójában a
> **MAST** két-részes hurdle modelljének újraimplementációja, ezért az új név:
> **DUET** = **D**etection–**E**xpression **U**nified **T**est (a két komponens
> „két hang, egy darab"). A csomag `CustomDE/scpyde.py` → **`duet/core.py`**;
> a `CustomDE` import továbbra is működik (deprecation figyelmeztetéssel), a
> `run_scpyde` a `run_duet` aliasa maradt.

## Módszerek

| Módszer | Mit csinál | Egység | Megjegyzés |
|---|---|---|---|
| **DUET** | a `duet/core.py` MAST-szerű két-részes hurdle modell | sejtenkénti | `mast_compat=True, eb_shrinkage=True, vectorized` |
| **diffxpy** | Wald-teszt, negatív-binomiális GLM (`~1+condition`) | sejtenkénti | nyers count-okon; lassú (TensorFlow) |
| **DESeq2** | `pydeseq2` **pseudobulk** (count-ok összegezve **`SourceFile` × kondíció**-ként) | minta-szintű | valódi replikáció → kevesebb „szignifikáns" gén |
| **Wilcoxon** | `scanpy.tl.rank_genes_groups(method='wilcoxon')` | sejtenkénti | log-normalizált `X`-en |
| **MAST** | **R MAST 1.33.0** (natív R 4.5.0) | sejtenkénti | `run_mast_dataset.py` futtatja azonos sejteken/géneken; `add_mast_and_rebuild.py` előre kiszámolt CSV-t olvas be |

> A „valódi" DESeq2 (R) nincs telepítve → **pydeseq2** (tiszta Python) helyettesíti.
>
> **R MAST 5. módszerként:** `python run_mast_dataset.py --h5ad <h5ad> --results-dir <dir>
> --ref-label <ref> --test-label <test>` — ez a DUET által tesztelt géneken,
> `~1+condition` modellel futtatja, majd `python rebuild_comparison.py --results-dir <dir>`.
> (A régi út: előre kiszámolt CSV + `add_mast_and_rebuild.py`.)

## Harmonizált kimeneti séma (minden módszer ugyanazt adja)

```
method, celltype, comparison, gene, pvalue, fdr, log2fc, neglog10p, stat,
n_ref, n_test, tested, skip_reason
```

- `comparison` = `"<ref>_vs_<test>"` a tényleges címkékből (pl. `Reference_vs_pancreas`, `ctrl_vs_stim`, `Vehicle_vs_LPS`); **pozitív `log2fc` = a test csoport felé up-regulált**.
- `neglog10p` = **alulcsordulás-biztos** −log10(p) → **rangsoroláshoz EZT használd, ne a `pvalue`-t** (lásd lent).
- `tested` = `True`, ha a gént ténylegesen tesztelte a módszer; különben `skip_reason` mondja meg, miért nem.
- Módszerenkénti fájl: `results/de_<method>_all_celltypes.csv` (pl. `de_DUET_all_celltypes.csv`).

## Bemenet

- Alapból: `DE_Compare/PancreasIntegratedAnnotated.h5ad` (229 262 sejt × 59 537 gén, 61 donor).
- Elvárások az AnnData-tól:
  - `layers['counts']` = nyers count-ok (kell a DESeq2-höz és a diffxpy-hez),
  - `X` = log-normalizált (kell a Wilcoxonhoz),
  - `obs['Sample']` = `Reference` / `pancreas` (a két csoport),
  - `obs['celltype']` = sejttípusok,
  - `obs['SourceFile']` = egyedi minták (a DESeq2 pseudobulk replikátumai).

## Használat

**Gyors füstteszt** (1 sejttípus, a 2000 legmagasabban expresszált gén — pár perc):

```powershell
python run_de_comparison.py --max-genes 2000 --n-celltypes 1
```

**Teljes futás** (alapból 3 kiegyensúlyozott sejttípus, az összes gén — kb. 45 perc):

```powershell
python run_de_comparison.py
python add_mast_and_rebuild.py      # R MAST behúzása 5. módszerként
python make_volcano.py --log-y      # vulkán-ábrák (LOG y-tengely, lásd lent)
python make_heatmaps.py
```

### Flagek

| Flag | Alap | Leírás |
|---|---|---|
| `--h5ad` | `./PancreasIntegratedAnnotated.h5ad` | bemeneti AnnData |
| `--output-dir` | `./results` | kimeneti mappa |
| `--celltypes` | (auto) | vesszővel elválasztott lista; ha üres → auto-választás |
| `--n-celltypes` | `3` | hány sejttípust válasszon automatikusan (a legkiegyensúlyozottabbak) |
| `--methods` | `duet,diffxpy,deseq2,wilcoxon` | futtatandó módszerek (`scpyde` = elavult alias a `duet`-re) |
| `--duet-min-detect-frac` | `0.10` | DUET min.pct szűrő (lásd lent); `0` = teljes hurdle |
| `--duet-linear-logfc` | ki | DUET LINEÁRIS (scanpy) log2FC; alapból TÖMÖRÍTETT (MAST-szerű) |
| `--fdr` | `0.05` | szignifikancia-küszöb (FDR) |
| `--top-k` | `100` | a top-K átfedéshez (**`neglog10p` szerint rangsorolva**) |
| `--max-genes` | (nincs) | **csak füstteszt**: a top-N legmagasabban expresszált génre korlátoz |
| `--diffxpy-max-cells-per-group` | `5000` | diffxpy dense mátrixot kíván → RAM-korlát (`0` = összes sejt) |
| `--diffxpy-max-genes` | `8000` | diffxpy csak a top-N gént teszteli (`0` = összes gén) |
| `--diffxpy-nproc` | `1` | diffxpy worker-processzek száma (`1` = legkisebb RAM) |

A régi `--scpyde-*` flagnevek aliasként továbbra is elfogadottak.

## Kimenetek (`results/`)

- `de_DUET_all_celltypes.csv`, `de_MAST_*`, `de_diffxpy_*`, `de_DESeq2_*`, `de_wilcoxon_*`
- `comparison_per_gene.csv` — soronként egy `(celltype, gene)`, módszerenként `pvalue_*`, `fdr_*`, `log2fc_*`, `neglog10p_*`
- `comparison_concordance.csv` — páronkénti egyezés sejttípusonként
- `consensus_genes.csv`, `comparison_summary.txt`
- `volcano_*.png` / `volcano_*_logy.png`, `heatmap_*.png`

---

## Több adatszett (validáció)

A pipeline három adatszetten futott le, **azonos előfeldolgozással**
(nyers count a `layers['counts']`-ban, log1p CP10K az `X`-ben), hogy a
normalizálás ne konfundáljon:

| Adatszett | Sejt × gén | Faj / szövet | Kontraszt | Minta | Terv |
|---|---:|---|---|---:|---|
| Pancreas *(eredeti)* | 229 262 × 59 537 | humán hasnyálmirigy (PDAC) | Reference vs pancreas | 61 donor | donorok közti, **konfundált** |
| **Kang 2018** | 24 673 × 15 706 | humán PBMC | ctrl vs **IFN-β** | 8 páciens | **párosított**, multiplexált pool |
| **Crowell 4v4** | 25 224 × 11 076 | **egér** kéreg | Vehicle vs **LPS** | 4 + 4 egér | páratlan, kiegyensúlyozott |

Beszerzés: `build_kang_h5ad.py` (scverse example data) és `get_crowell.R` +
`build_crowell_h5ad.py` (Bioconductor `muscData`). Futtatás:

```powershell
python run_de_comparison.py --h5ad data/kang_8donors.h5ad --ref-label ctrl --test-label stim --output-dir results_kang
python run_mast_dataset.py  --h5ad data/kang_8donors.h5ad --results-dir results_kang --ref-label ctrl --test-label stim
python rebuild_comparison.py --results-dir results_kang
```

### DUET ↔ R MAST azonos sejteken, mindhárom adatszetten

| Adatszett · sejttípus | log2FC r | előjel | −log10p ρ | top-100 | Jaccard |
|---|---:|---:|---:|---:|---:|
| Kang · B cells | 1,000 | 100 % | 1,000 | 100/100 | 0,848 |
| Kang · CD14+ Monocytes | 1,000 | 100 % | 1,000 | 100/100 | 0,990 |
| Kang · CD4 T cells | 1,000 | 100 % | 1,000 | 100/100 | 0,950 |
| Crowell · Astrocytes | 1,000 | 100 % | 0,997 | 98/100 | 0,944 |
| Crowell · Excit. Neuron | 1,000 | 100 % | 1,000 | 100/100 | 1,000 |
| Crowell · Inhib. Neuron | 1,000 | 100 % | 0,999 | 99/100 | 0,969 |

Futásidő is tartja magát: a Crowell excitatorikus neuronjain (16 622 sejt × 5 842
gén) a MAST **407 s / 5,3 GB**, a DUET ~1 s. (A log egész másodpercre kerekít, így
a 29×–407× gyorsulások **alsó becslések**.)

---

## Ground truth — muscat szimuláció

`simulate_muscat.R` → `build_sim_h5ad.py` → `run_de_comparison.py` +
`run_mast_dataset.py` → `evaluate_sim.py`.

A szimuláció a **Crowell referenciából** készül (4 000 gén × 24 000 sejt, 4v4
minta), így a **minták közti variancia valósághű**. Ez lényeges: egy lapos,
minta-szintű variancia nélküli szimuláción minden sejtszintű módszer tökéletesnek
látszana — vagyis pont azt nem tesztelné, amit a null-kísérletek problémának
mutattak. Igazság: **405 DE gén** (de/dp/dm/db) és **3 202 egyértelmű null** (ee).

| Módszer | AUC | Erő (TPR) | **Megfigyelt FDR** | Ítélet (nominális 0,05) |
|---|---:|---:|---:|---|
| **DESeq2** (pseudobulk) | 0,9965 | 0,710 | **0,000** | kontrollált |
| diffxpy | 0,9924 | 0,948 | 0,075 | határeset |
| **DUET** | 0,9778 | **0,993** | **0,806** | **inflált** |
| R MAST | 0,9776 | **0,993** | **0,805** | **inflált** |
| Wilcoxon | 0,9756 | 0,785 | 0,124 | inflált |

> Az `ep` kategóriát (azonos átlag, eltérő arány) **kihagytuk mindkét osztályból**:
> átlag-eltolódás tesztnek null, de valódi eloszlásbeli különbség, amit egy hurdle
> jogosan detektálhat. A kihagyás a DUET-nek **kedvez** — és az FDR így is 0,81.

### Kategóriánként: hol hoz valódi értéket a hurdle

| Igazi kategória | n | DUET | MAST | diffxpy | DESeq2 | Wilcoxon |
|---|---:|---:|---:|---:|---:|---:|
| de — átlag-eltolódás | 113 | **100,0** | 100,0 | 99,1 | 93,8 | 92,9 |
| dp — eltérő arány | 104 | **100,0** | 100,0 | 97,1 | 77,9 | 82,7 |
| dm — eltérő modalitás | 104 | **98,1** | 98,1 | 92,3 | 73,1 | 78,8 |
| **db — bimodális** | 83 | **98,8** | 98,8 | 89,2 | **28,9** | 53,0 |
| ee — **VALÓDI NULL** | 3 192 | **52,3** | 51,8 | 1,0 | **0,0** | 1,4 |

*(FDR<0,05-nél szignifikánsnak hívott gének %-a. de/dp/dm/db: magasabb a jobb; ee: alacsonyabb a jobb.)*

- **Ez a sor indokolja a hurdle létezését.** A bimodális géneken — ahol a változás a
  *detektálási arányban* van, nem az átlagban — a DUET 98,8 %-ot talál, a pseudobulk
  28,9 %-ot (**3,4×**). Modalitás-változásnál 98,1 % vs 73,1 %. Egy átlag-alapú
  pseudobulk teszt szerkezetileg vak pont azokra a hatásokra, amikre a detekciós
  komponenst tervezték.
- **Ez a sor pedig korlátozza az állításokat.** A DUET a valóban azonosan
  expresszált gének **52,3 %-át** hívja szignifikánsnak — nem mert a teszt rossz (a
  permutációs null szerint kalibrált), hanem mert 24 000 sejt 8 mintából nem
  24 000 független megfigyelés.
- **A DUET és a MAST végig megkülönböztethetetlen**: AUC 0,9778 vs 0,9776, azonos
  erő, FDR 0,806 vs 0,805, és minden kategóriában egy tizedesre ugyanaz. A DUET nem
  rosszabb annál a módszernél, amit újraimplementál — pontosan örökli a viselkedését.

**Ajánlás:** a DUET-et (vagy MAST-ot) **rangsorolásra** és eloszlásbeli /
detektálási-arány változások kimutatására használd — ott a legerősebb a mezőnyben,
és 117× gyorsabb a referencia-implementációnál. Azt viszont, hogy **mely hívások
valódiak**, több-mintás tervnél **pseudobulk** döntse el. Egy sejtszintű hurdle
nyers „FDR<0,05 génszáma" donorok között nem védhető — ez a szimuláció megmutatja,
mennyivel nem.

---

## ⚠️ FONTOS: null-kalibráció — a DUET pseudoreplikációra érzékeny

Ez a legfontosabb tudnivaló az eredmények értelmezéséhez. Két **null-kísérlet**
futott **mindhárom adatszetten** (nincs valódi csoportkülönbség; a helyes válasz
~0 szignifikáns gén). Azonos génhalmazra szűrve, újraszámolt BH-FDR-rel.

**(1) Címke-permutáció** (a sejtcímkék keverése) — **a DUET mindenhol átmegy**:
I. fajú hiba 0,034–0,059 (nominális 0,05), 0 szignifikáns gén. A teszt tehát
helyesen kalibrált. (Kivétel: pancreas Ductal 0,203 — még magyarázatlan.)

**(2) Donor-csere** (a referencia-minták véletlen kettéosztása) — **a bukás mértéke
az ADATSZETTŐL függ, nem a módszertől**:

| Adatszett | ref. donor | osztás | DUET | Wilcoxon | **DESeq2-pseudobulk** |
|---|---:|---|---:|---:|---:|
| Pancreas | 16 | 8v8 | **75,5 / 99,3 / 98,9 %** | 47,5 / 33,3 / 53,1 % | **0,85 / 0,00 / 0,00 %** |
| **Kang 2018** | 8 | 4v4 | **1,0 / 10,3 / 27,0 %** | 0,6 / 6,7 / 10,5 % | **0,00 / 0,00 / 0,00 %** |
| Crowell | 4 | 2v2 | **99,9 / 100,0 / 100,0 %** | 6,5 / 88,8 / 50,5 % | **0,00 / 0,37 / 0,08 %** |

- **Nem a donorszám magyarázza.** A pancreas 16 donort oszt 8v8-ra és 75–99 %-on
  bukik; a Kang 8 donort oszt 4v4-re és 1–27 %-on marad; a Crowell 4 egeret oszt
  2v2-re és ~100 %-on bukik. A sorrend nem monoton.
- **A Kang az egyetlen multiplexált terv**: mind a 8 donor közös 10x futásokba
  poolozva, genetikai alapon demultiplexálva (a demuxlet éppen ezt demonstrálta).
  A pancreas 16 külön mátrixból áll, a Crowell egereinek külön könyvtár-preppje
  volt. A donor-csere null tehát valójában a **minták közti technikai/batch
  varianciát** méri — amit egyetlen sejtszintű módszer sem tud elválasztani a
  kondíció-hatástól.
- **Nem DUET-specifikus**: a Wilcoxon 88,8 %-ot ér el a Crowell excitatorikus
  neuronjain.
- A DUET következetesen 2–3×-osan rosszabb a Wilcoxonnál a donor-csere nullon —
  egy parametrikus hurdle érzékenyebb a donor-varianciára, mint egy rangteszt.

- **A pseudobulk a kontroll, ami átmegy.** Ugyanezeken a nullokon a PyDESeq2
  (donor az aggregációs egység) **0,00–0,85 %** hamis pozitívot ad **mind a 9
  adatszett × sejttípus kombinációban** — beleértve azokat is, ahol a DUET ~100 %-on
  áll. Tehát amit a donor-csere null kimutat, az valódi minták közti variancia, és a
  minta-szintű aggregálás megszünteti.

**Gyakorlati szabály:** a pseudoreplikációs büntetés a minták közti batch-variancia
mértékével skálázódik. **Poolozott/multiplexált** tervnél a sejtszintű tesztelés
védhető; **külön feldolgozott** mintáknál viszont **pseudobulk kötelező** (DESeq2)
vagy donor-szintű random effekt kell. A szignifikáns gének puszta száma
több-donoros tervnél önmagában nem értelmezhető.

Reprodukálás: `null_calibration.py` (a session scratchpadban).

---

## Numerikus megjegyzés: p-érték alulcsordulás és a `neglog10p`

`chi2.sf(stat, df)` **pontosan `0.0`-t** ad vissza, ha a hurdle-statisztika ~1400
fölé megy, mert a valódi p-érték a float64 alsó határa (~5e-324) alá esik. Ez a
tesztelt gének **3,5–6,9 %-át** érinti:

| Sejttípus | tesztelt | `pvalue == 0` | valódi −log10 p tartomány |
|---|---|---|---|
| Ductal cell | 8 497 | 590 (6,9 %) | 311 → **3 258** |
| Endothelial cell | 7 551 | 452 (6,0 %) | 311 → 2 067 |
| Stellate cell | 6 658 | 236 (3,5 %) | 312 → 1 723 |

Ha `pvalue` szerint rangsorolsz, ez a több száz gén **holtversenybe kerül**, és a
sorrend önkényes lesz. Mért hatás: a **top-100 átfedés ±46 génnel** eltolódott
(30 módszerpárból 20 változott). A **Spearman rangkorreláció** viszont
gyakorlatilag érintetlen (|Δρ| ≤ 0,003) — a kár a top-K listákra koncentrálódik.

**Megoldás:** a `neglog10p` oszlop. χ²-eloszlásra `df = 2` esetén
`sf(x) = exp(−x/2)` **pontosan**, tehát `−log10 p = stat / (2·ln10)`, ami sosem
csordul alul. (Általánosan: `log sf(x;k) → −x/2 + (k/2−1)·log(x/2) − lgamma(k/2)`;
`chi2.logsf`-hez képest max. abs. hiba **0,000e+00** ott, ahol a scipy még számol.)
A `pvalue` és a `Pr(>Chisq)` **változatlan** maradt a MAST-kompatibilitás miatt.

Csak a **DUET** és a **Wilcoxon** javítható: mindkettőnek van ismert
null-eloszlású statisztikája (χ², ill. normál z). Az **R MAST** CSV-je nem
tartalmaz statisztikát, a **diffxpy** `stat` oszlopa pedig `coef_mle`-t (együtthatót,
nem tesztstatisztikát) — ezeknél a farok sorrendje véglegesen elveszett. A diffxpy
p-értékeinek ráadásul **41–54 %-a pontosan 0**.

**Ábrázolás:** a `make_volcano.py --log-y` log y-tengelyt használ és nem vág.
Enélkül a vulkán teteje **lapos** — ez a lapos vonal sosem volt biológiai jelenség,
hanem az alulcsordulás + az y-vágás együttes műterméke. A log-ábrán a DUET és a
Wilcoxon panelje sima, folytonos farkat mutat; a MAST és a diffxpy panelje továbbra
is lapos 323-nál, mert azokat nem lehetett helyreállítani.

---

## Hogyan értelmezd az összehasonlítást

- A módszerek **eltérő statisztikai egységen** dolgoznak: a DUET / diffxpy /
  Wilcoxon **sejtenkénti** (óriási N → nagyon kicsi p-értékek, sok „szignifikáns"
  gén); a DESeq2 **pseudobulk** (minta-szintű, valódi replikáció). Ezért a **nyers
  szignifikáns-szám** módszerek között nem hasonlítható. Megbízható mérőszámok:
  1. **log2FC előjel-egyezés** (`sign_agree_pct`) — skálafüggetlen,
  2. **log2FC Spearman** (rang-korreláció),
  3. **−log10(p) Spearman** (`neglog10p` alapján).

- **log2FC-skála (`--duet-linear-logfc`).** Alapból a DUET a **TÖMÖRÍTETT**
  MAST-stílusú log-átlagkülönbséget adja (`coef/ln2`). Ebben a módban a log2FC-je
  az **R MAST**-éhoz áll közel:

  | log2FC Pearson (DUET ↔ R MAST) | Ductal | Endothelial | Stellate |
  |---|---|---|---|
  | tömörített (alap) | **0,964** | **0,841** | **0,906** |

  A `--duet-linear-logfc` a **LINEÁRIS** scanpy-skálára vált
  (`log2((expm1(mean_test)+ε)/(expm1(mean_ref)+ε))`).

  > **⚠️ Ne hivatkozz a „DUET ↔ Wilcoxon Pearson = 1,00" számra!**
  > Lineáris módban a DUET **karakterről karakterre ugyanazt a képletet** számolja,
  > mint a scanpy `logfoldchanges` (ugyanazzal az `ε = 1e-9`-cel). Ez tehát egy
  > képlet önmagával vett korrelációja — tautológia, nem validáció; 1,00-nál kisebb
  > érték hibát jelentene. Ebben a párban csak a **p-érték** független (hurdle LRT
  > vs. rangösszeg). Az érdemi effektusméret-validáció az **R MAST** elleni
  > összevetés (fenti tábla). Az **előjel-egyezés** viszont skálafüggetlen, ezért
  > végig értelmes: DUET ↔ Wilcoxon **100 %** mindhárom sejttípuson.

- **Génszám / `min.pct` szűrő.** A hurdle **detekciós komponense** pár ezer sejtnél
  a sejtek <10 %-ában kifejeződő géneket is „szignifikánsnak" jelöli (egy 18 %→21 %
  detektálási különbség is p≈0). A `--duet-min-detect-frac` (alap **0,10** =
  Seurat/scanpy `min.pct`) csak azokat a géneket tartja meg, amik **legalább az
  egyik csoport sejtjeinek ≥10 %-ában** kifejeződnek → a DUET génszáma a többiével
  egy szintre kerül. Az effektusméret-szűrő (|log2FC|) önmagában NEM segít, mert a
  lineáris log2FC pont az egy-csoportos ritka géneknél robban nagyra (|log2FC| ~28
  az `ε` miatt — a Wilcoxonnál ugyanez a műtermék).

- **Futásidő** (Ductal cell, 11 106 sejt): DUET **2 s** (59 537 gén) · Wilcoxon 14 s ·
  PyDESeq2 9 s · diffxpy **1 096 s** (csak 8 000 génen). Az R MAST elleni
  head-to-head mérést lásd lent.

---

## DUET vs R MAST — head-to-head benchmark

`bench_mast_export.py` → `bench_mast_run.R` → `bench_mast_report.py`.
**Azonos sejtek, azonos gének, azonos modell** (`~1+condition`, kovariáns nélkül —
ez a DUET `mast_compat` beállítása), ugyanazon a gépen. R 4.5.0 + MAST 1.33.0
(natív Windows, WSL nélkül). A géneket egyszer, Pythonban szűrjük, és csak a
túlélőket exportáljuk — így egyik motor sem fizet olyan szűrésért, amit a másik
nem végezne el.

| Sejt | DUET | MAST fit | MAST end-to-end | Gyorsulás (fit) | Gyorsulás (e2e) | MAST csúcs-RAM |
|---:|---:|---:|---:|---:|---:|---:|
| 1 000 | 1,19 s | 28,1 s | 55,0 s | 24× | 46× | 1,2 GB |
| 2 500 | 1,34 s | 56,7 s | 106,4 s | 42× | 79× | 1,8 GB |
| 5 000 | 1,49 s | 103,2 s | 193,5 s | 69× | 130× | 2,8 GB |
| **11 106** | **1,96 s** | **227,9 s** | **438,6 s** | **117×** | **224×** | **5,2 GB** |

- **A gyorsulás nem konstans, hanem nő az adat méretével.** Fajlagos költség:
  MAST **0,0188 s/sejt**, DUET **0,000072 s/sejt** → **260×** különbség. Ez a
  szám a lényeg, nem egyetlen gyorsulási arány.
- **Memória**: a MAST densifikál (~0,39 MB/sejt), a DUET egyszerre egy ritka
  gén-oszlopot streamel. Extrapolálva 100 000 sejtre a MAST ~40 GB-ot kérne.
- **Két óra, két kérdés.** A `fit` oszlop a tisztán algoritmikus összevetés (ezt
  érdemes az absztraktban idézni); az `end-to-end` tartalmazza a lemezes
  oda-vissza utat is, amit a MAST szerkezetileg nem tud elkerülni, a DUET viszont
  igen (in-memory AnnData). A DUET ideje **tartalmazza a saját CSV írás-olvasását**,
  a MAST `fit` ideje viszont **nem** tartalmaz I/O-t — az összevetés tehát a DUET
  kárára konzervatív.
- A MAST `lrTest`-je majdnem annyiba kerül, mint maga az illesztés (202 s a 438-ból),
  mert újrailleszti a redukált modellt. Ez az LRT-tervezés velejárója, nem
  konfigurációs hiba.

### Egyezés azonos sejteken

A korábbi MAST-összevetés konfundált volt: az a CSV **másik pipeline-futásból, más
sejteken** készült. Azonos sejteken az egyezés gyakorlatilag tökéletes, és a
sejtszámmal **javul**:

| Sejt | log2FC Pearson | előjel-egyezés | statisztika Spearman | −log10 p Spearman | szign. Jaccard |
|---:|---:|---:|---:|---:|---:|
| 1 000 | 1,0000 | 99,95 % | 0,9997 | 0,9997 | 0,959 |
| 2 500 | 1,0000 | 99,99 % | 0,9999 | 0,9999 | 0,985 |
| 5 000 | 1,0000 | 99,99 % | 1,0000 | 1,0000 | 0,992 |
| 11 106 | 1,0000 | 100,00 % | 1,0000 | 0,9999 | 0,998 |

> **Melyik szám az érdemi?** A log2FC Pearson = 1,0000 **várható**, nem meglepő:
> `~condition` mellett kovariáns nélkül a hurdle modell telített, így a MAST
> modell-alapú marginális logFC-je analitikusan a csoportátlagok különbségére
> egyszerűsödik — pontosan azt számolja a DUET közvetlenül. Ez helyességi
> ellenőrzés (hibát kiszúrt volna), nem bizonyíték a kovariánsos viselkedésre.
> Az érdemi számok a **statisztika és a p-érték korrelációi**, mert azok valóban
> független kódból jönnek: MAST `bayesglm` + LRT vs. DUET IRLS + zárt alakú
> Gauss-LRT EB-zsugorítással.

Melléktermék: ez a futás igazolta a projektben mindenhol használt `coef / ln2`
harmonizációt is (`add_mast_and_rebuild.py`), amit korábban sosem ellenőriztünk
kontrollált futáson. A DUET log2FC-jét erre regresszálva a meredekség **1,0027**
(a két alternatív skálázás 1,4466 és 2,0870 lenne). A 0,27 % maradék valós: a MAST
logFC-je modell-alapú, gyenge priorral, ezért kicsit zsugorodik.

### ⚠️ Ismert korlát: a DUET statisztikája 1373,87-nél telítődik

A statisztika **Pearson** korrelációja a sejtszámmal *csökken* (0,9999 → 0,9421),
miközben a **Spearman** 1,0000-re *nő*. Ez nem eltérés, hanem a float64-padló
második előfordulása — ezúttal magában a statisztikában, nem a jelentett p-értékben.

Az empirikus Bayes lépés `1e-300`-nál levágja a moderált p-értéket, mielőtt
visszaalakítaná χ²₁ statisztikává, így a folytonos komponens sosem lépheti túl a
`chi2.isf(1e-300, 1)` = **1373,87** értéket. A telített gének száma pontosan követi
a Pearson-csökkenést: **0 → 5 → 16 → 70** gén, ahogy a sejtszám 1 000 → 11 106.

**Hatás:** a gének ~1 %-a, mind csillagászatian szignifikáns amúgy is; a rangsor
érintetlen (Spearman 1,0000), tudományos következtetés nem változik. **De** emiatt a
„numerikusan robusztus farok" állítás jelenleg a DUET **jelentett p-értékére** igaz,
a **statisztikájára** nem. Ugyanazzal a log-térbeli technikával javítható, mint a
`neglog10p` — csakhogy ez már **megváltoztatná a p-értékeket és az FDR-t**, tehát
statisztikai változtatás, nem pusztán egy új oszlop. **Még nincs javítva.**

## Megjegyzések / hibatűrés

- **diffxpy lassú és RAM-igényes**: dense mátrixot kíván, a batchglm worker-enként
  lemásolja az adatot. Három együttes korlát tartja kordában: sejt-almintavétel
  (`--diffxpy-max-cells-per-group 5000`), gén-limit (`--diffxpy-max-genes 8000`),
  `nproc=1`. Így a csúcs-RAM ~7 GB.
- **diffxpy + modern dask**: a batchglm nullával oszt az `auto_chunks`-ban; a script
  célzott, idempotens monkeypatch-csel kezeli (`_patch_dask_auto_chunks`).
- Bármelyik módszer hibája csak az adott módszert ejti (try/except + naplóüzenet),
  az összehasonlítás a többiből elkészül.
- **Verziókövetés**: a `results/` és a kód korábban szétcsúszott (az eredmények
  régebbi kódverzióval készültek, mint ami a mappában volt). A futáshoz tartozó log
  (`results/rerun_*.log`) most rögzíti a futás idejét — érdemes a kódot és az
  eredményt együtt verziózni.
