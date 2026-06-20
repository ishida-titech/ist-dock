# SIMD_OPTIMIZE.md — ISCT-Dock SIMD 最適化の知見

探索ホットパスの SIMD 化を検討・実装・計測した結果の知見集約（2026-06）。
結論を先に: **現状の探索数値（grid=f32）と決定論/移植性の制約下では、SIMD による安全で大幅な高速化は得られない。**
真の高速化は SIMD でなく (a) アルゴリズム（直線探索を grid に整合, 深 polish 撤廃）と (b) スクリーニングの
grid 再利用で得られた。詳細な実測は `docs/findings.202606200652.md` / `docs/findings.202606200835.md` /
`docs/results.202606191907.md` を参照。

---

## 0. TL;DR（要点）

1. **まずプロファイルせよ。想定が外れる。** 律速は grid 補間ではなく **分子内エネルギー（ペアあたり 2×exp）**。
2. **真の大勝ちは SIMD でなくアルゴリズム**: 直線探索の値が grid 時も「受容体全原子の直接和 O(n_lig×n_rec)」を
   計算していた → grid 補間に整合させて **3.5×**（[[docs/results.202606191907]]）。深 polish 撤廃でさらに 1.3×。
3. **分子内 SIMD は gather/scatter 律速**: ペアの原子インデックスが不規則。AVX2 無しでは **スカラより遅い**。
4. **探索は数値的に鋭敏**: grid が既に f32 のため、f32 SIMD・AVX2 codegen・exp 近似いずれも **1e-7 級の摂動でも
   borderline 標的を反転**させ、フル 74 標的で **net 精度低下**（−4〜−7/222）。
5. **ビルドは既定で AVX2/native 無効** → `wide::f32x8` は emulate 実行。既存の分子間 SIMD も実は効いていなかった。
6. **AVX-512 でも結論は変わらない見込み**（gather/scatter 不変＋本機 Cascade Lake の downclock/Downfall 緩和）。

---

## 1. ビルドフラグの前提（重要）

- 既定の `cargo build --release` は **baseline x86-64（SSE2）** 向け codegen。`target-cpu=native` も
  `target-feature=+avx2` も無し（`.cargo/config` 無し・`RUSTFLAGS` 未設定）。
- このため `wide::f32x8`（256-bit）は **AVX2 命令にならず emulate**（2×f32x4 など）。既存の
  `scoring_simd::intermolecular_value_grad_simd` も **実機では SIMD の恩恵をほぼ受けていなかった**。
- AVX2 を効かせるには `RUSTFLAGS="-C target-cpu=native"`（または `+avx2,+fma`）。ただし下記の精度・移植性問題あり。

---

## 2. プロファイル: 真の律速は分子内 exp

`crates/isct-dock/examples/profile_eval.rs`（1 評価の内訳, 1iep, grid, AVX2 無し, ns/call）:

| 関数 | ns/call | 備考 |
|---|---|---|
| **intramolecular_value_grad** | **8357** | ← 最大。ペアあたり gauss1/gauss2 の **2×exp** |
| value_and_force（grid 勾配） | 2926 | 補間 O(n_lig)、安価 |
| search_score（grid 値） | 2351 | 補間 O(n_lig) |
| dof_gradient | 1433 | |
| world_coords_frames | 957 | 全原子クォータニオン変換 |
| scored_atoms（world_coords＋抽出） | 839 | |

- 分子内ペア数: 1iep ~250, 1eby ~645（カットオフ内, フラグメント間, 1-2/1-3 除外済＝既に最小）。
- **同一フラグメント内ペアは定数距離なので除外済**（`build_intra_pairs` の `diff_fragment` 判定）。これ以上の
  ペア削減余地は無い。コストは純粋に exp 回数。

---

## 3. 試した手法と実測（すべて撤回）

| 手法 | 速度 | 精度（/222） | 判定 | 理由 |
|---|---|---|---|---|
| **分子内 f32x8 SIMD**（gather/scatter） | AVX2無: **遅い**(10438 vs 8357ns) / AVX2有: 7866ns(~6%) | f32 起因 | 撤回 | 不規則 gather/scatter が律速。`docs/findings.202606200652` |
| **AVX2 codegen**（target-cpu=native） | 2.05→**1.55s (1.32×)** | **139→132 (−7)** | 撤回 | f32 丸め（FMA 等）で borderline 反転＋移植性/決定論 |
| **ペア静的データ前計算**（半径和・フラグ） | **0%**（8357→8356ns） | ビット一致 | 撤回 | match ルックアップは元々安価。exp 律速 |
| **gauss1 スキップ**（d>2 で exp 省略） | per-call ~9% / 全体 2.05→**1.90s (1.08×)** | **139→135 (−4)** | 撤回 | d>2 ペアが少なく効果小。1e-7 近似でも borderline 反転 |

---

## 4. なぜ分子内 SIMD は割に合わないか

### (a) 不規則 gather/scatter が律速
- 分子内ペアは任意の原子インデックス対 `(a,b)`。8 ペア分の座標を読むには **scalar gather**（8×2×3=48 ロード）、
  力を書くには **scalar scatter**（48 ストア）が要る。SIMD 化できるのは exp/算術のみで、メモリ律速になる。
- 対照的に **分子間（受容体）SIMD は規則的**（リガンド 1 原子 × 受容体 SoA を連続走査）で gather 不要＝
  SIMD 向き。`ReceptorSimd` はこの形。ただし §1 の通り AVX2 を効かせないと emulate。

### (b) 探索の数値的鋭敏性（最重要）
- 探索は grid（f32）上の長い MC×BFGS 軌道。**目的関数への 1e-7 級の摂動でも、borderline（RMSD≈2Å）標的が
  成功↔失敗で反転**する。f32 SIMD・AVX2 FMA・exp 近似のいずれでも一貫して観測（net −4〜−7）。
- ＝「探索の数値を変えずに」速くする手段でないと精度を賭けることになる。SIMD/近似は本質的に数値を変える。

### (c) 決定論・移植性
- codegen 依存（AVX2/AVX-512 の有無）で結果が変わる＝プロジェクトの**決定論**価値と衝突。
- AVX-512 は大半のコンシューマ CPU に無い（Intel は Alder Lake 以降で無効化, AMD は Zen4+）。10M
  スクリーニングを多様なハードで回す前提と相性が悪い（PORTING）。

---

## 5. AVX-512 の予想（未実装・本機 Xeon Gold 5218R = Cascade Lake-SP）

`wide` は `f32x16`/`f64x8`（512-bit）を持ち、本機は AVX-512 対応のため**実装自体は可能**。だが結論は変わらない見込み:

- **gather/scatter は不変かむしろ悪化**: `VGATHER` は microcode（~1 要素/cycle）。本機は **Downfall(CVE-2022-40982)**
  対象で、緩和 microcode が AVX gather を**大幅に低速化**。`VSCATTER` は **力の集約に使えない**（同一原子が
  チャンク内複数ペアに出る＝重複インデックス衝突。`VPCONFLICT` 直列化が必要で遅い）。
- **Cascade Lake の AVX-512 ダウンクロック**: 重い AVX-512 でコア周波数が低下（2.1GHz→~1.5-1.8GHz）。混在
  ワークロード（grid/world_coords 等のスカラ部含む）でも巻き込まれ、**多スレッドのスクリーニングでは利得が
  相殺/逆転**しうる。
- **精度**: `f32x16` は f32 鋭敏性そのまま（−7）。`f64x8`（8 レーン f64）は**精度を保つ**唯一の妙手だが、レーンは半分で
  gather/scatter 律速＋ダウンクロックは同じ → 利得 ~1.1× 程度の見込み。

| 構成 | 単発 dock 予想 | 20スレ スクリーニング予想 | 精度 |
|---|---|---|---|
| AVX2 f32x8（実測） | 1.32× | （downclock 影響小） | −7 |
| AVX-512 f32x16（予想） | ~1.1-1.4× | **flat〜負（downclock）** | −7 |
| AVX-512 f64x8（予想） | ~1.05-1.15× | flat | **≈0（保たれる）** |

**予想結論: 本機では AVX-512 化は割に合わない。** 律速は SIMD 幅でなく不規則アクセスと精度鋭敏性で、幅拡大は
ボトルネックでない算術を広げるだけ。

---

## 6. 実際に効いた最適化（SIMD 以外・採用済み）

SIMD 調査の過程で特定・採用した、**精度を落とさない**高速化:

1. **直線探索を grid 値に整合**（`VinaGrid::value` / `Scorer::search_score`）: 直線探索の値が grid 時も直接和
   O(n_lig×n_rec) を計算していたのを補間に。**3.5×高速＋精度 +3**（表面整合で収束改善）。最大の勝ち。
2. **深 exact polish の撤廃**（polish_iters 200→20, polish_modes 12→9）: grid 整合後は代表が既に最適で深 polish は
   ほぼ無駄。**1.3×・精度不変**。
3. **torsion 連動の step 予算頭打ち**（剛性大リガンドの過剰 step 削減・ハード標的不変）。
4. ゼロアロケーション再利用バッファ（速度寄与は小だが多スレッド競合低減・冗長 world_coords 再計算除去）。

→ 現状 **139/222（63%）・平均 2.05s**、Vina（7.6s/64%）比 **~3.7×高速・精度ほぼ同等**。

---

## 7. もし将来 SIMD/exp 高速化を本気でやるなら（前提条件）

1. **アクセスを規則化**: 分子内ペアを連続レイアウト（pair-SoA を 1 パスで構築）にして gather を除去。force は
   原子単位の隣接リスト集約で scatter 衝突を回避。実装コスト大。
2. **精度鋭敏性を下げる**のが先決: より頑健な探索（粗→精の段階的精密化、決定的な数値で軌道を安定化）。これが
   無いと、どんな数値変更も borderline 標的を ±数件動かし net 利得が読めない。
3. **決定論・移植性を壊さない**: codegen 依存（native/avx512）を既定にしない。必要なら feature gate＋実機検証。
4. **多スレッド前提で計測**: 単発 dock の per-call 利得は、スクリーニング（全コア稼働＋AVX 周波数オフセット）で
   逆転しうる。スループット計測で判断。

## 8. 次の筋（SIMD 以外）

- **スクリーニングの grid 再利用**（`screen.rs` は現状クラスタ毎に grid 再構築）: 固定ポケットの 10M
  スクリーニングで全 XS 型マップをボックス毎 1 度だけ構築・全リガンド再利用。**探索数値に触れず**スループット
  直結。SIMD より筋が良い（[[docs/findings.202606191932]]）。
