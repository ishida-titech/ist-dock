//! グローバル探索（M2）。
//!
//! Vina 流の反復局所探索（Monte Carlo）: ランダム初期配置から「摂動 → 局所最適化 →
//! Metropolis 受理」を繰り返し、複数の独立 run で最良ポーズを探す。`rand_chacha` の
//! 固定シードで完全に決定論的。

use std::cell::RefCell;

use isct_core::geometry::{rmsd, Vec3};
use isct_core::scoring::ScoredAtom;
use rand::{Rng, RngCore, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;

use crate::energy::Scorer;
use crate::ligand::{Conformation, LigandModel};
use crate::optimizer::{minimize_free, rigid_and_torsions};
use crate::qvina::{dof_vector, project_grad, HistoryBuffer, QVinaParams};
use crate::tpe::{self, TpeParams};

/// ポーズ評価の再利用バッファ（ゼロアロケーション探索用）。run ごと（または polish ループごと）に
/// 1 つ確保し、毎ステップ・直線探索試行で使い回す。`minimize_free` は per-run で逐次のため、
/// `RefCell` 経由の内部可変で `Fn` クロージャから安全に再利用できる（スレッド間共有しない）。
#[derive(Default)]
struct EvalScratch {
    pos: Vec<Vec3>,
    ref_axis: Vec<Vec3>,
    ref_pivot: Vec<Vec3>,
    world_axis: Vec<Vec3>,
    world_pivot: Vec<Vec3>,
    atoms: Vec<ScoredAtom>,
    force: Vec<Vec3>,
    /// 直近に評価した conformation の scored 原子 world 座標（RMSD クラスタリングの再計算回避用）。
    scored_pos: Vec<Vec3>,
    /// 直近に評価した conformation の**分子間項（inter, box 込み）**。ポーズの保持・順位付けを
    /// Vina 流 affinity（inter 基準）で行うためのランクキー。箱内では box=0 なので純 inter に一致。
    /// MC 受理・BFGS は full objective（inter+intra）を使い続ける（力学は不変）。
    inter: f64,
    /// このバッファで行ったエネルギー評価回数（value/value_grad 呼び出し数）。探索アルゴリズムの
    /// 仕事量を計測し drift/vina モードを比較するための診断カウンタ。
    evals: u64,
}

/// 分子内項に渡す hunt_cap: polish（exact）は cap 無し＝authentic、探索（`!exact`）は `cap`。
#[inline]
fn intra_cap(exact: bool, cap: f64) -> f64 {
    if exact {
        f64::INFINITY
    } else {
        cap
    }
}

/// 値＋解析的勾配（全 DOF）。`exact=true` で厳密 f64 表面（polish 用）、false で探索バックエンド。
/// 勾配は渡された `out`（dim 長へリサイズ）へ書き込み、値を返す（Vec を毎回確保しない＝直線探索の
/// 各試行で融合評価してもアロクしない）。その他（world 座標・原子・力）もバッファ再利用。
fn eval_value_grad(
    lig: &LigandModel,
    scorer: &Scorer,
    s: &mut EvalScratch,
    c: &Conformation,
    exact: bool,
    cap: f64,
    out: &mut Vec<f64>,
) -> f64 {
    s.evals += 1;
    lig.world_coords_frames_into(
        c, &mut s.pos, &mut s.ref_axis, &mut s.ref_pivot, &mut s.world_axis, &mut s.world_pivot,
    );
    lig.scored_atoms_into(&s.pos, &mut s.atoms);
    // scored 原子 world 座標を控える（cluster_insert の RMSD 用。world_coords 再計算を避ける）。
    s.scored_pos.clear();
    s.scored_pos.extend(s.atoms.iter().map(|a| a.pos));
    s.force.clear();
    s.force.resize(s.atoms.len(), Vec3::zeros());
    let mut e = if exact {
        scorer.exact_value_and_force(&s.atoms, &mut s.force)
    } else {
        scorer.value_and_force(&s.atoms, &mut s.force, cap)
    };
    s.inter = e; // 順位付け用に分子間項を控える（intra 加算前。箱内 box=0）。
    let intra_cap = intra_cap(exact, cap);
    // 探索（!exact）は LUT 版、polish（exact）は厳密版（出力の厳密性を保つ）。
    e += if exact {
        lig.intramolecular_value_grad(&s.atoms, &mut s.force, intra_cap)
    } else {
        lig.intramolecular_fast_value_grad(&s.atoms, &mut s.force, intra_cap)
    };
    lig.dof_gradient_into(c, &s.pos, &s.world_axis, &s.world_pivot, &s.force, out);
    e
}

/// 探索パラメータ。
#[derive(Debug, Clone)]
pub struct SearchParams {
    pub center: Vec3,
    pub size: Vec3,
    pub seed: u64,
    /// 独立 run 数（探索の網羅性）。
    pub n_runs: usize,
    /// 各 run の MC ステップ数。
    pub n_steps: usize,
    /// ILS 鎖の各ステップの局所最適化反復（浅く保ち、鎖を基底間でドリフトさせ広く探索する）。
    pub chain_iters: usize,
    /// 代表ポーズの最終精密化の反復（厳密 f64 表面で厚く収束）。
    pub polish_iters: usize,
    /// ポーズ RMSD クラスタリングの閾値（Å）。これ未満を同一モードとみなす。
    pub cluster_rmsd: f64,
    /// 出力・保持する最大モード数（Vina 流）。
    pub n_modes: usize,
    /// 最終精密化にかける代表モード数（≥ n_modes）。粗極小の near-native を精密化で救う漏斗を広げる。
    pub polish_modes: usize,
    /// 早期停止の patience（新規モードも代表改善も無いステップ数の上限）。0 で無効（フル n_steps）。
    /// 収束した易標的の過剰計算を抑える。進展があるハード標的は停止しない。
    pub early_stop_patience: usize,
    /// 焼きなまし Metropolis 温度の初期値（kcal/mol。高温で誤基を脱出し広く探索）。
    pub temp_init: f64,
    /// 焼きなまし温度の終値（低温で収束）。run 内で temp_init→temp_final へ幾何冷却。
    pub temp_final: f64,
    /// 初期配置を一様ランダムにする（接触棄却を省く）。グリッドで勾配場が滑らかなとき有効。
    pub uniform_start: bool,
    /// 探索アルゴリズムの種類（既定 Drift＝ドリフト鎖／Qvina2＝局所探索スキップ／Vina＝忠実 MC）。
    pub kind: SearchKind,
    /// QuickVina2 の勾配方向一致閾値（cos）。近傍の勾配と cos>これで局所探索をスキップ。
    /// 高いほど保守的（スキップ少・速度↓精度↑）、低いほど攻める（スキップ多・速度↑精度↓）。既定 0.90。
    pub qvina_cos: f64,
    /// 探索中の正エネルギー上限（Vina hunt_cap, 既定 10.0）。激突の暴走を抑え深い極小へ抜けやすくする。
    /// polish は常に cap 無し（authentic）。`f64::INFINITY` で無効化。
    pub hunt_cap: f64,
    /// TPE モード（`kind == Tpe`）のハイパーパラメータ。他モードでは未使用。
    pub tpe: TpeParams,
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            center: Vec3::zeros(),
            size: Vec3::new(20.0, 20.0, 20.0),
            seed: 42,
            n_runs: 60,
            n_steps: 0, // 0 = 可動原子数から自動スケール（auto_n_steps）
            // 浅い per-step BFGS は、各ステップが decoy 基へ底打ちするのを避け、受理小移動を基底間で
            // ドリフト蓄積させて別基(native)へ届かせる（Vina 流）。単独の浅化は収束を損ねるため、
            // 多 step（auto_n_steps 増）と最終深 polish（polish_iters/polish_modes）とセットで用いる。
            // 直線探索 warm-start と併せて 35 が既定（RMSD/エネルギー中立で高速化。config と一致）。
            chain_iters: 35,
            polish_iters: 20,
            cluster_rmsd: 2.0,
            n_modes: 9,
            // 粗い near-native 代表が精密化で勝ちモードへ昇格する機会を増やすため、出力数よりやや広く
            // polish（ただし 20 まで深 polish するのは無駄＝12 で精度を保ちつつ polish コストを削減）。
            polish_modes: 9,
            // 早期停止は既定 OFF（0）。収束ベースの停止はドリフト鎖のハード標的を退行させうるため
            // （native へ終盤に跳ぶ標的を切ってしまう, measured）、opt-in に留める。
            early_stop_patience: 0,
            // 既定は固定温度（temp_init==temp_final）。焼きなまし（temp_init>temp_final）は
            // 誤基脱出に効く反面、探索が発散し exploitation 主体の標的を不安定化するため opt-in。
            temp_init: 1.0,
            temp_final: 1.0,
            uniform_start: false,
            kind: SearchKind::Drift,
            qvina_cos: 0.90,
            hunt_cap: 10.0,
            tpe: TpeParams::default(),
        }
    }
}

/// ドッキング結果。
pub struct DockResult {
    pub conf: Conformation,
    /// 順位キー = **exact intermolecular energy（= Vina affinity 基準, kcal/mol）**。モードは昇順（[0]=最良）。
    /// full objective(inter+intra) ではない（大リガンドで intra 歪みが affinity 良好な near-native を捨てないため）。
    pub energy: f64,
}

/// per-run の MC ステップ数を自動算出する。`SearchParams.n_steps == 0` のとき使う。
///
/// 基本は可動原子数スケール（`30 * n_scored`）だが、**探索難度は原子数でなく自由度（torsion）で決まる**
/// ため、原子数を torsion 由来の上限 `8 + 5*tors` で**頭打ち**にする。これにより 1 torsion の剛性大
/// リガンド（例 1gpk: 18 原子で 540→390）の過剰計算だけを削り、**高 torsion 標的の予算は不変**に保つ
/// （1iep 7tor: min(37,43)=37→1110, 1y6b 9tor→1020, 1eby 17tor→不変）＝ハード標的を退行させない。
pub fn auto_n_steps(n_scored_atoms: usize, n_torsions: usize) -> usize {
    let eff = n_scored_atoms.min(8 + 5 * n_torsions);
    (30 * eff).max(300)
}

/// vina-mode（毎ステップ収束 BFGS）の per-run ステップ数。**AutoDock Vina 1.x の per-run MC
/// ステップ数ヒューリスティック（`main.cpp`）に一致**させる（vina-mode を AutoDock Vina 相当の
/// 探索予算にする）:
///   `dof = num_torsions + 6`（単一リガンド: 3 平行移動 + 3 回転 + torsion; Vina `conf_size::num_degrees_of_freedom`）
///   `heuristic = num_movable_atoms + 10 * dof`
///   `num_steps = 70*3*(50 + heuristic)/2 = 105*(50 + heuristic)`
/// `num_movable_atoms` は重原子数（`n_scored_atoms`）で近似（Vina は極性 H も数えるが本エンジンは
/// 重原子スコア）。exhaustiveness は `n_runs` に対応（Vina 既定 8。vina-mode は n_runs 未指定時の
/// 既定を 8 にしてある＝`config::default_n_runs_for(Vina)`）。
/// 較正用に `n_steps` を明示すれば上書きされる。
pub fn auto_n_steps_vina(n_scored_atoms: usize, n_torsions: usize) -> usize {
    let dof = n_torsions + 6;
    let heuristic = n_scored_atoms + 10 * dof;
    105 * (50 + heuristic)
}

/// TPE モードの per-run 提案数。**TPEvina 既定に忠実**: 1 run = TPEvina の 1 task とみなし、per-task
/// steps = `num_steps/15`（`num_steps` は vina と同じ Vina ヒューリスティック `auto_n_steps_vina`）。
/// 既定 `n_runs=8`（=num_tasks）と組で **AutoDock Vina 相当の総探索量**（`n_runs × steps ≈ num_steps·8/15`）
/// になる。各提案が BFGS 局所最適化を伴うため「少数・深い run」。小リガンドは floor 200 で warmup を確保。
/// （`docs/results.202607090951`: 60 浅 run→8 深 run で redocking 改善）
pub fn auto_n_steps_tpe(n_scored_atoms: usize, n_torsions: usize) -> usize {
    (auto_n_steps_vina(n_scored_atoms, n_torsions) / 15).max(200)
}

/// 探索モードに応じた per-run ステップ数（`SearchParams.n_steps == 0` のとき使う）。
/// vina-mode は「少数・高コスト」ステップ（`auto_n_steps_vina`）、それ以外（drift/qvina2）は
/// 通常スケール（`auto_n_steps`）。dock・screen・CLI ログで共通利用し選択ロジックを一元化する。
pub fn auto_n_steps_for(kind: SearchKind, n_scored_atoms: usize, n_torsions: usize) -> usize {
    match kind {
        SearchKind::Vina => auto_n_steps_vina(n_scored_atoms, n_torsions),
        SearchKind::Tpe => auto_n_steps_tpe(n_scored_atoms, n_torsions),
        _ => auto_n_steps(n_scored_atoms, n_torsions),
    }
}

/// vina-mode の Metropolis 温度（AutoDock Vina の `monte_carlo::temperature`）。
const VINA_TEMP: f64 = 1.2;
/// vina-mode の変異振幅（AutoDock Vina の `monte_carlo::mutation_amplitude`, Å）。
const VINA_AMPLITUDE: f64 = 2.0;

/// 単位球内の一様乱点（棄却サンプリング）。Vina の `random_inside_sphere` 相当。
fn random_in_unit_sphere<R: Rng>(rng: &mut R) -> (f64, f64, f64) {
    loop {
        let x = rng.gen_range(-1.0..1.0);
        let y = rng.gen_range(-1.0..1.0);
        let z = rng.gen_range(-1.0..1.0);
        if x * x + y * y + z * z <= 1.0 {
            return (x, y, z);
        }
    }
}

/// scored 原子の重心まわり慣性半径（gyration radius）。Vina は配向変異をこれで割って
/// 「大きい分子ほど回転を小さく」スケールする。参照配座から 1 度だけ計算する。
fn gyration_radius(ref_pos: &[Vec3]) -> f64 {
    if ref_pos.is_empty() {
        return 1.0;
    }
    let n = ref_pos.len() as f64;
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for p in ref_pos {
        cx += p.x;
        cy += p.y;
        cz += p.z;
    }
    let (cx, cy, cz) = (cx / n, cy / n, cz / n);
    let mut s = 0.0;
    for p in ref_pos {
        s += (p.x - cx).powi(2) + (p.y - cy).powi(2) + (p.z - cz).powi(2);
    }
    (s / n).sqrt()
}

/// 変異の振幅スタイル。`Drift`＝従来のドリフト鎖（成分ごとの立方体）、`Vina`＝Vina 一致
/// （球内乱点・配向は gyration スケール）。entity 選択・torsion 摂動・apply は共通（[`apply_mutation`]）。
#[derive(Debug, Clone, Copy)]
enum MutationStyle {
    Drift,
    Vina { gyration: f64 },
}

/// 探索アルゴリズムの種類。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SearchKind {
    /// ドリフト鎖 ILS（既定）: 各ステップ浅い局所最適化（`chain_iters`）で広く探索。
    #[default]
    Drift,
    /// QuickVina2 相当の局所探索スキップ（opt-in 高速化・決定論維持）。
    Qvina2,
    /// Vina 忠実 MC: 各ステップ**収束まで BFGS**（実装レベル差の切り分け用）。
    Vina,
    /// TPE（Tree-structured Parzen Estimator）: TPEvina 由来の Parzen サンプリング探索（opt-in）。
    Tpe,
}

/// リガンドを受容体ボックスへドッキングし、**RMSD クラスタリングした上位モード**を
/// エネルギー昇順で返す（[0]=最良。全 torsion 自由）。返り値は `(モード, 総エネルギー評価回数)`。
pub fn dock(lig: &LigandModel, scorer: &Scorer, params: &SearchParams) -> (Vec<DockResult>, u64) {
    let free: Vec<usize> = (0..lig.n_torsions()).collect();
    dock_constrained(lig, scorer, params, &free)
}

/// Scaffold 固定ドッキング: `free_torsions` の torsion と剛体自由度のみを動かし、その他の
/// torsion（コアを動かすもの）は参照値 0 に固定したまま探索する。Phase 1 で seed のコアを
/// 剛体のまま位置・配向・R 基だけ大域探索するのに使う。
/// 返り値は `(上位モード, 総エネルギー評価回数)`。評価回数は探索の仕事量を比較する診断用。
pub fn dock_constrained(
    lig: &LigandModel,
    scorer: &Scorer,
    params: &SearchParams,
    free_torsions: &[usize],
) -> (Vec<DockResult>, u64) {
    // 各 run の seed を親 rng から順に生成する（実行順非依存・再現可能）。これにより run を
    // rayon で並列化しても結果が決定的になる。
    let mut seed_rng = ChaCha8Rng::seed_from_u64(params.seed);
    let run_seeds: Vec<u64> = (0..params.n_runs).map(|_| seed_rng.next_u64()).collect();

    // per-run ステップ数: n_steps==0 なら自動スケール（モード別、auto_n_steps_for に一元化）。
    let n_steps = if params.n_steps == 0 {
        auto_n_steps_for(params.kind, lig.n_scored(), free_torsions.len())
    } else {
        params.n_steps
    };

    // 最適化する全 DOF（剛体6 ＋ R 基 torsion）。コアを動かす torsion は除外される。
    let free_dofs = rigid_and_torsions(free_torsions);

    // 独立 run を並列実行。各 run は RMSD クラスタした代表ポーズ群を返す。collect で run 順保持＝決定的。
    // 評価バッファは per-run で確保（スレッドごとに独立）。TPE モードは専用ドライバへ分岐。
    let per_run: Vec<(Vec<(Conformation, f64)>, u64)> = run_seeds
        .par_iter()
        .map(|&seed| {
            if params.kind == SearchKind::Tpe {
                single_run_tpe(lig, scorer, params, n_steps, free_torsions, &free_dofs, seed)
            } else {
                single_run(lig, scorer, params, n_steps, free_torsions, &free_dofs, seed)
            }
        })
        .collect();

    // 全 run の代表を集約し、大域 RMSD クラスタリングで上位 polish_modes（≥ n_modes）モードへ絞る。
    // 代表のキーは inter（=affinity 基準）＝affinity の良い near-native を精密化候補へ確実に残す。
    let mut total_evals: u64 = per_run.iter().map(|(_, e)| e).sum();
    let all: Vec<(Conformation, f64)> = per_run.into_iter().flat_map(|(r, _)| r).collect();
    let polish_cap = params.polish_modes.max(params.n_modes);
    let reps = cluster_poses(lig, all, params.cluster_rmsd, polish_cap);

    // 各代表を **厳密 f64 表面**で厚く精密化（grid/f32 の平滑化ズレを補正）。**最適化目標は full objective
    // (inter+intra)**（ポーズを物理的に妥当に保つ, Vina と同一）。ただし精密化後の**順位付けは exact inter
    // (=Vina affinity 基準)** で行う: intra 歪みの大きい near-native を低歪み decoy に負けて捨てないため。
    let polish_scratch = RefCell::new(EvalScratch::default());
    // polish は厳密 f64・cap 無し（authentic）。融合 value_grad（値＋勾配をバッファに書く）。
    let exact_value_grad = |c: &Conformation, g: &mut Vec<f64>| {
        eval_value_grad(lig, scorer, &mut polish_scratch.borrow_mut(), c, true, f64::INFINITY, g)
    };
    let polished: Vec<(Conformation, f64)> = reps
        .into_iter()
        .map(|(conf, _)| {
            let r = minimize_free(conf, &exact_value_grad, params.polish_iters, &free_dofs);
            // 順位キー = 精密化後ポーズの exact intermolecular（cmd_dock 報告 affinity と同一定義, 少数で安価）。
            let inter = scorer.intermolecular(&lig.scored_atoms(&r.conf));
            (r.conf, inter)
        })
        .collect();
    total_evals += polish_scratch.borrow().evals;

    // 精密化でポーズが寄りうるので再クラスタ・再ソート → 上位モード（inter 昇順, [0]=affinity 最良）。
    // DockResult.energy は順位キー＝inter（分子間項）。cmd_dock は per-mode で affinity/intra を別途再計算する。
    let modes = cluster_poses(lig, polished, params.cluster_rmsd, params.n_modes)
        .into_iter()
        .map(|(conf, inter)| DockResult { conf, energy: inter })
        .collect();
    (modes, total_evals)
}

/// ランクキー昇順の greedy RMSD クラスタリング。候補をキー昇順で処理し、既存代表と RMSD<threshold なら
/// 同一モード（キーの低い方＝先に来る方が代表）、無ければ新代表。代表＝区別される最低キーのポーズ群
/// （最大 cap 個）。決定論的（安定ソート＋入力順）。**キーは inter（=Vina affinity 基準）**（呼び出し側が
/// そう渡す）＝affinity の良い near-native を低歪み decoy に負けて捨てない。
fn cluster_poses(
    lig: &LigandModel,
    mut cands: Vec<(Conformation, f64)>,
    threshold: f64,
    cap: usize,
) -> Vec<(Conformation, f64)> {
    cands.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut reps: Vec<(Conformation, f64, Vec<Vec3>)> = Vec::new();
    for (conf, e) in cands {
        let pos = lig.scored_positions(&conf);
        if reps.iter().any(|(_, _, rpos)| rmsd(&pos, rpos) < threshold) {
            continue; // 既存モードに支配される（より高エネルギー）
        }
        reps.push((conf, e, pos));
        if reps.len() >= cap {
            break;
        }
    }
    reps.into_iter().map(|(c, e, _)| (c, e)).collect()
}

/// 局所最小を per-run プールへクラスタ挿入（近傍代表より `rank` が低ければ置換、無ければ追加、cap 超過で
/// 最悪退避）。`rank` は **inter（分子間項＝Vina affinity 基準）**＝保持キー（objective ではない。affinity の
/// 良い near-native を低歪み decoy に負けて捨てないため）。`pos` は `conf` の scored 原子 world 座標
/// （呼び出し側が評価時に算出済みのものを渡す＝再計算回避）。
fn cluster_insert(
    pool: &mut Vec<(Conformation, f64, Vec<Vec3>)>,
    conf: Conformation,
    rank: f64,
    pos: Vec<Vec3>,
    threshold: f64,
    cap: usize,
) {
    for rep in pool.iter_mut() {
        if rmsd(&pos, &rep.2) < threshold {
            if rank < rep.1 {
                *rep = (conf, rank, pos);
            }
            return;
        }
    }
    pool.push((conf, rank, pos));
    if pool.len() > cap {
        // 最大 rank（=最悪 affinity）の代表を退避（タイは最初）。
        let worst = (0..pool.len())
            .max_by(|&i, &j| pool[i].1.partial_cmp(&pool[j].1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        pool.remove(worst);
    }
}

/// 1 つの独立 run（接触開始 → ILS）を実行し、その run で見つけた **区別されるモードの代表群**
/// （RMSD クラスタ済み, エネルギー昇順）を返す。鎖は浅い `chain_iters` で広く探索する。
#[allow(clippy::too_many_arguments)]
fn single_run(
    lig: &LigandModel,
    scorer: &Scorer,
    params: &SearchParams,
    n_steps: usize,
    free_torsions: &[usize],
    free_dofs: &[usize],
    seed: u64,
) -> (Vec<(Conformation, f64)>, u64) {
    // 各ステップの局所最適化反復: drift/qvina2 は浅い `chain_iters`、vina は **AutoDock Vina の
    // per-step 局所探索上限**に一致させる（`max_steps = ssd_par.evals = (25 + num_movable_atoms)/3`,
    // main.cpp）。num_movable_atoms は重原子数で近似。Vina は収束させず浅くキャップするので、
    // これで per-step コストも Vina 相当になる（多ステップ×浅局所＝Vina の探索形）。
    let is_vina = params.kind == SearchKind::Vina;
    let step_iters = if is_vina {
        ((25 + lig.n_scored()) / 3).max(1)
    } else {
        params.chain_iters
    };
    // 変異スタイル。vina-mode は配向を gyration でスケールするため参照配座から 1 度だけ計算（Vina 流）。
    let mut_style = if is_vina {
        MutationStyle::Vina { gyration: gyration_radius(&lig.reference_scored_positions()) }
    } else {
        MutationStyle::Drift
    };
    // この run 専用の評価バッファ（スレッドローカル）。融合直線探索では採択試行が最後の value_grad 呼び出し
    // となるため、`minimize_free` 返却後に `scratch_g.scored_pos` が **返却 conf の scored 座標**を保持する。
    // cluster_insert はこの座標を再利用でき world_coords 再計算を避けられる。
    let scratch_g = RefCell::new(EvalScratch::default());
    // 探索は hunt_cap で激突エネルギーを上限化（params.hunt_cap、無効化は INFINITY）。
    let cap = params.hunt_cap;
    let value_grad =
        |c: &Conformation, g: &mut Vec<f64>| eval_value_grad(lig, scorer, &mut scratch_g.borrow_mut(), c, false, cap, g);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // 初期配置（grid 時は一様、direct 時は接触棄却）。固定 torsion は 0 に保つ。
    let start = contact_start(
        lig,
        scorer,
        &mut rng,
        params.center,
        params.size,
        free_torsions,
        params.uniform_start,
        cap,
    );
    let mut cur = minimize_free(start, &value_grad, step_iters, free_dofs);

    // 訪れた局所最小をクラスタ保持（受理/非受理を問わず）＝探索の幅を保持。出力数(n_modes)より広く
    // 残し、ドリフトで稀に訪れた near-native モードを大域 polish 候補へ確実に渡す。
    let pool_cap = params.polish_modes.max(params.n_modes);
    // pool の第2要素は **inter（分子間項＝Vina affinity 基準）**＝保持・順位のキー。MC 力学とは分離。
    let mut pool: Vec<(Conformation, f64, Vec<Vec3>)> = Vec::new();
    let (cur_pos, cur_inter) = {
        let g = scratch_g.borrow();
        (g.scored_pos.clone(), g.inter)
    };
    cluster_insert(&mut pool, cur.conf.clone(), cur_inter, cur_pos, params.cluster_rmsd, pool_cap);

    // 早期停止（任意・既定 OFF）: best エネルギーが patience ステップ改善しなければ打ち切る。
    // 注意: ドリフト鎖のハード標的（1iep/1y6b 等）は **decoy 基で best が長く停滞してから終盤に native へ
    // 跳ぶ** ため、収束ベースの早期停止は native 到達前に切ってしまい退行させうる（measured）。よって
    // 既定は patience==0（無効＝フル n_steps）。収束が明確な易標的のみ短縮したい場合の opt-in。
    let patience = params.early_stop_patience;
    let mut best_e = cur.energy;
    let mut stale = 0usize;

    // QuickVina2: 変異後の試行点が「新しい井戸」でなければ局所探索をスキップ（per-run 履歴＝決定論）。
    let mut history = (params.kind == SearchKind::Qvina2)
        .then(|| HistoryBuffer::new(QVinaParams::for_dofs(free_dofs.len(), params.qvina_cos)));

    // 反復局所探索（ILS）＋焼きなまし: 毎ステップ 1 自由度変異 → 浅い局所最適化 → Metropolis 受理。
    let mut mut_delta: Vec<f64> = Vec::new();
    let mut qvina_grad: Vec<f64> = Vec::new(); // QVina2 スキップ判定の勾配バッファ（再利用）。
    for k in 0..n_steps {
        let mut trial = cur.conf.clone();
        apply_mutation(&mut trial, &mut rng, free_torsions, mut_style, &mut mut_delta);

        // QuickVina2 スキップ判定: 安価な勾配 1 回で「既知の斜面」と判定したら local 探索を省く。
        if let Some(hist) = history.as_mut() {
            value_grad(&trial, &mut qvina_grad);
            let dof = dof_vector(&trial, free_dofs);
            let g = project_grad(&qvina_grad, free_dofs);
            let significant = hist.is_new_well(&dof, &g);
            hist.push(dof, g);
            if !significant {
                continue; // 局所探索をスキップ（cur 不変＝棄却扱い）
            }
        }

        let res = minimize_free(trial, &value_grad, step_iters, free_dofs);
        let (res_pos, res_inter) = {
            let g = scratch_g.borrow();
            (g.scored_pos.clone(), g.inter)
        };
        cluster_insert(&mut pool, res.conf.clone(), res_inter, res_pos, params.cluster_rmsd, pool_cap);
        if patience > 0 {
            if res.energy < best_e - 1e-3 {
                best_e = res.energy;
                stale = 0;
            } else {
                stale += 1;
                if stale >= patience {
                    break;
                }
            }
        }

        // vina-mode は Vina と同じ固定温度（VINA_TEMP）。drift は params の焼きなまし（既定 1.0 一定）。
        let temp = if is_vina {
            VINA_TEMP
        } else {
            let frac = if n_steps > 1 {
                k as f64 / (n_steps - 1) as f64
            } else {
                1.0
            };
            params.temp_init * (params.temp_final / params.temp_init).powf(frac)
        };
        let accept = res.energy < cur.energy
            || rng.r#gen::<f64>() < (-(res.energy - cur.energy) / temp).exp();
        if accept {
            cur = res;
        }
    }

    let mut reps: Vec<(Conformation, f64)> = pool.into_iter().map(|(c, e, _)| (c, e)).collect();
    reps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let evals = scratch_g.borrow().evals;
    (reps, evals)
}

/// TPE（Tree-structured Parzen Estimator）の 1 run。`single_run` と同型の返り値
/// `(RMSD クラスタ済み代表群, 評価回数)` を返し、`dock_constrained` から同じ集約・polish に載る。
///
/// 流れ: `contact_start` で参照配座を得る → 前半 `warmup` 提案は一様ランダム、後半は良観測から
/// 構築した Parzen モデルでサンプル → 各提案を `minimize_free`（BFGS）で局所最適化 → 観測へ記録し
/// pool へクラスタ挿入。局所最適化・スコア・決定論（seeded `ChaCha8Rng`）は他モードと同一機構を再利用。
#[allow(clippy::too_many_arguments)]
fn single_run_tpe(
    lig: &LigandModel,
    scorer: &Scorer,
    params: &SearchParams,
    n_steps: usize,
    free_torsions: &[usize],
    free_dofs: &[usize],
    seed: u64,
) -> (Vec<(Conformation, f64)>, u64) {
    let scratch_g = RefCell::new(EvalScratch::default());
    let cap = params.hunt_cap;
    let value_grad =
        |c: &Conformation, g: &mut Vec<f64>| eval_value_grad(lig, scorer, &mut scratch_g.borrow_mut(), c, false, cap, g);

    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    // 参照配座（固定 torsion は 0 に保つ）。TPE のモデル空間は「この参照からの差」で定義。
    let reference = contact_start(
        lig, scorer, &mut rng, params.center, params.size, free_torsions, params.uniform_start, cap,
    );
    let (l_limit, g_limit) = tpe::bounds(params.center, params.size, free_torsions.len());
    let tp = params.tpe;
    let local_iters = if tp.local_iters > 0 { tp.local_iters } else { params.chain_iters };

    let pool_cap = params.polish_modes.max(params.n_modes);
    let mut pool: Vec<(Conformation, f64, Vec<Vec3>)> = Vec::new();
    let mut obs: Vec<tpe::Obs> = Vec::new();

    let warmup = (n_steps / tp.startup_frac.max(1)).max(1);
    let freq = (n_steps.saturating_sub(warmup) / tp.density_reuse.max(1)).max(1);
    let mut model: Option<tpe::TpeModel> = None;
    let mut since_rebuild = 0usize;

    // リガンド root 重心の絶対位置 = root_pivot + conf.position（`Conformation.position` は変位）。
    let root_pivot = lig.root_pivot();
    let half = params.size * 0.5;
    let in_box = |conf: &Conformation| -> bool {
        let abs = conf.position + root_pivot;
        (abs.x - params.center.x).abs() <= half.x
            && (abs.y - params.center.y).abs() <= half.y
            && (abs.z - params.center.z).abs() <= half.z
    };

    for p in 0..n_steps {
        // 提案ベクトル: 前半ウォームアップは一様、後半は Parzen モデル（観測不足時は一様へフォールバック）。
        let v = if p < warmup {
            tpe::warmup_sample(&mut rng, &l_limit, &g_limit)
        } else {
            if model.is_none() || since_rebuild >= freq {
                model = tpe::TpeModel::build(&obs, &tp, p, &l_limit, &g_limit);
                since_rebuild = 0;
            }
            since_rebuild += 1;
            match &model {
                Some(m) => m.sample(&mut rng, &tp, &l_limit, &g_limit),
                None => tpe::warmup_sample(&mut rng, &l_limit, &g_limit),
            }
        };

        // 提案を配座へ写し BFGS 局所最適化。
        let start = tpe::conf_from_vec(&reference, free_torsions, root_pivot, &v);
        let res = minimize_free(start, &value_grad, local_iters, free_dofs);
        // pool 保持は inter キー、TPE 観測（下）は objective（探索地形の学習元）＝分離。
        let (res_pos, res_inter) = {
            let g = scratch_g.borrow();
            (g.scored_pos.clone(), g.inter)
        };
        cluster_insert(&mut pool, res.conf.clone(), res_inter, res_pos, params.cluster_rmsd, pool_cap);

        // ボックス内なら観測へ記録（Parzen モデルの更新元）。
        if in_box(&res.conf) {
            obs.push(tpe::Obs {
                v: tpe::vec_from_conf(&reference, free_torsions, root_pivot, &res.conf),
                score: res.energy,
            });
        }
    }

    // 代表は (conf, inter) で返す。inter 昇順（=affinity 良い順）。
    let mut reps: Vec<(Conformation, f64)> = pool.into_iter().map(|(c, e, _)| (c, e)).collect();
    reps.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let evals = scratch_g.borrow().evals;
    (reps, evals)
}

/// 受容体と好ましく接触する初期配置を**複数試行から best を選ぶ**（Vina の randomize 流＝各 run に
/// productive な launch を与える）。位置・配向・torsion は random_in_box で完全ランダム化済み。
/// 選択エネルギーは hunt_cap 付き search_score（激突を上限化して過剰な棄却を避ける）。
/// 浮遊（E≈0, 勾配≈0 で局所最適化が無効）や激突開始を避け、深い基へ届きやすくする。
#[allow(clippy::too_many_arguments)]
fn contact_start<R: Rng>(
    lig: &LigandModel,
    scorer: &Scorer,
    rng: &mut R,
    center: Vec3,
    size: Vec3,
    free_torsions: &[usize],
    uniform: bool,
    cap: f64,
) -> Conformation {
    // 固定 torsion を 0（参照＝コア剛体）に戻した初期配置を作る。
    let sample = |rng: &mut R| -> Conformation {
        let mut c = lig.random_in_box(rng, center, size);
        lock_torsions(&mut c, free_torsions);
        c
    };
    // grid は勾配場が滑らかなため少なめの試行で十分。direct は接触棄却が要るので多め。
    let n_tries = if uniform { 20 } else { 60 };
    let energy = |c: &Conformation| scorer.search_score(&lig.scored_atoms(c), cap);
    let mut best_c = sample(rng);
    let mut best_e = energy(&best_c);
    for _ in 1..n_tries {
        // 好ましい接触（負エネルギー）に達したら早期終了（決定論・RNG 消費は seed 依存で再現）。
        if best_e < -1.0 {
            break;
        }
        let c = sample(rng);
        let e = energy(&c);
        if e < best_e {
            best_e = e;
            best_c = c;
        }
    }
    best_c
}

/// `free_torsions` に含まれない torsion を 0（参照）に固定する。
/// `free_torsions` は小さいため線形 `contains` で十分（HashSet 確保を避け contact_start の毎試行アロクを除去）。
fn lock_torsions(conf: &mut Conformation, free_torsions: &[usize]) {
    for (k, t) in conf.torsions.iter_mut().enumerate() {
        if !free_torsions.contains(&k) {
            *t = 0.0;
        }
    }
}

/// 1 自由度変異: 並進・配向・**自由** torsion のうち 1 つだけを動かす（固定 torsion は不動）。
/// entity 選択・torsion 摂動（±π）・apply は共通で、並進/配向の**振幅のみ `style` で分岐**する。
/// 乱数消費はスタイルごとに従来 `mutate`/`mutate_vina` と一致（決定論維持）。
fn apply_mutation<R: Rng>(
    conf: &mut Conformation,
    rng: &mut R,
    free_torsions: &[usize],
    style: MutationStyle,
    delta: &mut Vec<f64>,
) {
    use std::f64::consts::PI;
    // entity: 0=並進, 1=配向, 2..=自由 torsion
    let k = rng.gen_range(0..(2 + free_torsions.len()));
    // 再利用バッファをゼロ初期化（毎ステップの `vec![0.0; dim]` 確保を回避）。RNG 消費順は不変。
    delta.clear();
    delta.resize(conf.dim(), 0.0);
    let d = delta;
    match k {
        0 => match style {
            // drift: 各成分 ±2 Å。vina: amplitude · 単位球内乱点。
            MutationStyle::Drift => {
                for v in d.iter_mut().take(3) {
                    *v = rng.gen_range(-2.0..2.0);
                }
            }
            MutationStyle::Vina { .. } => {
                let (x, y, z) = random_in_unit_sphere(rng);
                d[0] = VINA_AMPLITUDE * x;
                d[1] = VINA_AMPLITUDE * y;
                d[2] = VINA_AMPLITUDE * z;
            }
        },
        1 => match style {
            // drift: 回転ベクトル各成分 ±1.0 rad。vina: (amplitude/gyration) · 単位球内乱点。
            MutationStyle::Drift => {
                for v in d.iter_mut().take(6).skip(3) {
                    *v = rng.gen_range(-1.0..1.0);
                }
            }
            MutationStyle::Vina { gyration } => {
                let s = if gyration > 1e-6 {
                    VINA_AMPLITUDE / gyration
                } else {
                    VINA_AMPLITUDE
                };
                let (x, y, z) = random_in_unit_sphere(rng);
                d[3] = s * x;
                d[4] = s * y;
                d[5] = s * z;
            }
        },
        _ => {
            // 1 つの自由 torsion を再サンプリング（±π）。両スタイル共通。
            let t = free_torsions[k - 2];
            d[6 + t] = rng.gen_range(-PI..PI);
        }
    }
    conf.apply_delta(d);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ligand::LigandModel;
    use isct_io::pdbqt;

    const LIG: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  0.00  0.00    +0.000 C
ATOM      3  O   LIG A   1       0.000   1.500   0.000  0.00  0.00    -0.300 OA
ENDROOT
TORSDOF 0
";

    fn translated(base: &Conformation, dx: f64) -> Conformation {
        let mut c = base.clone();
        c.apply_delta(&[dx, 0.0, 0.0, 0.0, 0.0, 0.0]);
        c
    }

    #[test]
    fn cluster_poses_separates_modes_and_keeps_lowest() {
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let id = Conformation::identity(lig.n_torsions());

        // モード A: 原点近傍に 2 ポーズ（0Å e=-5, 0.5Å e=-6）→ 同一クラスタ、代表は -6。
        // モード B: 10Å 離れ e=-4 → 別クラスタ。
        let cands = vec![
            (id.clone(), -5.0),
            (translated(&id, 0.5), -6.0),
            (translated(&id, 10.0), -4.0),
        ];
        let reps = cluster_poses(&lig, cands, 2.0, 9);
        assert_eq!(reps.len(), 2, "2 distinct modes expected");
        // 昇順: [0]=モードA代表(-6), [1]=モードB(-4)。
        assert!((reps[0].1 - (-6.0)).abs() < 1e-9, "lowest-energy rep of mode A");
        assert!((reps[1].1 - (-4.0)).abs() < 1e-9, "mode B");
        // 代表 A は 0.5Å 側（低エネルギー）であること。
        let p0 = lig.scored_positions(&reps[0].0);
        assert!((p0[0].x - 0.5).abs() < 1e-9);
    }

    #[test]
    fn cluster_poses_respects_cap() {
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let id = Conformation::identity(lig.n_torsions());
        // 5 個の十分離れたモード、cap=3 → 3 個（最低エネルギー側）。
        let cands: Vec<(Conformation, f64)> = (0..5)
            .map(|i| (translated(&id, 10.0 * i as f64), -(i as f64)))
            .collect();
        let reps = cluster_poses(&lig, cands, 2.0, 3);
        assert_eq!(reps.len(), 3);
    }
}
