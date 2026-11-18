//! リガンドのトポロジー表現とポーズ再構成（M2）。
//!
//! PDBQT の torsion 木（フラグメント）から、剛体並進(3)＋回転(四元数)＋torsion(n) を
//! 自由度とするポーズ再構成を行う。参照座標（入力ポーズ）を基準に、torsion を木順
//! （親→子）で適用してから剛体変換を掛ける。恒等変換は入力ポーズを再現する。

use std::collections::HashSet;

use ist_core::atom_types::XsType;
use ist_core::geometry::{distance, Vec3};
use ist_core::intra_lut;
use ist_core::scoring::{curl, eval_pair, pair_energy_grad, ScoredAtom, CUTOFF};
use ist_io::pdbqt::Molecule;

use crate::torsion::TorsionTerm;
use nalgebra::{Unit, UnitQuaternion};
use rand::Rng;

/// リガンドの自由度状態。`torsions[k]` は非 root フラグメント(k+1) の絶対角（0=参照）。
#[derive(Debug, Clone)]
pub struct Conformation {
    pub position: Vec3,
    pub orientation: UnitQuaternion<f64>,
    pub torsions: Vec<f64>,
}

impl Conformation {
    /// 恒等変換（参照ポーズ）。
    pub fn identity(n_torsions: usize) -> Self {
        Conformation {
            position: Vec3::zeros(),
            orientation: UnitQuaternion::identity(),
            torsions: vec![0.0; n_torsions],
        }
    }

    /// 自由度数 = 並進3 + 回転3 + torsion n。
    pub fn dim(&self) -> usize {
        6 + self.torsions.len()
    }

    /// 増分ベクトル（長さ dim）を適用する。
    /// `d[0..3]`=並進, `d[3..6]`=回転ベクトル(軸×角), `d[6..]`=torsion 増分。
    pub fn apply_delta(&mut self, d: &[f64]) {
        self.position += Vec3::new(d[0], d[1], d[2]);
        let rotvec = Vec3::new(d[3], d[4], d[5]);
        self.orientation = UnitQuaternion::from_scaled_axis(rotvec) * self.orientation;
        for (t, dt) in self.torsions.iter_mut().zip(&d[6..]) {
            *t += dt;
        }
    }
}

/// リガンドの実行時モデル（参照座標・木構造・スコア対象重原子）。
#[derive(Debug, Clone)]
pub struct LigandModel {
    /// 全原子の参照座標（水素含む）。
    ref_coords: Vec<Vec3>,
    /// フラグメントごとの回転結合 (parent側 atom idx, 自身側 atom idx)。root は None。
    bonds: Vec<Option<(usize, usize)>>,
    /// 各フラグメントの部分木に属する全原子インデックス。
    subtree: Vec<Vec<usize>>,
    /// root 原子の重心（剛体変換のピボット）。
    root_pivot: Vec3,
    /// スコア対象重原子の (全原子 idx, XS 型)。
    scored: Vec<(usize, XsType)>,
    /// 分子内エネルギーを評価する重原子ペア（scored インデックス空間）。
    /// 異なるフラグメント間（torsion で距離が変わる）かつ**結合グラフ距離 ≥5**（1-6 以上）。
    intra_pairs: Vec<(usize, usize)>,
    /// 1-5 ペア（結合グラフ距離がちょうど 4・異フラグメント）。AutoDock Vina は**数える**ので
    /// `intra_15_scale` の既定は 1.0 ＝ Vina と同じ。
    ///
    /// **段 105**: 1-5 距離は結合長・結合角とねじれ角 1〜2 本でほぼ決まるのに、Vina の表面距離
    /// `d = r − (R_i+R_j)` は炭素・窒素の半径を 1.8〜1.9 Å と置くので半径の和が 3.4〜3.8 Å に
    /// なる。1-5 ペアが syn 寄りに来ると r が 2.2〜3.2 Å まで下がって反発が立つ。
    /// **結晶リガンドそのものが最大 +2.08 kcal/mol の反発を受け、その中央 88% が 1-5 由来。**
    /// これは立体衝突ではなく結合でつながった原子の必然的な近さなので、係数で外せるようにした。
    intra_pairs_15: Vec<(usize, usize)>,
    /// 1-4 ペア（結合グラフ距離がちょうど 3・異フラグメント）。**AutoDock Vina は数えない。**
    /// `intra_14_scale` を掛けて分子内エネルギーに足す。既定 0.0 ＝ Vina と同じ。
    intra_pairs_14: Vec<(usize, usize)>,
    /// 1-5 ペアに掛ける係数。**既定 1.0 ＝ AutoDock Vina と同じ**（段 105 で診断した
    /// 見かけの反発を外したいときに 0.0 にする）。
    intra_15_scale: f64,
    /// 1-4 ペアに**反発項だけ**を入れる係数（B-1）。**既定 0.0。**
    ///
    /// `intra_14_scale` とは独立に効く。1-4 に Vina の分子内項をそのまま入れると引力も
    /// 一緒に入って折りたたみを促すが（段 90・92.1）、反発だけなら立体障害だけを表現できる。
    /// ビアリールのねじれはオルト位の立体反発で決まり、オルトの重原子どうしは結合グラフ
    /// 距離 4 ＝ 1-4 なので、**既定の Vina ではその反発が存在しない**（段 95.2）。
    /// 追加パラメータは無い（Vina の反発の重みと XS 半径をそのまま使う）。
    intra_14_repulsion_scale: f64,
    /// 1-4 ペアに掛ける係数（0.0 ＝ Vina 互換で数えない、1.0 ＝ 分子間と同じ重みで数える）。
    /// 力場が 1-4 相互作用を縮小して入れる慣例に対応する。**既定 0.0。**
    intra_14_scale: f64,
    /// フラグメント f（≥1）の部分木に属する **scored 原子インデックス**（torsion 勾配用）。
    scored_subtree: Vec<Vec<usize>>,
    n_torsions: usize,
    /// 各フラグメントの回転結合(BRANCH)を ligand.pdbqt の記述で保持（root=None）。torsion k ↔ `bond_atoms[k+1]`。
    /// `(親側 serial, 親側 name, 子側 serial, 子側 name)`＝BRANCH 行の 2 serial とその原子名。トレースヘッダで
    /// `torsions` 列が pdbqt のどの回転結合に対応するかを示す用（探索計算では未使用）。
    bond_atoms: Vec<Option<(u32, String, u32, String)>>,
    /// torsion k を回すと採点対象の重原子が動くか（A-4）。構築時に実際に回して測る。
    moves_scored: Vec<bool>,
    /// A-2 のねじれポテンシャル（torsion ごと、UFF の一般則で PDBQT から自己完結に決める）。
    /// 詳細は `crate::torsion`。
    torsion_terms: Vec<TorsionTerm>,
    /// ねじれポテンシャル全体に掛ける係数（0.0 ＝ 無効＝ AutoDock Vina と同じ。**既定 0.0**）。
    torsion_scale: f64,
    /// sp2–sp3（6 回対称）の規則を使うか（段 98.8 の案 (a)。**既定 true**）。
    /// false にすると sp3–sp3 とアミド様だけが働く。
    torsion_sp2_sp3: bool,
    /// ねじれ項を**最終の polish（厳密評価）だけ**で使うか（段 98.8 の案 (c)。**既定 false**）。
    /// true にすると大域探索中はねじれ項なしで動き、代表ポーズの精密化と順位付けにだけ入る。
    /// 地形の凸凹で探索が捕まる問題（段 98.4）を避け、採点の補正だけを取るための設計。
    torsion_polish_only: bool,
    /// 候補の保持と最終の順位を**全目的関数**で付けるか（段 101。**既定 false**＝従来どおり
    /// Vina の分子間（＋箱の罰則＋水、ただし最終順位は分子間だけ）で付ける）。
    rank_by_objective: bool,
}

impl LigandModel {
    pub fn new(mol: &Molecule) -> Self {
        let ref_coords: Vec<Vec3> = mol.atoms.iter().map(|a| a.pos).collect();
        let nf = mol.fragments.len();

        // 子リスト
        let mut children: Vec<Vec<usize>> = vec![Vec::new(); nf];
        for (i, f) in mol.fragments.iter().enumerate() {
            if let Some(p) = f.parent {
                children[p].push(i);
            }
        }
        // 部分木原子（DFS）
        let subtree: Vec<Vec<usize>> = (0..nf)
            .map(|f| {
                let mut acc = Vec::new();
                let mut stack = vec![f];
                while let Some(g) = stack.pop() {
                    acc.extend_from_slice(&mol.fragments[g].atoms);
                    stack.extend_from_slice(&children[g]);
                }
                acc
            })
            .collect();

        let bonds: Vec<Option<(usize, usize)>> = mol.fragments.iter().map(|f| f.bond).collect();

        // 回転結合の pdbqt 記述（BRANCH の 2 serial ＋原子名）。トレースヘッダで torsion→pdbqt 対応を出す用。
        let bond_atoms: Vec<Option<(u32, String, u32, String)>> = mol
            .fragments
            .iter()
            .map(|f| {
                f.bond.map(|(a0, a1)| {
                    (
                        mol.atoms[a0].serial,
                        mol.atoms[a0].name.clone(),
                        mol.atoms[a1].serial,
                        mol.atoms[a1].name.clone(),
                    )
                })
            })
            .collect();

        let root_atoms = &mol.fragments[0].atoms;
        let root_pivot = root_atoms.iter().fold(Vec3::zeros(), |s, &i| s + ref_coords[i])
            / root_atoms.len().max(1) as f64;

        let xs = mol.xs_types();
        let scored: Vec<(usize, XsType)> = xs
            .iter()
            .enumerate()
            .filter_map(|(i, x)| x.map(|t| (i, t)))
            .collect();

        let (intra_pairs, intra_pairs_14, intra_pairs_15) =
            Self::build_intra_pairs(mol, &ref_coords, &scored);

        // 各フラグメントの部分木に属する scored 原子（all-atom idx → scored idx の逆引き）。
        let mut all_to_scored: Vec<Option<usize>> = vec![None; ref_coords.len()];
        for (k, &(ai, _)) in scored.iter().enumerate() {
            all_to_scored[ai] = Some(k);
        }
        let scored_subtree: Vec<Vec<usize>> = subtree
            .iter()
            .map(|atoms| atoms.iter().filter_map(|&ai| all_to_scored[ai]).collect())
            .collect();

        let mut m = LigandModel {
            ref_coords,
            bonds,
            subtree,
            root_pivot,
            scored,
            intra_pairs,
            intra_pairs_14,
            intra_pairs_15,
            intra_15_scale: 1.0,
            intra_14_scale: 0.0,
            intra_14_repulsion_scale: 0.0,
            scored_subtree,
            n_torsions: nf - 1,
            bond_atoms,
            moves_scored: Vec::new(),
            torsion_terms: Vec::new(),
            torsion_scale: 0.0,
            torsion_sp2_sp3: true,
            torsion_polish_only: false,
            rank_by_objective: false,
        };
        // A-4: torsion ごとに「採点対象の重原子が動くか」を**実際に回して**測る。
        // 試験角は 0.7 rad（0 や π の近傍を避ける）。軸上の原子は動かないので、
        // 部分木の構成では判定できない（末端 -OH の O は部分木にあるが軸上）。
        let base = m.scored_positions(&Conformation::identity(m.n_torsions));
        m.moves_scored = (0..m.n_torsions)
            .map(|k| {
                let mut c = Conformation::identity(m.n_torsions);
                c.torsions[k] = 0.7;
                m.scored_positions(&c)
                    .iter()
                    .zip(&base)
                    .any(|(a, b)| (a - b).norm() > 1e-9)
            })
            .collect();
        // ねじれポテンシャルの係数（A-2）。`torsion_scale` が 0 の間は評価されないが、
        // 構築は 1 回で済み費用も小さいので常に用意する（診断にも使う）。
        // 符号と参照角は**実際に微小回転を適用して測る**ので、`world_coords` が
        // 完成してからでないと決められない。そのため構築後に入れる。
        let moves_scored = m.moves_scored.clone();
        let terms = crate::torsion::build_terms(
            mol,
            &m.ref_coords.clone(),
            &m.bonds.clone(),
            m.n_torsions,
            &moves_scored,
            |k, t| {
                let mut c = Conformation::identity(m.n_torsions);
                c.torsions[k] = t;
                m.world_coords(&c)
            },
        );
        m.torsion_terms = terms;
        m
    }

    /// ねじれポテンシャル（A-2）の全体係数を設定する。**0.0 ＝ 無効（AutoDock Vina と同じ）**。
    pub fn set_torsion_scale(&mut self, scale: f64) {
        self.torsion_scale = scale;
    }

    /// 現在のねじれポテンシャル係数。
    pub fn torsion_scale(&self) -> f64 {
        self.torsion_scale
    }

    /// torsion ごとのねじれポテンシャル（診断・記録用）。
    pub fn torsion_terms(&self) -> &[TorsionTerm] {
        &self.torsion_terms
    }

    /// sp2–sp3 の規則を使うかを設定する（既定 true）。
    pub fn set_torsion_sp2_sp3(&mut self, on: bool) {
        self.torsion_sp2_sp3 = on;
    }

    /// ねじれ項を最終の polish だけで使うかを設定する（既定 false）。
    pub fn set_torsion_polish_only(&mut self, on: bool) {
        self.torsion_polish_only = on;
    }

    /// ねじれ項を最終の polish だけで使うか。
    pub fn torsion_polish_only(&self) -> bool {
        self.torsion_polish_only
    }

    /// 候補の保持と最終の順位を全目的関数で付けるかを設定する（既定 false）。
    pub fn set_rank_by_objective(&mut self, on: bool) {
        self.rank_by_objective = on;
    }

    /// 候補の保持と最終の順位を全目的関数で付けるか。
    pub fn rank_by_objective(&self) -> bool {
        self.rank_by_objective
    }

    /// torsion k のねじれ項を評価するか（検証を通った種別で、かつ規則が有効なもの）。
    #[inline]
    fn torsion_term_on(&self, k: usize) -> bool {
        let t = &self.torsion_terms[k];
        if !self.torsion_sp2_sp3 && t.kind == crate::torsion::TorsionKind::Sp2Sp3 {
            return false;
        }
        true
    }

    /// いま実際に評価される torsion の本数（ログ用）。
    pub fn n_torsions_active(&self) -> usize {
        (0..self.n_torsions)
            .filter(|&k| self.torsion_terms[k].is_active() && self.torsion_term_on(k))
            .count()
    }

    /// ねじれポテンシャルの値（kcal/mol）。`torsion_scale` が 0 なら 0 を返す。
    pub fn torsional(&self, conf: &Conformation) -> f64 {
        if self.torsion_scale == 0.0 {
            return 0.0;
        }
        let mut e = 0.0;
        for (k, term) in self.torsion_terms.iter().enumerate() {
            if !self.torsion_term_on(k) {
                continue;
            }
            e += term.value_grad(conf.torsions[k]).0;
        }
        self.torsion_scale * e
    }

    /// ねじれポテンシャルの値と、**DOF 勾配への加算**。`grad` は長さ `6 + n_torsions`
    /// （`Conformation::apply_delta` と同じ並び）で、torsion k は `grad[6 + k]`。
    ///
    /// 原子座標を経由しないので連鎖律が要らず、費用は torsion 本数にしか比例しない。
    pub fn torsional_value_grad(&self, conf: &Conformation, grad: &mut [f64]) -> f64 {
        if self.torsion_scale == 0.0 {
            return 0.0;
        }
        let mut e = 0.0;
        for (k, term) in self.torsion_terms.iter().enumerate() {
            if !self.torsion_term_on(k) {
                continue;
            }
            let (v, g) = term.value_grad(conf.torsions[k]);
            e += v;
            if let Some(slot) = grad.get_mut(6 + k) {
                *slot += self.torsion_scale * g;
            }
        }
        self.torsion_scale * e
    }

    /// トレース `torsions` 列（`free_torsions` 順）の各 torsion が ligand.pdbqt のどの回転結合(BRANCH)に対応するかを
    /// 示すヘッダ行を返す。列 i（1 始まり）→ フラグメント `free_torsions[i-1]+1` の BRANCH（親 serial-子 serial）。
    /// 外部ツールが内部 torsion 順を pdbqt の回転結合へ対応付けて構造再生できるようにする。
    pub fn torsion_trace_header(&self, free_torsions: &[usize]) -> Vec<String> {
        free_torsions
            .iter()
            .enumerate()
            .map(|(i, &t)| match self.bond_atoms.get(t + 1).and_then(|b| b.as_ref()) {
                Some((ps, pn, cs, cn)) => format!(
                    "# TORSION {} serials {}-{} atoms {}-{}",
                    i + 1,
                    ps,
                    cs,
                    pn.trim(),
                    cn.trim()
                ),
                None => format!("# TORSION {} (unknown bond)", i + 1),
            })
            .collect()
    }

    /// 分子内エネルギー対象ペアを事前計算する。返り値は `(1-5 以上, 1-4)` の 2 本。
    ///
    /// どちらも**異なるフラグメント間**（torsion で距離が変化するペア）に限る。
    /// 1 本目は**結合グラフ距離 ≥4**（1-2・1-3・1-4 を除外）で、**AutoDock Vina の規則に一致する**
    /// （段 88 で `vina --score_only` の内部エネルギーと突き合わせて確認。結晶ポーズ 465 件で
    /// 絶対差の中央 0.075 kcal/mol・91% が 0.2 以内）。
    /// 2 本目は**結合グラフ距離がちょうど 3**（1-4）で、`intra_14_scale` を掛けて足す。
    ///
    /// 1-4 を既定で数えないのは AutoDock Vina に合わせるためだが、**1-4 距離は結合長・結合角と
    /// ねじれ角でほぼ決まるので、1-4 の立体反発は実質的に粗いねじれポテンシャルとして働く。**
    /// Vina にはねじれポテンシャルが無いため、ここを係数で入れられるようにしてある。
    fn build_intra_pairs(
        mol: &Molecule,
        ref_coords: &[Vec3],
        scored: &[(usize, XsType)],
    ) -> (Vec<(usize, usize)>, Vec<(usize, usize)>, Vec<(usize, usize)>) {
        let n = scored.len();
        let orig: Vec<usize> = scored.iter().map(|&(i, _)| i).collect();

        // 重原子間の結合隣接（scored インデックス空間）。
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for a in 0..n {
            for b in (a + 1)..n {
                let (ia, ib) = (orig[a], orig[b]);
                let ea = mol.atoms[ia].ad.element();
                let eb = mol.atoms[ib].ad.element();
                let r = distance(&ref_coords[ia], &ref_coords[ib]);
                if r < (ea.covalent_radius() + eb.covalent_radius()) * 1.1 {
                    adj[a].push(b);
                    adj[b].push(a);
                }
            }
        }

        // 各原子から結合グラフ距離 4 までを幅優先で測る（5 以上は `usize::MAX` のまま）。
        // 1-2・1-3 は常に除外、ちょうど 3（1-4）と ちょうど 4（1-5）は別枠、5 以上が残り。
        const FAR: usize = 5;
        let mut gdist: Vec<Vec<usize>> = vec![vec![usize::MAX; n]; n];
        for a in 0..n {
            let d = &mut gdist[a];
            d[a] = 0;
            let mut q = std::collections::VecDeque::new();
            q.push_back(a);
            while let Some(u) = q.pop_front() {
                if d[u] >= FAR - 1 {
                    continue;
                }
                for &v in &adj[u] {
                    if d[v] == usize::MAX {
                        d[v] = d[u] + 1;
                        q.push_back(v);
                    }
                }
            }
        }

        let mut pairs = Vec::new();
        let mut pairs14 = Vec::new();
        let mut pairs15 = Vec::new();
        for a in 0..n {
            for b in (a + 1)..n {
                if mol.atom_fragment[orig[a]] == mol.atom_fragment[orig[b]] {
                    continue; // 同一フラグメント内は torsion で距離が変わらない。
                }
                match gdist[a][b] {
                    0..=2 => {}                       // 1-2・1-3 は常に除外。
                    3 => pairs14.push((a, b)),        // 1-4 は別枠（既定 0 ＝ Vina）。
                    4 => pairs15.push((a, b)),        // 1-5 は別枠（既定 1 ＝ Vina）。
                    _ => pairs.push((a, b)),          // 1-6 以上。
                }
            }
        }
        (pairs, pairs14, pairs15)
    }

    /// 1-4 ペアに掛ける係数を設定する（0.0 ＝ AutoDock Vina 互換、1.0 ＝ 分子間と同じ重み）。
    pub fn set_intra_14_scale(&mut self, scale: f64) {
        self.intra_14_scale = scale;
    }

    /// 現在の 1-4 係数。
    pub fn intra_14_scale(&self) -> f64 {
        self.intra_14_scale
    }

    /// 1-4 ペアに反発項だけを入れる係数（B-1）を設定する。**0.0 ＝ 無効（既定）。**
    pub fn set_intra_14_repulsion_scale(&mut self, scale: f64) {
        self.intra_14_repulsion_scale = scale;
    }

    /// 現在の 1-4 反発係数。
    pub fn intra_14_repulsion_scale(&self) -> f64 {
        self.intra_14_repulsion_scale
    }

    /// 1-4 ペアの数（診断用）。
    pub fn n_intra_pairs_14(&self) -> usize {
        self.intra_pairs_14.len()
    }

    /// 1-5 ペアに掛ける係数を設定する（**1.0 ＝ AutoDock Vina 互換（既定）**、0.0 で外す）。
    pub fn set_intra_15_scale(&mut self, scale: f64) {
        self.intra_15_scale = scale;
    }

    /// 現在の 1-5 係数。
    pub fn intra_15_scale(&self) -> f64 {
        self.intra_15_scale
    }

    /// 1-5 ペアの数（診断用）。
    pub fn n_intra_pairs_15(&self) -> usize {
        self.intra_pairs_15.len()
    }

    /// 1-5 の重原子ペア。分解・診断用の読み取り専用。
    pub fn intra_pairs_15(&self) -> &[(usize, usize)] {
        &self.intra_pairs_15
    }

    /// 分子内項に入る重原子ペア（**1-6 以上**。フラグメント間）。分解・診断用の読み取り専用。
    pub fn intra_pairs(&self) -> &[(usize, usize)] {
        &self.intra_pairs
    }

    /// 1-4 の重原子ペア（既定では分子内項に入らない）。分解・診断用の読み取り専用。
    pub fn intra_pairs_14(&self) -> &[(usize, usize)] {
        &self.intra_pairs_14
    }

    /// 分子内エネルギー（フラグメント間の非結合重原子ペアの Vina スコア和）。
    /// `cap`（hunt_cap）で各ペアの正エネルギー（激突）を上限化する（探索用。無効化は `f64::INFINITY`）。
    pub fn intramolecular(&self, atoms: &[ScoredAtom], cap: f64) -> f64 {
        let mut e = 0.0;
        for &(a, b) in &self.intra_pairs {
            let r = distance(&atoms[a].pos, &atoms[b].pos);
            if r < CUTOFF {
                let pe = eval_pair(atoms[a].xs, atoms[b].xs, r).weighted_sum();
                e += curl(pe, cap).0;
            }
        }
        // 1-5 ペア（既定 1.0 ＝ Vina と同じ。段 105 の見かけの反発を外すときは 0.0）。
        if self.intra_15_scale != 0.0 {
            for &(a, b) in &self.intra_pairs_15 {
                let r = distance(&atoms[a].pos, &atoms[b].pos);
                if r < CUTOFF {
                    let pe = eval_pair(atoms[a].xs, atoms[b].xs, r).weighted_sum();
                    e += self.intra_15_scale * curl(pe, cap).0;
                }
            }
        }
        // 1-4 ペア（既定 0.0 で寄与なし）。打ち切り（curl）を先に当ててから係数を掛けるので、
        // 係数 1.0 のとき 1-5 以上と同じ扱いになる。
        if self.intra_14_scale != 0.0 {
            for &(a, b) in &self.intra_pairs_14 {
                let r = distance(&atoms[a].pos, &atoms[b].pos);
                if r < CUTOFF {
                    let pe = eval_pair(atoms[a].xs, atoms[b].xs, r).weighted_sum();
                    e += self.intra_14_scale * curl(pe, cap).0;
                }
            }
        }
        // 1-4 の反発だけの項（B-1。既定 0.0 で寄与なし）。打ち切りは掛けない
        // （反発は本来いくらでも大きくなる量で、上限を掛けると立体障害が消える）。
        if self.intra_14_repulsion_scale != 0.0 {
            for &(a, b) in &self.intra_pairs_14 {
                let r = distance(&atoms[a].pos, &atoms[b].pos);
                if r < CUTOFF {
                    e += self.intra_14_repulsion_scale
                        * ist_core::scoring::pair_repulsion(atoms[a].xs, atoms[b].xs, r);
                }
            }
        }
        e
    }

    /// 1-4 の反発だけの項（B-1）の値と力を加算する。打ち切り（curl）は掛けない
    /// （反発は本来いくらでも大きくなる量で、上限を掛けると立体障害が消える）。
    /// 2 つの `*_value_grad` から共通で呼ぶ（反発は解析式が安いので LUT 版でも同じ実装）。
    #[inline]
    fn add_intra_14_repulsion(&self, atoms: &[ScoredAtom], force: &mut [Vec3]) -> f64 {
        if self.intra_14_repulsion_scale == 0.0 {
            return 0.0;
        }
        let w = self.intra_14_repulsion_scale;
        let mut e = 0.0;
        for &(a, b) in &self.intra_pairs_14 {
            let diff = atoms[a].pos - atoms[b].pos;
            let r = diff.norm();
            if r >= CUTOFF || r <= 1e-8 {
                continue;
            }
            let (v, dvdr) = ist_core::scoring::pair_repulsion_grad(atoms[a].xs, atoms[b].xs, r);
            if v == 0.0 && dvdr == 0.0 {
                continue;
            }
            e += w * v;
            let f = diff * (w * dvdr / r);
            force[a] -= f;
            force[b] += f;
        }
        e
    }

    /// 分子内エネルギーと、各 scored 原子に働く力（`force[k] -= dE/dpos_k`）を加算する。
    /// `cap`（hunt_cap）で各ペアの正エネルギーを上限化し、勾配も curl のスケールを掛ける。
    pub fn intramolecular_value_grad(&self, atoms: &[ScoredAtom], force: &mut [Vec3], cap: f64) -> f64 {
        let mut e = 0.0;
        for (k, pairs) in [&self.intra_pairs, &self.intra_pairs_14, &self.intra_pairs_15]
            .into_iter()
            .enumerate()
        {
            // k=0 は 1-6 以上（係数 1）、k=1 は 1-4（既定 0）、k=2 は 1-5（既定 1）。
            let w = match k {
                0 => 1.0,
                1 => self.intra_14_scale,
                _ => self.intra_15_scale,
            };
            if w == 0.0 {
                continue;
            }
            for &(a, b) in pairs {
                let diff = atoms[a].pos - atoms[b].pos;
                let r = diff.norm();
                if r < CUTOFF && r > 1e-8 {
                    let (v, dvdr) = pair_energy_grad(atoms[a].xs, atoms[b].xs, r);
                    let (cv, scale) = curl(v, cap);
                    e += w * cv;
                    // dE/dpos_a = dvdr*diff/r, dE/dpos_b = -それ。force = -dE/dpos。curl の勾配スケール込み。
                    let f = diff * (w * dvdr / r * scale);
                    force[a] -= f;
                    force[b] += f;
                }
            }
        }
        e += self.add_intra_14_repulsion(atoms, force);
        e
    }

    /// 分子内エネルギー（**探索用・LUT 版**）。厳密 exp の代わりに事前計算テーブル（`intra_lut`）を r² で引く。
    /// exp/sqrt/div を回さず高速。curl は据置。polish/最終スコアは厳密版 `intramolecular` を使うこと。
    pub fn intramolecular_fast(&self, atoms: &[ScoredAtom], cap: f64) -> f64 {
        let t = intra_lut::table();
        let mut e = 0.0;
        for &(a, b) in &self.intra_pairs {
            let r2 = (atoms[a].pos - atoms[b].pos).norm_squared();
            let pe = t.value(atoms[a].xs, atoms[b].xs, r2); // r²≥cutoff² は 0
            e += curl(pe, cap).0;
        }
        if self.intra_15_scale != 0.0 {
            for &(a, b) in &self.intra_pairs_15 {
                let r2 = (atoms[a].pos - atoms[b].pos).norm_squared();
                let pe = t.value(atoms[a].xs, atoms[b].xs, r2);
                e += self.intra_15_scale * curl(pe, cap).0;
            }
        }
        if self.intra_14_scale != 0.0 {
            for &(a, b) in &self.intra_pairs_14 {
                let r2 = (atoms[a].pos - atoms[b].pos).norm_squared();
                let pe = t.value(atoms[a].xs, atoms[b].xs, r2);
                e += self.intra_14_scale * curl(pe, cap).0;
            }
        }
        if self.intra_14_repulsion_scale != 0.0 {
            for &(a, b) in &self.intra_pairs_14 {
                let r2 = (atoms[a].pos - atoms[b].pos).norm_squared();
                if r2 < CUTOFF * CUTOFF {
                    e += self.intra_14_repulsion_scale
                        * ist_core::scoring::pair_repulsion(
                            atoms[a].xs,
                            atoms[b].xs,
                            r2.sqrt(),
                        );
                }
            }
        }
        e
    }

    /// 分子内の値と力（**探索用・LUT 版**）。`intramolecular_value_grad` の LUT 版。
    /// `d(value)/d(r²)` を引き、力は `2·dvalue_dr2·diff`（sqrt/div 不要, curl スケール込み）。
    pub fn intramolecular_fast_value_grad(
        &self,
        atoms: &[ScoredAtom],
        force: &mut [Vec3],
        cap: f64,
    ) -> f64 {
        let t = intra_lut::table();
        let mut e = 0.0;
        for (k, pairs) in [&self.intra_pairs, &self.intra_pairs_14, &self.intra_pairs_15]
            .into_iter()
            .enumerate()
        {
            let w = match k {
                0 => 1.0,
                1 => self.intra_14_scale,
                _ => self.intra_15_scale,
            };
            if w == 0.0 {
                continue;
            }
            for &(a, b) in pairs {
                let diff = atoms[a].pos - atoms[b].pos;
                let r2 = diff.norm_squared();
                let (v, dvdr2) = t.value_grad_r2(atoms[a].xs, atoms[b].xs, r2);
                if v == 0.0 && dvdr2 == 0.0 {
                    continue; // カットオフ外
                }
                let (cv, scale) = curl(v, cap);
                e += w * cv;
                // dE/dpos_a = dvdr2·d(r²)/dpos_a = dvdr2·2·diff。force=-dE/dpos。curl スケール込み。
                let f = diff * (w * 2.0 * dvdr2 * scale);
                force[a] -= f;
                force[b] += f;
            }
        }
        e += self.add_intra_14_repulsion(atoms, force);
        e
    }

    /// world 座標（全原子）に加え、各非 root フラグメントの **world 系**回転軸・ピボットを返す。
    /// torsion 勾配の連鎖則に使う。`world_coords` と同一のポーズ再構成を行う。
    pub fn world_coords_frames(&self, conf: &Conformation) -> (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>) {
        let (mut pos, mut world_axis, mut world_pivot) = (Vec::new(), Vec::new(), Vec::new());
        let (mut ref_axis, mut ref_pivot) = (Vec::new(), Vec::new());
        self.world_coords_frames_into(
            conf, &mut pos, &mut ref_axis, &mut ref_pivot, &mut world_axis, &mut world_pivot,
        );
        (pos, world_axis, world_pivot)
    }

    /// `world_coords_frames` のゼロアロケーション版: 与えられた再利用バッファへ書き込む。
    /// `ref_axis`/`ref_pivot` は中間バッファ（呼び出し側は無視してよい）。
    pub fn world_coords_frames_into(
        &self,
        conf: &Conformation,
        pos: &mut Vec<Vec3>,
        ref_axis: &mut Vec<Vec3>,
        ref_pivot: &mut Vec<Vec3>,
        world_axis: &mut Vec<Vec3>,
        world_pivot: &mut Vec<Vec3>,
    ) {
        let nf = self.bonds.len();
        pos.clear();
        pos.extend_from_slice(&self.ref_coords);
        ref_axis.clear();
        ref_axis.resize(nf, Vec3::zeros());
        ref_pivot.clear();
        ref_pivot.resize(nf, Vec3::zeros());

        for f in 1..nf {
            let (a0, a1) = self.bonds[f].expect("non-root fragment must have a bond");
            let axis = pos[a1] - pos[a0];
            let an = axis.norm();
            // 親 torsion 適用後・子 torsion はピボット/軸を動かさないため、ここが最終 ref 系の値。
            ref_axis[f] = if an < 1e-9 { Vec3::zeros() } else { axis / an };
            ref_pivot[f] = pos[a1];

            let angle = conf.torsions[f - 1];
            if angle != 0.0 && an >= 1e-9 {
                let q = UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle);
                let pivot = pos[a1];
                for &i in &self.subtree[f] {
                    pos[i] = pivot + q * (pos[i] - pivot);
                }
            }
        }

        // 剛体変換（root 重心を中心に回転 → 並進）。
        for p in pos.iter_mut() {
            *p = self.root_pivot + conf.orientation * (*p - self.root_pivot) + conf.position;
        }
        // ref 系の軸・ピボットを world 系へ。
        world_axis.clear();
        world_axis.extend(ref_axis.iter().map(|a| conf.orientation * a));
        world_pivot.clear();
        world_pivot.extend(
            ref_pivot
                .iter()
                .map(|p| self.root_pivot + conf.orientation * (p - self.root_pivot) + conf.position),
        );
    }

    /// 事前計算した world 座標から scored 原子（位置＋XS 型）を作る（再計算を避ける）。
    pub fn scored_atoms_at(&self, world_pos: &[Vec3]) -> Vec<ScoredAtom> {
        let mut atoms = Vec::new();
        self.scored_atoms_into(world_pos, &mut atoms);
        atoms
    }

    /// `scored_atoms_at` のゼロアロケーション版: 再利用バッファ `atoms` へ書き込む。
    pub fn scored_atoms_into(&self, world_pos: &[Vec3], atoms: &mut Vec<ScoredAtom>) {
        atoms.clear();
        atoms.extend(self.scored.iter().map(|&(i, xs)| ScoredAtom { pos: world_pos[i], xs }));
    }

    /// scored 原子に働く力 `force[k] = -dE/dpos_k` から、DOF 勾配（長さ dim）を連鎖則で求める。
    /// `world_pos`/`world_axis`/`world_pivot` は `world_coords_frames` の出力。
    pub fn dof_gradient(
        &self,
        conf: &Conformation,
        world_pos: &[Vec3],
        world_axis: &[Vec3],
        world_pivot: &[Vec3],
        force: &[Vec3],
    ) -> Vec<f64> {
        let mut grad = Vec::new();
        self.dof_gradient_into(conf, world_pos, world_axis, world_pivot, force, &mut grad);
        grad
    }

    /// `dof_gradient` のゼロアロケーション版: 再利用バッファ `grad`（長さ dim）へ書き込む。
    pub fn dof_gradient_into(
        &self,
        conf: &Conformation,
        world_pos: &[Vec3],
        world_axis: &[Vec3],
        world_pivot: &[Vec3],
        force: &[Vec3],
        grad: &mut Vec<f64>,
    ) {
        grad.clear();
        grad.resize(6 + self.n_torsions, 0.0);
        // 回転のピボットは world 系の root（root_pivot + 並進）。
        let rot_pivot = self.root_pivot + conf.position;

        let mut g_trans = Vec3::zeros();
        let mut g_rot = Vec3::zeros();
        for (k, &(ai, _)) in self.scored.iter().enumerate() {
            let g = -force[k]; // dE/dpos
            g_trans += g;
            g_rot += (world_pos[ai] - rot_pivot).cross(&g);
        }
        grad[0] = g_trans.x;
        grad[1] = g_trans.y;
        grad[2] = g_trans.z;
        grad[3] = g_rot.x;
        grad[4] = g_rot.y;
        grad[5] = g_rot.z;

        for f in 1..self.bonds.len() {
            let mut acc = Vec3::zeros();
            for &k in &self.scored_subtree[f] {
                let ai = self.scored[k].0;
                let g = -force[k];
                acc += (world_pos[ai] - world_pivot[f]).cross(&g);
            }
            grad[6 + f - 1] = world_axis[f].dot(&acc);
        }
    }

    pub fn n_torsions(&self) -> usize {
        self.n_torsions
    }

    /// 分子内エネルギー評価ペア数。
    pub fn n_intra_pairs(&self) -> usize {
        self.intra_pairs.len()
    }

    pub fn n_scored(&self) -> usize {
        self.scored.len()
    }

    /// 全原子の参照座標数。
    pub fn n_atoms(&self) -> usize {
        self.ref_coords.len()
    }

    /// 全原子の参照座標（入力ポーズ）。Phase 2 で子コアの参照 world 座標を取るのに使う。
    pub fn ref_coords(&self) -> &[Vec3] {
        &self.ref_coords
    }

    /// 剛体変換のピボット（root 重心）。Phase 2 で剛体配置を Conformation に変換するのに使う。
    pub fn root_pivot(&self) -> Vec3 {
        self.root_pivot
    }

    /// スコア対象重原子それぞれの全原子インデックス（コア/R 基マスク構築用）。
    pub fn scored_all_indices(&self) -> Vec<usize> {
        self.scored.iter().map(|&(i, _)| i).collect()
    }

    /// R 基 torsion のインデックス（0-based, torsion k ↔ フラグメント k+1）。
    /// 「回しても **コア原子を 1 つも動かさない**」torsion のみ＝部分木にコアを含まないもの。
    /// これにより、コアが root でなくても（Meeko の root 選択に依存せず）コアを固定したまま
    /// R 基だけを最適化できる。`core_atoms` は全原子インデックス集合。
    pub fn rgroup_torsions(&self, core_atoms: &HashSet<usize>) -> Vec<usize> {
        (0..self.n_torsions)
            .filter(|&k| {
                let f = k + 1; // torsion k は非 root フラグメント k+1 に対応
                !self.subtree[f].iter().any(|i| core_atoms.contains(i))
            })
            .filter(|&k| self.torsion_moves_scored_atom(k))
            .collect()
    }

    /// torsion `k` を回すと**採点対象の重原子が 1 つでも動くか**（A-4）。
    ///
    /// **部分木に採点対象の原子があるかでは判定できない。** 回転結合の自分側の原子は
    /// **回転軸の上に乗っている**ので、部分木に入っていても動かない。末端の -OH では
    /// 部分木は {O, H} で O は採点対象だが軸上にあり、動くのは水素だけになる。
    /// そのため構築時に**実際に回して変位を測った結果**（`moves_scored`）を使う。
    ///
    /// エンジンは水素に XS 型を与えず全項から除外するので、ここが false の自由度は
    /// 分子間項・分子内項・箱の罰則のいずれも変えられない。
    #[inline]
    pub fn torsion_moves_scored_atom(&self, k: usize) -> bool {
        self.moves_scored[k]
    }

    /// 探索で動かす意味のある torsion の添字（A-4）。
    ///
    /// 採点対象の重原子を 1 つも動かさない torsion を**探索空間から外す**。段 92.2 の実測では
    /// Astex 85 標的の回転結合 476 本のうち **91 本（19%）**がこれに当たり、探索はそこへ
    /// 提案を費やしていた。
    ///
    /// **報告する親和性は変わらない。** Vina の torsion entropy 補正の除数
    /// （`Molecule::num_tors_vina`）は末端基への回転結合を 0.5 本として数える定義で、
    /// これは AutoDock Vina のトポロジー量なのでそのまま使う。
    pub fn active_torsions(&self) -> Vec<usize> {
        (0..self.n_torsions)
            .filter(|&k| self.torsion_moves_scored_atom(k))
            .collect()
    }

    /// 採点対象の重原子を動かさない torsion の本数（診断・ログ用）。
    pub fn n_null_torsions(&self) -> usize {
        (0..self.n_torsions)
            .filter(|&k| !self.torsion_moves_scored_atom(k))
            .count()
    }

    /// 与えられた conformation の全原子 world 座標を再構成する。
    pub fn world_coords(&self, conf: &Conformation) -> Vec<Vec3> {
        let mut pos = Vec::new();
        self.world_coords_into(conf, &mut pos);
        pos
    }

    /// `world_coords` のゼロアロケーション版: 再利用バッファ `pos` へ書き込む。
    pub fn world_coords_into(&self, conf: &Conformation, pos: &mut Vec<Vec3>) {
        pos.clear();
        pos.extend_from_slice(&self.ref_coords);

        // torsion をフラグメント木順（親→子, 添字昇順）に適用。
        for f in 1..self.bonds.len() {
            let angle = conf.torsions[f - 1];
            if angle == 0.0 {
                continue;
            }
            let (a0, a1) = self.bonds[f].expect("non-root fragment must have a bond");
            let axis = pos[a1] - pos[a0];
            if axis.norm() < 1e-9 {
                continue;
            }
            let q = UnitQuaternion::from_axis_angle(&Unit::new_normalize(axis), angle);
            let pivot = pos[a1];
            for &i in &self.subtree[f] {
                pos[i] = pivot + q * (pos[i] - pivot);
            }
        }

        // 剛体変換（root 重心を中心に回転 → 並進）。
        for p in pos.iter_mut() {
            *p = self.root_pivot + conf.orientation * (*p - self.root_pivot) + conf.position;
        }
    }

    /// スコア対象重原子（位置＋XS 型）を返す。
    pub fn scored_atoms(&self, conf: &Conformation) -> Vec<ScoredAtom> {
        let pos = self.world_coords(conf);
        self.scored
            .iter()
            .map(|&(i, xs)| ScoredAtom { pos: pos[i], xs })
            .collect()
    }

    /// スコア対象重原子の world 座標のみ（RMSD 用）。
    pub fn scored_positions(&self, conf: &Conformation) -> Vec<Vec3> {
        let pos = self.world_coords(conf);
        self.scored.iter().map(|&(i, _)| pos[i]).collect()
    }

    /// 参照ポーズ（恒等変換）の重原子 world 座標。
    pub fn reference_scored_positions(&self) -> Vec<Vec3> {
        self.scored.iter().map(|&(i, _)| self.ref_coords[i]).collect()
    }

    /// 探索ボックス内にランダム配置した初期 conformation を作る。
    pub fn random_in_box<R: Rng>(&self, rng: &mut R, center: Vec3, size: Vec3) -> Conformation {
        let target = Vec3::new(
            center.x + (rng.r#gen::<f64>() - 0.5) * size.x,
            center.y + (rng.r#gen::<f64>() - 0.5) * size.y,
            center.z + (rng.r#gen::<f64>() - 0.5) * size.z,
        );
        Conformation {
            position: target - self.root_pivot,
            orientation: random_quaternion(rng),
            torsions: (0..self.n_torsions)
                .map(|_| (rng.r#gen::<f64>() - 0.5) * 2.0 * std::f64::consts::PI)
                .collect(),
        }
    }
}

/// 一様ランダムな単位四元数。
pub fn random_quaternion<R: Rng>(rng: &mut R) -> UnitQuaternion<f64> {
    // Shoemake 法。
    let u1: f64 = rng.r#gen();
    let u2: f64 = rng.r#gen();
    let u3: f64 = rng.r#gen();
    let tau = std::f64::consts::TAU;
    let q = nalgebra::Quaternion::new(
        (u1).sqrt() * (tau * u3).cos(),
        (1.0 - u1).sqrt() * (tau * u2).sin(),
        (1.0 - u1).sqrt() * (tau * u2).cos(),
        (u1).sqrt() * (tau * u3).sin(),
    );
    UnitQuaternion::from_quaternion(q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ist_io::pdbqt;

    const LIG: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  0.00  0.00    +0.000 C
ENDROOT
BRANCH   2   3
ATOM      3  C   LIG A   1       3.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      4  O   LIG A   1       3.000   1.200   0.000  0.00  0.00    -0.300 OA
ENDBRANCH   2   3
TORSDOF 1
";

    #[test]
    fn identity_reproduces_reference() {
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let conf = Conformation::identity(lig.n_torsions());
        let coords = lig.world_coords(&conf);
        for (c, a) in coords.iter().zip(&mol.atoms) {
            assert!((c - a.pos).norm() < 1e-9, "got {c:?} want {:?}", a.pos);
        }
    }

    #[test]
    fn torsion_rotates_only_subtree() {
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let mut conf = Conformation::identity(lig.n_torsions());
        conf.torsions[0] = std::f64::consts::PI; // 180度
        let coords = lig.world_coords(&conf);
        // root 原子 (idx0,1) は不動。
        assert!((coords[0] - Vec3::new(0.0, 0.0, 0.0)).norm() < 1e-9);
        assert!((coords[1] - Vec3::new(1.5, 0.0, 0.0)).norm() < 1e-9);
        // bond 軸上の atom idx2 (3,0,0) は不動。
        assert!((coords[2] - Vec3::new(3.0, 0.0, 0.0)).norm() < 1e-9);
        // 軸(x方向)まわり180度で O(idx3) (3,1.2,0) → (3,-1.2,0)。
        assert!((coords[3] - Vec3::new(3.0, -1.2, 0.0)).norm() < 1e-6, "got {:?}", coords[3]);
    }

    /// 分子内ペアの規則を固定する。`LIG` はフラグメント0={C0,C1}・フラグメント1={C2,O3} の
    /// 4 原子鎖で、異フラグメントの重原子ペアは C0-C2（グラフ距離 2）・C0-O3（距離 3）・
    /// C1-C2（距離 1）・C1-O3（距離 2）の 4 つしかない。
    /// **AutoDock Vina の規則（1-4 まで除外）では 1 つも残らない。**
    #[test]
    fn intra_pairs_follow_vina_rule() {
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        assert_eq!(lig.n_intra_pairs(), 0, "1-4 まで除外するので残らないはず");
        // 評価が有限値を返す（panic しない）。
        let e = lig.intramolecular(&lig.scored_atoms(&Conformation::identity(lig.n_torsions())), f64::INFINITY);
        assert!(e.is_finite());
    }

    /// 1-4 の係数の意味を固定する。係数 0（既定）で 1-4 の寄与は厳密に 0、
    /// 係数 1 で 1-5 以上と同じ重みになり、中間は線形に比例する。
    #[test]
    fn intra_14_scale_is_linear_and_defaults_to_zero() {
        const CHAIN4: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  0.00  0.00    +0.000 C
ENDROOT
BRANCH   2   3
ATOM      3  C   LIG A   1       2.250   1.300   0.000  0.00  0.00    +0.000 C
ATOM      4  C   LIG A   1       1.500   2.600   0.000  0.00  0.00    +0.000 C
ENDBRANCH   2   3
TORSDOF 1
";
        let mol = pdbqt::parse_str(CHAIN4).unwrap();
        let mut lig = LigandModel::new(&mol);
        // この分子には 1-5 以上のペアが無く、1-4 が 1 本だけある。
        assert_eq!(lig.n_intra_pairs(), 0);
        assert_eq!(lig.n_intra_pairs_14(), 1);
        let conf = Conformation::identity(lig.n_torsions());
        let atoms = lig.scored_atoms(&conf);
        // 既定は 0.0 なので寄与なし。
        assert_eq!(lig.intra_14_scale(), 0.0);
        assert_eq!(lig.intramolecular(&atoms, f64::INFINITY), 0.0);
        assert_eq!(lig.intramolecular_fast(&atoms, f64::INFINITY), 0.0);
        // 係数 1.0 で非零、0.5 でちょうど半分（線形）。
        lig.set_intra_14_scale(1.0);
        let e1 = lig.intramolecular(&atoms, f64::INFINITY);
        assert!(e1.abs() > 1e-6, "1-4 の寄与が入るはず: {e1}");
        lig.set_intra_14_scale(0.5);
        let eh = lig.intramolecular(&atoms, f64::INFINITY);
        assert!((eh - 0.5 * e1).abs() < 1e-12, "線形でない: {eh} vs {}", 0.5 * e1);
        // 力の経路でも同じ線形性が成り立つ（値側で確認）。
        let mut f = vec![Vec3::zeros(); atoms.len()];
        lig.set_intra_14_scale(1.0);
        let g1 = lig.intramolecular_value_grad(&atoms, &mut f, f64::INFINITY);
        assert!((g1 - e1).abs() < 1e-12);
        let mut f2 = vec![Vec3::zeros(); atoms.len()];
        lig.set_intra_14_scale(0.5);
        let gh = lig.intramolecular_value_grad(&atoms, &mut f2, f64::INFINITY);
        assert!((gh - 0.5 * g1).abs() < 1e-12);
        // 力も半分になる。
        for (a, b) in f.iter().zip(&f2) {
            assert!(((*a * 0.5) - *b).norm() < 1e-12);
        }
    }

    /// 5 原子鎖で「1-4 は別枠（既定 0）、1-5 も別枠（既定 1）」を直接確かめる。
    /// フラグメント0={C0,C1}、フラグメント1={C2,C3,C4}。異フラグメントの重原子ペアのうち
    /// グラフ距離は C1-C2=1, C0-C2=2, C1-C3=2, C0-C3=3, C1-C4=3, C0-C4=4。
    /// **1-6 以上は 0 本、1-4 は C0-C3 と C1-C4 の 2 本、1-5 は C0-C4 の 1 本。**
    /// 段 105 で 1-5 を係数で外せるようにしたので、1-5 は `intra_pairs` ではなく
    /// `intra_pairs_15` に入る（既定の係数 1.0 でエネルギーは従来と同じ）。
    #[test]
    fn intra_pairs_keep_one_five_drop_one_four() {
        const CHAIN5: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.500   0.000   0.000  0.00  0.00    +0.000 C
ENDROOT
BRANCH   2   3
ATOM      3  C   LIG A   1       3.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      4  C   LIG A   1       4.500   0.000   0.000  0.00  0.00    +0.000 C
ATOM      5  C   LIG A   1       6.000   0.000   0.000  0.00  0.00    +0.000 C
ENDBRANCH   2   3
TORSDOF 1
";
        let mol = pdbqt::parse_str(CHAIN5).unwrap();
        let mut lig = LigandModel::new(&mol);
        assert_eq!(lig.n_intra_pairs(), 0, "1-6 以上は無い");
        assert_eq!(lig.n_intra_pairs_15(), 1, "C0-C4（1-5）の 1 本");
        assert_eq!(lig.n_intra_pairs_14(), 2, "C0-C3 と C1-C4（1-4）の 2 本");
        assert_eq!(lig.intra_15_scale(), 1.0, "1-5 の既定係数は 1.0（Vina と同じ）");

        // 既定（1-5 係数 1.0）では C0-C4 がエネルギーに入り、0.0 にすると厳密に 0 になる。
        let atoms = lig.scored_atoms(&Conformation::identity(lig.n_torsions()));
        let with15 = lig.intramolecular(&atoms, f64::INFINITY);
        assert!(with15 != 0.0, "既定では 1-5 が寄与する");
        lig.set_intra_15_scale(0.0);
        assert_eq!(lig.intramolecular(&atoms, f64::INFINITY), 0.0, "0.0 で寄与が消える");
    }

    #[test]
    fn rigid_translation() {
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let mut conf = Conformation::identity(lig.n_torsions());
        conf.apply_delta(&[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        let coords = lig.world_coords(&conf);
        for (c, a) in coords.iter().zip(&mol.atoms) {
            assert!((c.x - (a.pos.x + 10.0)).abs() < 1e-9);
        }
    }

    #[test]
    fn world_coords_frames_matches_world_coords() {
        // frames 版の world 座標が world_coords と一致する（ポーズ再構成の同一性）。
        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let mut conf = Conformation::identity(lig.n_torsions());
        conf.position = Vec3::new(0.7, -0.3, 0.4);
        conf.orientation =
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(Vec3::new(0.2, 1.0, -0.4)), 0.6);
        conf.torsions[0] = 0.5;
        let a = lig.world_coords(&conf);
        let (b, _, _) = lig.world_coords_frames(&conf);
        for (x, y) in a.iter().zip(&b) {
            assert!((x - y).norm() < 1e-12);
        }
    }

    #[test]
    fn analytic_dof_gradient_matches_numerical() {
        // 連鎖則（座標力→DOF 勾配）をスカラ f64 カーネルで厳密検証する。
        // SIMD(f32) との一致は scoring_simd の単体テストで別途検証している。
        use ist_core::scoring::{intermolecular_terms, intermolecular_value_grad, ScoredAtom};

        let mol = pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        let receptor = vec![
            ScoredAtom { pos: Vec3::new(2.0, 2.0, 0.0), xs: XsType::OxygenA },
            ScoredAtom { pos: Vec3::new(-1.0, -1.0, 1.0), xs: XsType::CarbonH },
            ScoredAtom { pos: Vec3::new(4.0, 0.0, 0.5), xs: XsType::NitrogenD },
        ];

        let mut conf = Conformation::identity(lig.n_torsions());
        conf.position = Vec3::new(0.7, -0.3, 0.4);
        conf.orientation =
            UnitQuaternion::from_axis_angle(&Unit::new_normalize(Vec3::new(0.2, 1.0, -0.4)), 0.6);
        conf.torsions[0] = 0.5;

        // 値（罰則なし: 分子間＋分子内）。
        let value = |c: &Conformation| {
            let a = lig.scored_atoms(c);
            intermolecular_terms(&a, &receptor).weighted_sum() + lig.intramolecular(&a, f64::INFINITY)
        };

        let (wp, wa, wpv) = lig.world_coords_frames(&conf);
        let atoms = lig.scored_atoms_at(&wp);
        let mut force = vec![Vec3::zeros(); atoms.len()];
        let mut e = intermolecular_value_grad(&atoms, &receptor, &mut force);
        e += lig.intramolecular_value_grad(&atoms, &mut force, f64::INFINITY);
        let grad = lig.dof_gradient(&conf, &wp, &wa, &wpv, &force);

        assert!((e - value(&conf)).abs() < 1e-9, "value mismatch {e}");

        let eps = 1e-6;
        for k in 0..conf.dim() {
            let mut d = vec![0.0; conf.dim()];
            d[k] = eps;
            let mut cp = conf.clone();
            cp.apply_delta(&d);
            let fp = value(&cp);
            d[k] = -eps;
            let mut cm = conf.clone();
            cm.apply_delta(&d);
            let fm = value(&cm);
            let num = (fp - fm) / (2.0 * eps);
            assert!(
                (grad[k] - num).abs() < 1e-3,
                "dof {k}: analytic {} numerical {}",
                grad[k],
                num
            );
        }
    }

    /// A-4: 水素しか動かさない torsion は探索空間から外れる。
    #[test]
    fn null_torsions_are_excluded_from_search_space() {
        // root = {C1, C2}。回転結合 3 本:
        //   torsion 0 = C2-C3: 部分木 {C3, O4, H5}。C3 は軸上だが **O4 が軸外で採点対象**→有効。
        //   torsion 1 = C3-O4: 部分木 {O4, H5}。**O4 は軸上なので動かず**、動くのは H5 だけ→空。
        //   torsion 2 = O4-H5: 部分木 {H5}。H5 は軸上→何も動かない→空。
        // torsion 1 が「部分木に採点対象の原子がある（O4）のに空の自由度」という罠そのもの。
        const LIG: &str = "\
ROOT
ATOM      1  C   LIG A   1       0.000   0.000   0.000  0.00  0.00    +0.000 C
ATOM      2  C   LIG A   1       1.520   0.000   0.000  0.00  0.00    +0.000 C
ENDROOT
BRANCH   2   3
ATOM      3  C   LIG A   1       2.030   1.430   0.000  0.00  0.00    +0.000 C
BRANCH   3   4
ATOM      4  O   LIG A   1       3.450   1.460   0.000  0.00  0.00    -0.300 OA
BRANCH   4   5
ATOM      5  HO  LIG A   1       3.760   2.370   0.000  0.00  0.00    +0.200 HD
ENDBRANCH   4   5
ENDBRANCH   3   4
ENDBRANCH   2   3
TORSDOF 3
";
        let mol = ist_io::pdbqt::parse_str(LIG).unwrap();
        let lig = LigandModel::new(&mol);
        assert_eq!(lig.n_torsions(), 3, "回転結合 3 本");
        // torsion 2（O-H）は水素しか動かさない。
        assert!(lig.torsion_moves_scored_atom(0));
        assert!(!lig.torsion_moves_scored_atom(1), "O4 は回転軸上なので動かない");
        assert!(!lig.torsion_moves_scored_atom(2));
        assert_eq!(lig.active_torsions(), vec![0]);
        assert_eq!(lig.n_null_torsions(), 2);

        // 空の自由度を回しても採点対象の原子は 1 つも動かない（＝どの項も変わらない）。
        let base = lig.scored_positions(&Conformation::identity(3));
        for k in [1usize, 2] {
            let mut c = Conformation::identity(3);
            c.torsions[k] = 1.234;
            for (a, b) in base.iter().zip(&lig.scored_positions(&c)) {
                assert!((a - b).norm() < 1e-12, "torsion {k} が採点原子を動かした");
            }
        }
        // 有効な自由度は実際に動かす（対照）。
        let mut c = Conformation::identity(3);
        c.torsions[0] = 1.234;
        assert!(base
            .iter()
            .zip(&lig.scored_positions(&c))
            .any(|(a, b)| (a - b).norm() > 1e-6));
    }
}
