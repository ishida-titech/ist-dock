//! ビルド時に git のコミットハッシュ（＋作業ツリーの dirty 状態）を取得し、
//! `ISCT_GIT_HASH` 環境変数としてバイナリに埋め込む。バージョン更新は頻繁でないため、
//! 同一バージョン表記でもこの git 情報で「どのソースからビルドしたか」を保証する。
use std::process::Command;

fn git(args: &[&str]) -> Option<String> {
    let out = Command::new("git").args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

fn main() {
    // 短縮コミットハッシュ。git 情報が無い（tarball 等）場合は "unknown"。
    let hash = git(&["rev-parse", "--short", "HEAD"]).unwrap_or_else(|| "unknown".to_string());
    // 追跡ファイルに未コミットの変更があれば dirty（untracked は無視）。
    let dirty = git(&["status", "--porcelain", "--untracked-files=no"])
        .map(|s| !s.is_empty())
        .unwrap_or(false);
    let git_desc = if hash == "unknown" {
        hash
    } else if dirty {
        format!("{hash}-dirty")
    } else {
        hash
    };
    println!("cargo:rustc-env=ISCT_GIT_HASH={git_desc}");

    // HEAD / index が変わったら再ビルドしてハッシュを更新する。
    if let Some(gitdir) = git(&["rev-parse", "--git-dir"]) {
        println!("cargo:rerun-if-changed={gitdir}/HEAD");
        println!("cargo:rerun-if-changed={gitdir}/index");
    }
}
