use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

pub fn secure_file(path: &Path) -> Result<()> {
    clear_extended_acl(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("无法设置私有文件权限 {}", path.display()))?;
    }
    Ok(())
}

pub fn secure_directory(path: &Path) -> Result<()> {
    clear_extended_acl(path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("无法设置私有目录权限 {}", path.display()))?;
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn clear_extended_acl(path: &Path) -> Result<()> {
    let absolute = absolute_path(path)?;
    let status = std::process::Command::new("/bin/chmod")
        .arg("-N")
        .arg(&absolute)
        .status()
        .with_context(|| format!("无法清除扩展 ACL {}", absolute.display()))?;
    if !status.success() {
        bail!("无法清除扩展 ACL {}", absolute.display());
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
fn clear_extended_acl(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(target_os = "macos")]
fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir()
            .context("无法确定当前目录")?
            .join(path))
    }
}
