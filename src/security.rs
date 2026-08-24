use std::path::Path;

#[cfg(unix)]
use std::fs;
#[cfg(target_os = "macos")]
use std::path::PathBuf;

#[cfg(unix)]
use anyhow::Context;
use anyhow::Result;

pub fn secure_file(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        clear_extended_acl(path)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("无法设置私有文件权限 {}", path.display()))?;
    }

    #[cfg(windows)]
    windows::set_private_dacl(path, false)?;

    #[cfg(not(any(unix, windows)))]
    anyhow::bail!("当前平台不支持安全文件权限 {}", path.display());

    Ok(())
}

pub fn secure_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        clear_extended_acl(path)?;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700))
            .with_context(|| format!("无法设置私有目录权限 {}", path.display()))?;
    }

    #[cfg(windows)]
    windows::set_private_dacl(path, true)?;

    #[cfg(not(any(unix, windows)))]
    anyhow::bail!("当前平台不支持安全目录权限 {}", path.display());

    Ok(())
}

/// Validates an owner-controlled directory write boundary without changing its permissions.
/// Read-only access by other principals is outside this check.
pub fn validate_private_directory(path: &Path) -> Result<()> {
    #[cfg(unix)]
    {
        validate_private_unix_directory(path)
    }

    #[cfg(windows)]
    {
        windows::validate_private_directory(path)
    }

    #[cfg(not(any(unix, windows)))]
    {
        anyhow::bail!("当前平台不支持私有目录验证 {}", path.display())
    }
}

#[cfg(unix)]
fn validate_private_unix_directory(path: &Path) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt};

    let directory = fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_DIRECTORY | libc::O_CLOEXEC)
        .open(path)
        .with_context(|| format!("无法安全打开私有目录 {}", path.display()))?;
    let metadata = directory
        .metadata()
        .with_context(|| format!("无法读取私有目录元数据 {}", path.display()))?;
    if !metadata.is_dir() {
        anyhow::bail!("私有目录对象类型无效 {}", path.display());
    }

    // SAFETY: geteuid takes no arguments and has no memory-safety preconditions.
    let effective_uid = unsafe { libc::geteuid() };
    if metadata.uid() != effective_uid {
        anyhow::bail!("私有目录不属于当前用户 {}", path.display());
    }
    if metadata.mode() & 0o022 != 0 {
        anyhow::bail!("私有目录允许组或其他用户写入 {}", path.display());
    }
    Ok(())
}

#[cfg(target_os = "macos")]
fn clear_extended_acl(path: &Path) -> Result<()> {
    let absolute = absolute_path(path)?;
    let status = std::process::Command::new("/bin/chmod")
        .env_remove("OPENROUTER_API_KEY")
        .arg("-N")
        .arg(&absolute)
        .status()
        .with_context(|| format!("无法清除扩展 ACL {}", absolute.display()))?;
    if !status.success() {
        anyhow::bail!("无法清除扩展 ACL {}", absolute.display());
    }
    Ok(())
}

#[cfg(all(unix, not(target_os = "macos")))]
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

#[cfg(windows)]
mod windows {
    use std::ffi::c_void;
    use std::fs::{File, OpenOptions};
    use std::mem::{size_of, size_of_val};
    use std::os::windows::fs::OpenOptionsExt;
    use std::os::windows::io::AsRawHandle;
    use std::path::Path;
    use std::ptr::{null, null_mut, read_unaligned};

    use anyhow::{Context, Result, bail};

    type Bool = i32;
    type Dword = u32;
    type Handle = *mut c_void;
    type Sid = *mut c_void;

    const TOKEN_QUERY: Dword = 0x0008;
    const TOKEN_USER_CLASS: Dword = 1;
    const ERROR_INSUFFICIENT_BUFFER: i32 = 122;

    const READ_CONTROL: Dword = 0x0002_0000;
    const WRITE_DAC: Dword = 0x0004_0000;
    const FILE_SHARE_READ: Dword = 0x0000_0001;
    const FILE_SHARE_WRITE: Dword = 0x0000_0002;
    const FILE_SHARE_DELETE: Dword = 0x0000_0004;
    const FILE_FLAG_OPEN_REPARSE_POINT: Dword = 0x0020_0000;
    const FILE_FLAG_BACKUP_SEMANTICS: Dword = 0x0200_0000;
    const FILE_ATTRIBUTE_REPARSE_POINT: Dword = 0x0000_0400;
    const FILE_ATTRIBUTE_TAG_INFO_CLASS: Dword = 9;

    const SE_FILE_OBJECT: Dword = 1;
    const OWNER_SECURITY_INFORMATION: Dword = 0x0000_0001;
    const DACL_SECURITY_INFORMATION: Dword = 0x0000_0004;
    const PROTECTED_DACL_SECURITY_INFORMATION: Dword = 0x8000_0000;

    const WIN_LOCAL_SYSTEM_SID: Dword = 22;
    const WIN_BUILTIN_ADMINISTRATORS_SID: Dword = 26;
    const SECURITY_MAX_SID_SIZE: usize = 68;

    const ACL_REVISION: Dword = 2;
    const OBJECT_INHERIT_ACE: Dword = 0x01;
    const CONTAINER_INHERIT_ACE: Dword = 0x02;
    const FILE_ALL_ACCESS: Dword = 0x001f_01ff;
    const MAX_TOKEN_USER_BYTES: Dword = 64 * 1024;
    const MAX_ACL_ACE_COUNT: Dword = 8_192;

    const ACL_SIZE_INFORMATION_CLASS: Dword = 2;
    const ACCESS_ALLOWED_ACE_TYPE: u8 = 0x00;
    const ACCESS_ALLOWED_COMPOUND_ACE_TYPE: u8 = 0x04;
    const ACCESS_ALLOWED_OBJECT_ACE_TYPE: u8 = 0x05;
    const ACCESS_ALLOWED_CALLBACK_ACE_TYPE: u8 = 0x09;
    const ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE: u8 = 0x0b;
    const ACE_OBJECT_TYPE_PRESENT: Dword = 0x0000_0001;
    const ACE_INHERITED_OBJECT_TYPE_PRESENT: Dword = 0x0000_0002;
    const GUID_BYTES: usize = 16;
    const MINIMUM_SID_BYTES: usize = 8;

    const FILE_WRITE_DATA: Dword = 0x0000_0002;
    const FILE_APPEND_DATA: Dword = 0x0000_0004;
    const FILE_WRITE_EA: Dword = 0x0000_0010;
    const FILE_DELETE_CHILD: Dword = 0x0000_0040;
    const FILE_WRITE_ATTRIBUTES: Dword = 0x0000_0100;
    const DELETE_ACCESS: Dword = 0x0001_0000;
    const WRITE_DAC_ACCESS: Dword = 0x0004_0000;
    const WRITE_OWNER_ACCESS: Dword = 0x0008_0000;
    const ACCESS_SYSTEM_SECURITY: Dword = 0x0100_0000;
    const MAXIMUM_ALLOWED: Dword = 0x0200_0000;
    const GENERIC_ALL: Dword = 0x1000_0000;
    const GENERIC_WRITE: Dword = 0x4000_0000;
    const NON_PRIVATE_DIRECTORY_ACCESS: Dword = FILE_WRITE_DATA
        | FILE_APPEND_DATA
        | FILE_WRITE_EA
        | FILE_DELETE_CHILD
        | FILE_WRITE_ATTRIBUTES
        | DELETE_ACCESS
        | WRITE_DAC_ACCESS
        | WRITE_OWNER_ACCESS
        | ACCESS_SYSTEM_SECURITY
        | MAXIMUM_ALLOWED
        | GENERIC_ALL
        | GENERIC_WRITE;

    #[repr(C)]
    struct SidAndAttributes {
        sid: Sid,
        attributes: Dword,
    }

    #[repr(C)]
    struct TokenUser {
        user: SidAndAttributes,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct AclHeader {
        acl_revision: u8,
        sbz1: u8,
        acl_size: u16,
        ace_count: u16,
        sbz2: u16,
    }

    #[derive(Clone, Copy)]
    #[repr(C)]
    struct AceHeader {
        ace_type: u8,
        ace_flags: u8,
        ace_size: u16,
    }

    #[repr(C)]
    struct AccessAllowedAce {
        header: AceHeader,
        mask: Dword,
        sid_start: Dword,
    }

    #[repr(C)]
    struct AclSizeInformation {
        ace_count: Dword,
        acl_bytes_in_use: Dword,
        acl_bytes_free: Dword,
    }

    #[repr(C)]
    struct FileAttributeTagInfo {
        file_attributes: Dword,
        reparse_tag: Dword,
    }

    struct LocalSecurityDescriptor(Handle);

    impl Drop for LocalSecurityDescriptor {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: GetSecurityInfo allocated this descriptor with LocalAlloc.
                unsafe {
                    let _ = LocalFree(self.0);
                }
            }
        }
    }

    struct TokenHandle(Handle);

    impl Drop for TokenHandle {
        fn drop(&mut self) {
            if !self.0.is_null() {
                // SAFETY: OpenProcessToken returned this owned kernel handle.
                unsafe {
                    let _ = CloseHandle(self.0);
                }
            }
        }
    }

    struct TokenUserStorage {
        _storage: Vec<usize>,
        sid: Sid,
    }

    struct OwnedSid {
        _storage: Vec<usize>,
        sid: Sid,
    }

    pub(super) fn set_private_dacl(path: &Path, directory: bool) -> Result<()> {
        let file = open_security_handle(path, directory, READ_CONTROL | WRITE_DAC)?;
        reject_reparse_point(&file, path)?;
        verify_object_kind(&file, path, directory)?;

        let raw_handle = file.as_raw_handle().cast::<c_void>();
        let mut owner_sid = null_mut();
        let mut descriptor = null_mut();
        // SAFETY: raw_handle is live for the call and all requested output pointers are valid.
        let status = unsafe {
            GetSecurityInfo(
                raw_handle,
                SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION,
                &mut owner_sid,
                null_mut(),
                null_mut(),
                null_mut(),
                &mut descriptor,
            )
        };
        let _descriptor = LocalSecurityDescriptor(descriptor);
        if status != 0 {
            return win32_status_error(status, "无法读取对象所有者", path);
        }
        validate_sid(owner_sid)
            .with_context(|| format!("对象所有者 SID 无效 {}", path.display()))?;

        let current_user = current_user_sid()
            .with_context(|| format!("无法读取当前用户 SID {}", path.display()))?;
        let system = well_known_sid(WIN_LOCAL_SYSTEM_SID)
            .with_context(|| format!("无法创建 SYSTEM SID {}", path.display()))?;
        let administrators = well_known_sid(WIN_BUILTIN_ADMINISTRATORS_SID)
            .with_context(|| format!("无法创建 Administrators SID {}", path.display()))?;

        let mut principals = Vec::with_capacity(4);
        push_unique_sid(&mut principals, owner_sid);
        push_unique_sid(&mut principals, current_user.sid);
        push_unique_sid(&mut principals, system.sid);
        push_unique_sid(&mut principals, administrators.sid);

        let inheritance = if directory {
            OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE
        } else {
            0
        };
        let mut acl = build_private_acl(&principals, inheritance)
            .with_context(|| format!("无法构造私有 DACL {}", path.display()))?;

        // SAFETY: raw_handle remains live, acl is a valid initialized ACL, and null owner/group/SACL
        // pointers mean that only the protected DACL is changed.
        let status = unsafe {
            SetSecurityInfo(
                raw_handle,
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
                null_mut(),
                null_mut(),
                acl.as_mut_ptr().cast::<c_void>(),
                null_mut(),
            )
        };
        if status != 0 {
            return win32_status_error(status, "无法设置受保护的私有 DACL", path);
        }

        Ok(())
    }

    pub(super) fn validate_private_directory(path: &Path) -> Result<()> {
        let file = open_security_handle(path, true, READ_CONTROL)?;
        reject_reparse_point(&file, path)?;
        verify_object_kind(&file, path, true)?;

        let raw_handle = file.as_raw_handle().cast::<c_void>();
        let mut owner_sid = null_mut();
        let mut dacl = null_mut();
        let mut descriptor = null_mut();
        // SAFETY: raw_handle is live and every requested output pointer is writable.
        let status = unsafe {
            GetSecurityInfo(
                raw_handle,
                SE_FILE_OBJECT,
                OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
                &mut owner_sid,
                null_mut(),
                &mut dacl,
                null_mut(),
                &mut descriptor,
            )
        };
        let _descriptor = LocalSecurityDescriptor(descriptor);
        if status != 0 {
            return win32_status_error(status, "无法读取目录 DACL", path);
        }
        validate_sid(owner_sid)
            .with_context(|| format!("目录所有者 SID 无效 {}", path.display()))?;
        if dacl.is_null() {
            bail!("目录使用不受限的空 DACL {}", path.display());
        }

        let current_user = current_user_sid()
            .with_context(|| format!("无法读取当前用户 SID {}", path.display()))?;
        // SAFETY: both SIDs were validated and remain alive.
        if unsafe { EqualSid(owner_sid, current_user.sid) } == 0 {
            bail!("私有目录不属于当前用户 {}", path.display());
        }

        let system = well_known_sid(WIN_LOCAL_SYSTEM_SID)
            .with_context(|| format!("无法创建 SYSTEM SID {}", path.display()))?;
        let administrators = well_known_sid(WIN_BUILTIN_ADMINISTRATORS_SID)
            .with_context(|| format!("无法创建 Administrators SID {}", path.display()))?;
        validate_directory_allow_aces(dacl, &[current_user.sid, system.sid, administrators.sid])
            .with_context(|| format!("目录 DACL 不是私有写入边界 {}", path.display()))?;
        Ok(())
    }

    fn open_security_handle(path: &Path, directory: bool, desired_access: Dword) -> Result<File> {
        let mut options = OpenOptions::new();
        let flags = FILE_FLAG_OPEN_REPARSE_POINT
            | if directory {
                FILE_FLAG_BACKUP_SEMANTICS
            } else {
                0
            };
        options
            .access_mode(desired_access)
            .share_mode(FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE)
            .custom_flags(flags)
            .open(path)
            .with_context(|| format!("无法打开对象以设置私有权限 {}", path.display()))
    }

    fn reject_reparse_point(file: &File, path: &Path) -> Result<()> {
        let mut info = FileAttributeTagInfo {
            file_attributes: 0,
            reparse_tag: 0,
        };
        // SAFETY: the file handle is live and info points to writable storage of the stated size.
        let ok = unsafe {
            GetFileInformationByHandleEx(
                file.as_raw_handle().cast::<c_void>(),
                FILE_ATTRIBUTE_TAG_INFO_CLASS,
                (&mut info as *mut FileAttributeTagInfo).cast::<c_void>(),
                Dword::try_from(size_of_val(&info)).expect("attribute structure fits DWORD"),
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error())
                .with_context(|| format!("无法检查对象重解析属性 {}", path.display()));
        }
        if info.file_attributes & FILE_ATTRIBUTE_REPARSE_POINT != 0 {
            bail!("拒绝为重解析点设置私有权限 {}", path.display());
        }
        Ok(())
    }

    fn verify_object_kind(file: &File, path: &Path, directory: bool) -> Result<()> {
        let metadata = file
            .metadata()
            .with_context(|| format!("无法检查权限对象类型 {}", path.display()))?;
        let valid = if directory {
            metadata.is_dir()
        } else {
            metadata.is_file()
        };
        if !valid {
            bail!("权限对象类型不符合预期 {}", path.display());
        }
        Ok(())
    }

    fn validate_directory_allow_aces(dacl: *mut c_void, permitted_sids: &[Sid]) -> Result<()> {
        // SAFETY: dacl was returned inside a live GetSecurityInfo descriptor.
        if unsafe { IsValidAcl(dacl) } == 0 {
            bail!("DACL 格式无效");
        }

        // SAFETY: IsValidAcl accepted the ACL header at dacl.
        let header = unsafe { read_unaligned(dacl.cast::<AclHeader>()) };
        let acl_size = usize::from(header.acl_size);
        if acl_size < size_of::<AclHeader>() {
            bail!("DACL 长度不足");
        }
        let acl_start = dacl as usize;
        let acl_end = acl_start.checked_add(acl_size).context("DACL 边界溢出")?;

        let mut size_information = AclSizeInformation {
            ace_count: 0,
            acl_bytes_in_use: 0,
            acl_bytes_free: 0,
        };
        // SAFETY: dacl is valid and size_information has the exact declared writable size.
        let ok = unsafe {
            GetAclInformation(
                dacl,
                (&mut size_information as *mut AclSizeInformation).cast::<c_void>(),
                Dword::try_from(size_of_val(&size_information))
                    .expect("ACL size structure fits DWORD"),
                ACL_SIZE_INFORMATION_CLASS,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error()).context("GetAclInformation 失败");
        }
        if size_information.ace_count > MAX_ACL_ACE_COUNT {
            bail!("DACL ACE 数量超过安全上限");
        }
        let bytes_in_use =
            usize::try_from(size_information.acl_bytes_in_use).context("DACL 已用长度无效")?;
        let bytes_free =
            usize::try_from(size_information.acl_bytes_free).context("DACL 空闲长度无效")?;
        if bytes_in_use < size_of::<AclHeader>()
            || bytes_in_use > acl_size
            || bytes_in_use
                .checked_add(bytes_free)
                .is_none_or(|total| total > acl_size)
        {
            bail!("DACL 大小信息不一致");
        }

        for index in 0..size_information.ace_count {
            let mut ace = null_mut();
            // SAFETY: dacl is valid, index is below the reported ACE count, and ace is writable.
            let ok = unsafe { GetAce(dacl, index, &mut ace) };
            if ok == 0 {
                return Err(std::io::Error::last_os_error()).context("GetAce 失败");
            }
            let Some((mask, sid)) = parse_allow_ace(ace, acl_start, acl_end)? else {
                continue;
            };
            if mask & NON_PRIVATE_DIRECTORY_ACCESS == 0 {
                continue;
            }
            let permitted = permitted_sids.iter().any(|permitted_sid| {
                // SAFETY: both SIDs were validated and their backing storage remains alive.
                unsafe { EqualSid(*permitted_sid, sid) != 0 }
            });
            if !permitted {
                bail!("非授权主体拥有目录写入类权限");
            }
        }
        Ok(())
    }

    fn parse_allow_ace(
        ace: *mut c_void,
        acl_start: usize,
        acl_end: usize,
    ) -> Result<Option<(Dword, Sid)>> {
        if ace.is_null() {
            bail!("DACL 包含空 ACE");
        }
        let ace_start = ace as usize;
        let minimum_header_end = ace_start
            .checked_add(size_of::<AceHeader>())
            .context("ACE 头边界溢出")?;
        if ace_start < acl_start || minimum_header_end > acl_end {
            bail!("ACE 头超出 DACL 边界");
        }

        // SAFETY: the complete ACE header was bounded inside the validated ACL.
        let header = unsafe { read_unaligned(ace.cast::<AceHeader>()) };
        let ace_size = usize::from(header.ace_size);
        let ace_end = ace_start.checked_add(ace_size).context("ACE 边界溢出")?;
        if ace_size < size_of::<AceHeader>() || ace_end > acl_end {
            bail!("ACE 长度超出 DACL 边界");
        }

        let sid_offset = match header.ace_type {
            ACCESS_ALLOWED_ACE_TYPE | ACCESS_ALLOWED_CALLBACK_ACE_TYPE => {
                size_of::<AceHeader>() + size_of::<Dword>()
            }
            ACCESS_ALLOWED_OBJECT_ACE_TYPE | ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE => {
                let flags_offset = size_of::<AceHeader>() + size_of::<Dword>();
                let fixed_end = flags_offset
                    .checked_add(size_of::<Dword>())
                    .context("object ACE 固定区长度溢出")?;
                if fixed_end > ace_size {
                    bail!("object ACE 固定区长度不足");
                }
                // SAFETY: the object flags DWORD was bounded inside this ACE.
                let flags =
                    unsafe { read_unaligned(ace.cast::<u8>().add(flags_offset).cast::<Dword>()) };
                if flags & !(ACE_OBJECT_TYPE_PRESENT | ACE_INHERITED_OBJECT_TYPE_PRESENT) != 0 {
                    bail!("object ACE 使用未知标志");
                }
                let mut offset = fixed_end;
                if flags & ACE_OBJECT_TYPE_PRESENT != 0 {
                    offset = offset
                        .checked_add(GUID_BYTES)
                        .context("object ACE 长度溢出")?;
                }
                if flags & ACE_INHERITED_OBJECT_TYPE_PRESENT != 0 {
                    offset = offset
                        .checked_add(GUID_BYTES)
                        .context("inherited object ACE 长度溢出")?;
                }
                offset
            }
            ACCESS_ALLOWED_COMPOUND_ACE_TYPE => {
                bail!("不支持 compound allow ACE");
            }
            _ => return Ok(None),
        };

        let mask_offset = size_of::<AceHeader>();
        if mask_offset + size_of::<Dword>() > ace_size {
            bail!("allow ACE 缺少访问掩码");
        }
        // SAFETY: the mask DWORD was bounded inside this ACE.
        let mask = unsafe { read_unaligned(ace.cast::<u8>().add(mask_offset).cast::<Dword>()) };

        let minimum_sid_end = sid_offset
            .checked_add(MINIMUM_SID_BYTES)
            .context("ACE SID 边界溢出")?;
        if minimum_sid_end > ace_size {
            bail!("allow ACE 缺少完整 SID");
        }
        // SAFETY: the minimum SID header was bounded inside this ACE.
        let sub_authority_count = unsafe { *ace.cast::<u8>().add(sid_offset + 1) } as usize;
        let sid_size = MINIMUM_SID_BYTES
            .checked_add(
                sub_authority_count
                    .checked_mul(size_of::<Dword>())
                    .context("ACE SID 长度溢出")?,
            )
            .context("ACE SID 长度溢出")?;
        if sid_offset
            .checked_add(sid_size)
            .is_none_or(|sid_end| sid_end > ace_size)
        {
            bail!("allow ACE SID 超出 ACE 边界");
        }
        // SAFETY: the calculated SID range is fully contained by this ACE.
        let sid = unsafe { ace.cast::<u8>().add(sid_offset).cast::<c_void>() };
        validate_sid(sid)?;
        // SAFETY: sid was bounded and validated.
        let reported_sid_size = unsafe { GetLengthSid(sid) };
        if usize::try_from(reported_sid_size).context("ACE SID 长度无效")? != sid_size {
            bail!("allow ACE SID 长度不一致");
        }
        Ok(Some((mask, sid)))
    }

    fn current_user_sid() -> Result<TokenUserStorage> {
        let mut token = null_mut();
        // SAFETY: the output pointer is valid; GetCurrentProcess returns a process pseudo-handle.
        let ok = unsafe { OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &mut token) };
        if ok == 0 {
            return Err(std::io::Error::last_os_error()).context("OpenProcessToken 失败");
        }
        let token = TokenHandle(token);

        let mut needed = 0;
        // SAFETY: a zero-length probe with a null buffer is the documented size-query pattern.
        let ok =
            unsafe { GetTokenInformation(token.0, TOKEN_USER_CLASS, null_mut(), 0, &mut needed) };
        if ok != 0 {
            bail!("GetTokenInformation 大小查询意外成功");
        }
        let probe_error = std::io::Error::last_os_error();
        if probe_error.raw_os_error() != Some(ERROR_INSUFFICIENT_BUFFER) {
            return Err(probe_error).context("GetTokenInformation 大小查询失败");
        }
        if needed < Dword::try_from(size_of::<TokenUser>()).expect("TokenUser fits DWORD")
            || needed > MAX_TOKEN_USER_BYTES
        {
            bail!("GetTokenInformation 返回了无效长度");
        }

        let mut storage = aligned_storage(needed)?;
        // SAFETY: storage has at least needed writable bytes and token remains live.
        let ok = unsafe {
            GetTokenInformation(
                token.0,
                TOKEN_USER_CLASS,
                storage.as_mut_ptr().cast::<c_void>(),
                needed,
                &mut needed,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error()).context("GetTokenInformation 失败");
        }

        // SAFETY: GetTokenInformation populated a TOKEN_USER at the aligned buffer start.
        let user = unsafe { &*storage.as_ptr().cast::<TokenUser>() };
        validate_sid(user.user.sid).context("当前用户 SID 无效")?;
        Ok(TokenUserStorage {
            _storage: storage,
            sid: user.user.sid,
        })
    }

    fn well_known_sid(kind: Dword) -> Result<OwnedSid> {
        let mut storage = vec![0usize; SECURITY_MAX_SID_SIZE.div_ceil(size_of::<usize>())];
        let mut size = Dword::try_from(SECURITY_MAX_SID_SIZE).expect("SID size fits DWORD");
        // SAFETY: storage is aligned and has SECURITY_MAX_SID_SIZE writable bytes.
        let ok = unsafe {
            CreateWellKnownSid(
                kind,
                null(),
                storage.as_mut_ptr().cast::<c_void>(),
                &mut size,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error()).context("CreateWellKnownSid 失败");
        }
        let sid = storage.as_mut_ptr().cast::<c_void>();
        validate_sid(sid).context("well-known SID 无效")?;
        Ok(OwnedSid {
            _storage: storage,
            sid,
        })
    }

    fn validate_sid(sid: Sid) -> Result<()> {
        if sid.is_null() {
            bail!("SID 为空");
        }
        // SAFETY: callers pass SIDs returned by Windows security APIs or our owned SID storage.
        if unsafe { IsValidSid(sid) } == 0 {
            bail!("SID 格式无效");
        }
        Ok(())
    }

    fn push_unique_sid(sids: &mut Vec<Sid>, candidate: Sid) {
        let duplicate = sids.iter().any(|existing| {
            // SAFETY: all entries were validated and remain alive while this list is used.
            unsafe { EqualSid(*existing, candidate) != 0 }
        });
        if !duplicate {
            sids.push(candidate);
        }
    }

    fn build_private_acl(sids: &[Sid], inheritance: Dword) -> Result<Vec<usize>> {
        if sids.is_empty() {
            bail!("私有 DACL 没有主体");
        }

        let mut acl_bytes = size_of::<AclHeader>();
        let ace_prefix = size_of::<AccessAllowedAce>()
            .checked_sub(size_of::<Dword>())
            .context("ACE 布局无效")?;
        for sid in sids {
            validate_sid(*sid)?;
            // SAFETY: sid was validated and remains alive for this call.
            let sid_bytes = unsafe { GetLengthSid(*sid) };
            if sid_bytes == 0 {
                return Err(std::io::Error::last_os_error()).context("GetLengthSid 失败");
            }
            acl_bytes = acl_bytes
                .checked_add(ace_prefix)
                .and_then(|value| value.checked_add(sid_bytes as usize))
                .context("私有 DACL 长度溢出")?;
        }
        let acl_bytes = Dword::try_from(acl_bytes).context("私有 DACL 过大")?;
        let mut storage = aligned_storage(acl_bytes)?;

        // SAFETY: storage has acl_bytes writable bytes and is suitably aligned.
        let ok = unsafe {
            InitializeAcl(
                storage.as_mut_ptr().cast::<c_void>(),
                acl_bytes,
                ACL_REVISION,
            )
        };
        if ok == 0 {
            return Err(std::io::Error::last_os_error()).context("InitializeAcl 失败");
        }

        for sid in sids {
            // SAFETY: the ACL is initialized with enough space for every validated SID.
            let ok = unsafe {
                AddAccessAllowedAceEx(
                    storage.as_mut_ptr().cast::<c_void>(),
                    ACL_REVISION,
                    inheritance,
                    FILE_ALL_ACCESS,
                    *sid,
                )
            };
            if ok == 0 {
                return Err(std::io::Error::last_os_error()).context("AddAccessAllowedAceEx 失败");
            }
        }
        Ok(storage)
    }

    fn aligned_storage(bytes: Dword) -> Result<Vec<usize>> {
        let bytes = usize::try_from(bytes).context("Windows 缓冲区长度无效")?;
        let words = bytes
            .checked_add(size_of::<usize>() - 1)
            .context("Windows 缓冲区长度溢出")?
            / size_of::<usize>();
        if words == 0 {
            bail!("Windows 缓冲区长度为零");
        }
        Ok(vec![0usize; words])
    }

    fn win32_status_error<T>(status: Dword, operation: &str, path: &Path) -> Result<T> {
        Err(std::io::Error::from_raw_os_error(status as i32))
            .with_context(|| format!("{operation} {}", path.display()))
    }

    #[link(name = "advapi32")]
    unsafe extern "system" {
        fn OpenProcessToken(
            process_handle: Handle,
            desired_access: Dword,
            token_handle: *mut Handle,
        ) -> Bool;
        fn GetTokenInformation(
            token_handle: Handle,
            token_information_class: Dword,
            token_information: *mut c_void,
            token_information_length: Dword,
            return_length: *mut Dword,
        ) -> Bool;
        fn GetSecurityInfo(
            handle: Handle,
            object_type: Dword,
            security_info: Dword,
            owner: *mut Sid,
            group: *mut Sid,
            dacl: *mut *mut c_void,
            sacl: *mut *mut c_void,
            security_descriptor: *mut Handle,
        ) -> Dword;
        fn SetSecurityInfo(
            handle: Handle,
            object_type: Dword,
            security_info: Dword,
            owner: Sid,
            group: Sid,
            dacl: *mut c_void,
            sacl: *mut c_void,
        ) -> Dword;
        fn CreateWellKnownSid(
            well_known_sid_type: Dword,
            domain_sid: *const c_void,
            sid: Sid,
            sid_size: *mut Dword,
        ) -> Bool;
        fn IsValidSid(sid: Sid) -> Bool;
        fn IsValidAcl(acl: *const c_void) -> Bool;
        fn EqualSid(first: Sid, second: Sid) -> Bool;
        fn GetLengthSid(sid: Sid) -> Dword;
        fn GetAclInformation(
            acl: *const c_void,
            acl_information: *mut c_void,
            acl_information_length: Dword,
            acl_information_class: Dword,
        ) -> Bool;
        fn GetAce(acl: *const c_void, ace_index: Dword, ace: *mut *mut c_void) -> Bool;
        fn InitializeAcl(acl: *mut c_void, acl_length: Dword, acl_revision: Dword) -> Bool;
        fn AddAccessAllowedAceEx(
            acl: *mut c_void,
            ace_revision: Dword,
            ace_flags: Dword,
            access_mask: Dword,
            sid: Sid,
        ) -> Bool;
    }

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> Handle;
        fn CloseHandle(object: Handle) -> Bool;
        fn LocalFree(memory: Handle) -> Handle;
        fn GetFileInformationByHandleEx(
            file: Handle,
            file_information_class: Dword,
            file_information: *mut c_void,
            buffer_size: Dword,
        ) -> Bool;
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn windows_acl_layout_matches_documented_headers() {
            assert_eq!(size_of::<AclHeader>(), 8);
            assert_eq!(size_of::<AceHeader>(), 4);
            assert_eq!(size_of::<AccessAllowedAce>(), 12);
            assert_eq!(size_of::<AclSizeInformation>(), 12);
            assert_eq!(OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE, 3);
            assert_eq!(FILE_ALL_ACCESS, 0x001f_01ff);
        }

        #[test]
        fn private_directory_mask_rejects_every_write_class() {
            let dangerous = [
                FILE_WRITE_DATA,
                FILE_APPEND_DATA,
                FILE_WRITE_EA,
                FILE_DELETE_CHILD,
                FILE_WRITE_ATTRIBUTES,
                DELETE_ACCESS,
                WRITE_DAC_ACCESS,
                WRITE_OWNER_ACCESS,
                ACCESS_SYSTEM_SECURITY,
                MAXIMUM_ALLOWED,
                GENERIC_ALL,
                GENERIC_WRITE,
            ];
            for access in dangerous {
                assert_ne!(access & NON_PRIVATE_DIRECTORY_ACCESS, 0);
            }
            assert_eq!(0x0000_0001 & NON_PRIVATE_DIRECTORY_ACCESS, 0);
            assert_eq!(0x8000_0000 & NON_PRIVATE_DIRECTORY_ACCESS, 0);
        }
    }
}

#[cfg(all(test, unix))]
mod unix_tests {
    use std::fs;
    use std::os::unix::fs::{PermissionsExt, symlink};

    use super::*;

    #[test]
    fn private_directory_validation_uses_handle_owner_and_mode() {
        let root = tempfile::tempdir().expect("tempdir");
        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o700)).expect("chmod 700");
        validate_private_directory(root.path()).expect("private directory should pass");

        fs::set_permissions(root.path(), fs::Permissions::from_mode(0o720)).expect("chmod 720");
        assert!(validate_private_directory(root.path()).is_err());
    }

    #[test]
    fn private_directory_validation_rejects_terminal_symlink() {
        let root = tempfile::tempdir().expect("tempdir");
        let target = root.path().join("target");
        fs::create_dir(&target).expect("create target");
        fs::set_permissions(&target, fs::Permissions::from_mode(0o700)).expect("chmod target");
        let link = root.path().join("link");
        symlink(&target, &link).expect("create symlink");
        assert!(validate_private_directory(&link).is_err());
    }
}
