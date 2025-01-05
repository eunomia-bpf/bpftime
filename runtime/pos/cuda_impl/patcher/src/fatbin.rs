#![allow(unused)]

use anyhow::anyhow;
use binrw::BinRead;
use regex::Regex;
use std::io::{Seek, Write};
use std::process::Command;
use std::{fs::File, io::Read};
use tempfile::{NamedTempFile, TempDir};

use crate::parser;
use crate::patcher;
use crate::printer;

pub fn patch_ptx(ptx: &str) -> anyhow::Result<String> {
    let (_, mut ptx) = parser::parse_ptx(ptx).map_err(|_| anyhow!("parse failed"))?;
    patcher::patch_ptx(&mut ptx)?;
    let mut ptx_out = Vec::new();
    printer::print_ptx(&ptx, &mut ptx_out)?;
    Ok(String::from_utf8(ptx_out)?)
}

pub fn patch_fatbin(fatbin: &[u8]) -> anyhow::Result<Vec<u8>> {
    // dump ptx from fatbin
    let mut old = NamedTempFile::new()?;
    old.write_all(fatbin)?;
    let dump_dir = TempDir::new()?;
    let result = Command::new("cuobjdump")
        .args(["-all", "-xptx", "all", &old.path().display().to_string()])
        .current_dir(&dump_dir)
        .output()?;
    assert!(result.status.success(), "cuobjdump failed, {:#?}", result);
    let output = String::from_utf8(result.stdout)?;

    // patch dumped ptx
    let mut patched_ptxs = Vec::new();
    let patch_dir = TempDir::new()?;
    let re = Regex::new(r"[a-zA-Z0-9\._-]+\.ptx")?;
    for ptx_name in re.find_iter(&output).map(|m| m.as_str()) {
        let mut ptx_buf = String::new();
        File::open(dump_dir.path().join(ptx_name))?.read_to_string(&mut ptx_buf)?;
        let (_, mut ptx) = parser::parse_ptx(&ptx_buf).map_err(|_| anyhow!("parse failed"))?;

        patcher::patch_ptx(&mut ptx)?;

        let ptx_out_path = patch_dir.path().join(ptx_name);
        let mut ptx_out = File::create(&ptx_out_path)?;
        printer::print_ptx(&ptx, &mut ptx_out)?;

        let sm_arch = ptx_name
            .split('.')
            .skip(2)
            .next()
            .expect("unexpected ptx name");
        let arch = Regex::new(r"sm_(\d+)")?
            .captures(sm_arch)
            .map(|m| m[1].to_string())
            .expect("unexpected ptx name");
        patched_ptxs.push((ptx_out_path, arch));
    }

    // pack patched ptx to fatbin
    let mut patched_fatbin = NamedTempFile::new()?;
    // FIXME: parse sm from fatbin/ptx
    let result = Command::new("fatbinary")
        .args(patched_ptxs.iter().map(|ptx| {
            let (path, arch) = ptx;
            format!("--image=file={},profile=compute_{}", path.display(), arch)
        }))
        .arg(format!("--create={}", patched_fatbin.path().display()))
        .output()?;
    assert!(result.status.success(), "fatbinary failed, {:#?}", result);
    let mut ret = Vec::<u8>::new();
    patched_fatbin.read_to_end(&mut ret)?;
    Ok(ret)
}

#[test]
fn test_patch_fatbin() {
    let mut buf = Vec::new();
    File::open("ptx/torch.fatbin")
        .unwrap()
        .read_to_end(&mut buf)
        .unwrap();
    let _ = patch_fatbin(&buf).unwrap();
}

#[derive(BinRead)]
#[br(magic = 0xba55ed50u32)]
pub struct FatbinHeader {
    pub version: u16,
    pub header_size: u16,
    pub size: u64,
}

impl FatbinHeader {
    pub const SIZE: usize = 16;
}

pub fn parse_fatbin_header<T: Read + Seek>(fatbin: &mut T) -> anyhow::Result<FatbinHeader> {
    let header = FatbinHeader::read_ne(fatbin)?;
    Ok(header)
}
