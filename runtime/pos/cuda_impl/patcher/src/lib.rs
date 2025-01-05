mod fatbin;
mod parser;
mod patcher;
mod printer;
mod to_cxx;

use cxx::{CxxString, CxxVector, UniquePtr};
use fatbin::FatbinHeader;
use std::ffi::{c_char, CStr, CString};

#[cxx::bridge]
mod ffi {
    extern "Rust" {
        // we can not return std::string from rust to cxx
        // see: https://github.com/dtolnay/cxx/issues/1339
        unsafe fn patch_ptx(ptx: *const c_char) -> UniquePtr<CxxString>;
        unsafe fn patch_fatbin(fatbin: *const u8) -> UniquePtr<CxxVector<u8>>;
        unsafe fn patch_raw_image(image: *const u8) -> UniquePtr<CxxVector<u8>>;
    }
}

macro_rules! ok_or_null {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(_) => return UniquePtr::null(),
        }
    };
}

unsafe fn patch_ptx(ptx: *const c_char) -> UniquePtr<CxxString> {
    let ptx = ok_or_null!(CStr::from_ptr(ptx).to_str());
    let patched = ok_or_null!(fatbin::patch_ptx(ptx));
    to_cxx::ffi::to_cxx_string(&patched)
}

unsafe fn patch_fatbin(fatbin: *const u8) -> UniquePtr<CxxVector<u8>> {
    let header_buf = std::slice::from_raw_parts(fatbin, FatbinHeader::SIZE);
    let header = ok_or_null!(fatbin::parse_fatbin_header(&mut std::io::Cursor::new(
        header_buf
    )));
    let fatbin = std::slice::from_raw_parts(fatbin, header.size as usize + FatbinHeader::SIZE);
    let patched = ok_or_null!(fatbin::patch_fatbin(fatbin));
    to_cxx::ffi::to_cxx_vec(&patched)
}

unsafe fn patch_raw_image(image: *const u8) -> UniquePtr<CxxVector<u8>> {
    let header_buf = std::slice::from_raw_parts(image, FatbinHeader::SIZE);
    match fatbin::parse_fatbin_header(&mut std::io::Cursor::new(header_buf)) {
        Ok(header) => {
            let fatbin =
                std::slice::from_raw_parts(image, header.size as usize + FatbinHeader::SIZE);
            let patched = ok_or_null!(fatbin::patch_fatbin(fatbin));
            to_cxx::ffi::to_cxx_vec(&patched)
        }
        Err(_) => {
            let ptx = ok_or_null!(CStr::from_ptr(image as *const i8).to_str());
            let patched = ok_or_null!(fatbin::patch_ptx(ptx));
            // zero terminated cstr
            to_cxx::ffi::to_cxx_vec(ok_or_null!(CString::new(patched)).as_bytes_with_nul())
        }
    }
}
