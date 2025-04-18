#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("to_cxx.h");

        fn to_cxx_string(s: &str) -> UniquePtr<CxxString>;
        fn to_cxx_vec(v: &[u8]) -> UniquePtr<CxxVector<u8>>;
    }
}
