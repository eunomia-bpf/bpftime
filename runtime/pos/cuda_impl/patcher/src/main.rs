mod parser;
mod printer;

use clap::Parser;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

#[derive(Parser, Debug)]
struct Args {
    /// Name of the person to greet
    #[arg(short='i', long)]
    ptx_in: PathBuf,

    /// Number of times to greet
    #[arg(short='o', long, default_value = "a.ptx")]
    ptx_out: PathBuf,
}

fn main() {
    let config = Args::parse();
    let mut ptx_buf = String::new();
    File::open(config.ptx_in)
        .expect("open failed")
        .read_to_string(&mut ptx_buf)
        .expect("read failed");
    let (_, ptx) = parser::parse_ptx(&ptx_buf).expect("parse failed");
    let mut writer = File::create(config.ptx_out).expect("create failed");
    printer::print_ptx(&ptx, &mut writer).expect("write failed");
}
