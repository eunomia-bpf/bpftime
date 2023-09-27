use std::{
    ffi::{CStr, CString},
    path::{Path, PathBuf},
};

use anyhow::{anyhow, bail, Context};
use clap::{Parser, Subcommand};
use frida::{
    _GError, _frida_g_error_free, frida_deinit, frida_init,
    frida_injector_inject_library_file_sync, frida_injector_new, frida_unref,
};

use crate::frida::frida_injector_close_sync;

mod frida;

#[derive(Subcommand, Debug)]
enum SubCommand {
    #[clap(about = "Start an application with bpftime-server injected")]
    Load {
        #[arg(help = "Path to the executable that will be injected with syscall-server")]
        executable_path: String,
    },
    #[clap(about = "Start an application with bpftime-agent injected")]
    Start {
        #[arg(help = "Path to the executable that will be injected with agent")]
        executable_path: String,
    },
    #[clap(about = "Inject bpftime-agent to a certain pid")]
    Attach { pid: i32 },
    #[cfg(feature = "support-load-bpf")]
    #[clap(about = "Load and attach an eBPF object into kernel")]
    LoadBpf {
        #[arg(help = "Path to the ELF file")]
        path: String,
    },
}
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[command(subcommand)]
    command: SubCommand,
    #[arg(short, long, help = "Run without commiting any modifications")]
    dry_run: bool,
    #[arg(
        short,
        long,
        help = "Installing location of bpftime",
        default_value_t = dirs::home_dir().unwrap().join(".bpftime").to_string_lossy().to_string()
    )]
    install_location: String,
}

fn inject_by_frida(pid: i32, agent: impl AsRef<Path>) -> anyhow::Result<()> {
    let agent = CString::new(agent.as_ref().to_string_lossy().as_bytes())
        .with_context(|| anyhow!("Invalid agent path"))?;
    let entry_point = CString::new("bpftime_agent_main").unwrap();
    let empty_string = CString::new("").unwrap();
    unsafe { frida_init() };
    let injector = unsafe { frida_injector_new() };
    let mut err: *mut _GError = std::ptr::null_mut();
    let id = unsafe {
        frida_injector_inject_library_file_sync(
            injector,
            pid as _,
            agent.as_ptr(),
            entry_point.as_ptr(),
            empty_string.as_ptr(),
            std::ptr::null_mut(),
            &mut err as *mut _,
        )
    };
    if !err.is_null() {
        let anyhow_err = anyhow!(
            "Failed to do injection: {}",
            unsafe { CStr::from_ptr((*err).message) }.to_str().unwrap()
        );
        unsafe { _frida_g_error_free(err) };
        unsafe { frida_unref(injector as *mut _) };
        unsafe { frida_deinit() };

        return Err(anyhow_err);
    }
    println!("Successfully injected. ID: {}", id);
    unsafe { frida_injector_close_sync(injector, std::ptr::null_mut(), std::ptr::null_mut()) };
    unsafe { frida_unref(injector as *mut _) };
    unsafe { frida_deinit() };

    Ok(())
}

#[cfg(feature = "support-load-bpf")]
fn load_ebpf_object_into_kernel(path: impl AsRef<Path>) -> anyhow::Result<()> {
    let mut obj = libbpf_rs::ObjectBuilder::default()
        .open_file(path.as_ref())
        .with_context(|| anyhow!("Failed to open ebpf object"))?
        .load()
        .with_context(|| anyhow!("Failed to load ebpf object"))?;
    for prog in obj.progs_iter_mut() {
        prog.attach()
            .with_context(|| anyhow!("Failed to attach program: {}", prog.name()))?;
    }
    Ok(())
}
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let install_path = PathBuf::from(args.install_location);
    match args.command {
        SubCommand::Load { executable_path } => {
            let so_path = install_path.join("libbpftime-syscall-server.so");
            if !so_path.exists() {
                bail!("Library not found: {:?}", so_path);
            }
            let err = exec::Command::new("bash")
                .arg("-c")
                .arg(format!(
                    "LD_PRELOAD={} {}",
                    so_path.to_string_lossy(),
                    executable_path
                ))
                .exec();
            return Err(err.into());
        }
        SubCommand::Start { executable_path } => {
            let so_path = install_path.join("libbpftime-agent.so");
            if !so_path.exists() {
                bail!("Library not found: {:?}", so_path);
            }
            let err = exec::Command::new("bash")
                .arg("-c")
                .arg(format!(
                    "LD_PRELOAD={} {}",
                    so_path.to_string_lossy(),
                    executable_path
                ))
                .exec();
            return Err(err.into());
        }
        SubCommand::Attach { pid } => {
            let so_path = install_path.join("libbpftime-agent.so");
            if !so_path.exists() {
                bail!("Library not found: {:?}", so_path);
            }
            println!("Inject: {:?}", so_path);
            inject_by_frida(pid, so_path)
        }
        #[cfg(feature = "support-load-bpf")]
        SubCommand::LoadBpf { path } => load_ebpf_object_into_kernel(PathBuf::from(path))
            .with_context(|| anyhow!("Failed to load ebpf object into kernel")),
    }
}
