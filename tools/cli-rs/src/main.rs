use std::{
    ffi::{CStr, CString},
    path::{Path, PathBuf},
    process::exit,
    sync::atomic::{AtomicU32, Ordering},
};

use anyhow::{anyhow, bail, Context};
use clap::{Parser, Subcommand};
use frida::{
    _GError, _frida_g_error_free, frida_deinit, frida_init,
    frida_injector_inject_library_file_sync, frida_injector_new, frida_unref,
};
use libc::{execve, kill, pid_t};
use signal_hook::{
    consts::{SIGINT, SIGTSTP},
    iterator::Signals,
};

use crate::frida::frida_injector_close_sync;

mod frida;

#[derive(Subcommand, Debug)]
enum SubCommand {
    #[clap(about = "Start an application with bpftime-server injected")]
    Load {
        #[arg(help = "Path to the executable that will be injected with syscall-server")]
        executable_path: String,
        #[arg(help = "Other arguments to the program injected")]
        extra_args: Vec<String>,
    },
    #[clap(about = "Start an application with bpftime-agent injected")]
    Start {
        #[arg(help = "Path to the executable that will be injected with agent")]
        executable_path: String,
        #[arg(help = "Whether to enable syscall trace", short = 's', long)]
        enable_syscall_trace: bool,
        #[arg(help = "Other arguments to the program injected")]
        extra_args: Vec<String>,
    },
    #[clap(about = "Inject bpftime-agent to a certain pid")]
    Attach {
        pid: i32,
        #[arg(help = "Whether to enable syscall trace", short = 's', long)]
        enable_syscall_trace: bool,
    },
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

fn inject_by_frida(
    pid: i32,
    agent: impl AsRef<Path>,
    agent_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    let agent = CString::new(agent.as_ref().to_string_lossy().as_bytes())
        .with_context(|| anyhow!("Invalid agent path"))?;
    let entry_point = CString::new("bpftime_agent_main").unwrap();
    let agent_path = CString::new(agent_path.as_ref().to_str().unwrap()).unwrap();
    unsafe { frida_init() };
    let injector = unsafe { frida_injector_new() };
    let mut err: *mut _GError = std::ptr::null_mut();
    let id = unsafe {
        frida_injector_inject_library_file_sync(
            injector,
            pid as _,
            agent.as_ptr(),
            entry_point.as_ptr(),
            agent_path.as_ptr(),
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

#[allow(non_upper_case_globals)]
static subprocess_pid: AtomicU32 = AtomicU32::new(0);

#[allow(unused)]
fn run_command(
    path: impl AsRef<str>,
    argv: &[String],
    ld_preload: impl AsRef<str>,
    agent_so: Option<String>,
) -> anyhow::Result<()> {
    let mut cmd = std::process::Command::new(path.as_ref());

    cmd.args(argv).env("LD_PRELOAD", ld_preload.as_ref());
    if let Some(v) = agent_so {
        cmd.env("AGENT_SO", v);
    }
    let mut handle = cmd.spawn()?;
    subprocess_pid.store(handle.id(), Ordering::Relaxed);
    let status = handle
        .wait()
        .with_context(|| anyhow!("Failed to run the specific program"))?;

    if !status.success() {
        bail!("Exited with code: {:?}", status.code());
        exit(status.code().unwrap_or(0));
    }
    Ok(())
}
#[allow(unused)]
fn my_execve(
    path: impl AsRef<str>,
    argv: &[String],
    ld_preload: impl AsRef<str>,
    agent_so: Option<String>,
) -> anyhow::Error {
    let prog = CString::new(path.as_ref()).unwrap();
    let args_holder = argv
        .iter()
        .map(|v| CString::new(v.as_str()).unwrap())
        .collect::<Vec<_>>();
    let mut args_to_execve = args_holder.iter().map(|v| v.as_ptr()).collect::<Vec<_>>();
    args_to_execve.push(std::ptr::null());
    args_to_execve.insert(0, prog.as_ptr());
    let ld_preload_cstring = CString::new(format!("LD_PRELOAD={}", ld_preload.as_ref())).unwrap();
    let agent_so_cstring =
        agent_so.map(|v| CString::new(format!("AGENT_SO={}", v.as_str())).unwrap());
    let mut envp = vec![ld_preload_cstring.as_ptr()];
    if let Some(v) = agent_so_cstring.as_ref() {
        envp.push(v.as_ptr());
    }
    envp.push(std::ptr::null());
    let err = unsafe { execve(prog.as_ptr(), args_to_execve.as_ptr(), envp.as_ptr()) };
    anyhow!("Failed to run: err={}", err)
}

fn main() -> anyhow::Result<()> {
    // Handle signals
    let mut signals = Signals::new([SIGINT, SIGTSTP])?;
    std::thread::spawn(move || {
        for sig in signals.forever() {
            let pid = subprocess_pid.load(Ordering::Relaxed);
            if pid != 0 {
                unsafe { kill(pid as pid_t, sig) };
            }
        }
    });
    let args = Args::parse();
    let install_path = PathBuf::from(args.install_location);
    match args.command {
        SubCommand::Load {
            executable_path,
            extra_args,
        } => {
            let so_path = install_path.join("libbpftime-syscall-server.so");
            if !so_path.exists() {
                bail!("Library not found: {:?}", so_path);
            }
            run_command(
                executable_path,
                &extra_args,
                so_path.to_string_lossy(),
                None,
            )
        }
        SubCommand::Start {
            executable_path,
            enable_syscall_trace,
            extra_args,
        } => {
            let agent_path = install_path.join("libbpftime-agent.so");
            if !agent_path.exists() {
                bail!("Library not found: {:?}", agent_path);
            }
            if enable_syscall_trace {
                let transformr_path = install_path.join("libbpftime-agent-transformer.so");

                if !transformr_path.exists() {
                    bail!("Library not found: {:?}", transformr_path);
                }

                run_command(
                    executable_path,
                    &extra_args,
                    transformr_path.to_string_lossy(),
                    Some(agent_path.to_string_lossy().to_string()),
                )
            } else {
                run_command(
                    executable_path,
                    &extra_args,
                    agent_path.to_string_lossy(),
                    None,
                )
            }
        }
        SubCommand::Attach {
            pid,
            enable_syscall_trace,
        } => {
            let agent_path = install_path.join("libbpftime-agent.so");
            if !agent_path.exists() {
                bail!("Library not found: {:?}", agent_path);
            }
            if enable_syscall_trace {
                let transformr_path = install_path.join("libbpftime-agent-transformer.so");

                if !transformr_path.exists() {
                    bail!("Library not found: {:?}", transformr_path);
                }
                inject_by_frida(pid, transformr_path, agent_path)
            } else {
                inject_by_frida(pid, agent_path, "")
            }
        }
        #[cfg(feature = "support-load-bpf")]
        SubCommand::LoadBpf { path } => load_ebpf_object_into_kernel(PathBuf::from(path))
            .with_context(|| anyhow!("Failed to load ebpf object into kernel")),
    }
}
