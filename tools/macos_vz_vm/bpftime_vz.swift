// bpftime_vz.swift
//
// Lightweight ARM64 Linux VM manager for the bpftime macOS eBPF bridge.
// Boots an Ubuntu 24.04 (noble) arm64 cloud image under the macOS
// Virtualization.framework:
//   - NAT virtio networking; the guest IP and host gateway are recorded in
//     vm-info.json (the guest reaches host TCP ports via the gateway)
//   - virtiofs share of the host /Users tagged "host_users", mounted at the
//     identical guest path so Mach-O workload paths map 1:1
//   - NoCloud seed.iso carrying an embedded SSH public key (no password)
//   - EFI boot straight from the cloud image; efi-nvram.bin is recreated on
//     every run so an ungraceful kill never leaves corrupted UEFI variables
//   - vCPU count pinned to the host logical CPU count so per-CPU BPF map
//     slot counts match the Darwin host 1:1
//
// Build (Apple Silicon host only):
//   swiftc -O bpftime_vz.swift -o bpftime-vz -framework Virtualization
//   codesign --sign - --entitlements entitlements.plist --force bpftime-vz
//
// Subcommands: setup | run | ip
//
// SPDX-License-Identifier: MIT

import Foundation
import Virtualization

#if arch(arm64)

let IMAGE_URL =
    "https://cloud-images.ubuntu.com/noble/current/noble-server-cloudimg-arm64.img"
let CHECKSUM_URL =
    "https://cloud-images.ubuntu.com/noble/current/SHA256SUMS"
let DEFAULT_VM_DIR =
    ProcessInfo.processInfo.environment["BPFTIME_VM_DIR"]
        ?? (NSHomeDirectory() + "/.bpftime/macos-vm")
let DISK_SIZE_BYTES: UInt64 = 16 * 1024 * 1024 * 1024
let SSH_USER = "ubuntu"
let FS_TAG = "host_users"
let HOST_USERS_PATH = "/Users"

struct CLI {
    var subcommand = "run"
    var vmDir = DEFAULT_VM_DIR
    var memoryGb: UInt64 = 8
    var cpus = ProcessInfo.processInfo.activeProcessorCount

    static func parse(_ args: [String]) -> CLI {
        var cli = CLI()
        var positional = [String]()
        var i = 0
        while i < args.count {
            switch args[i] {
            case "--vm-dir":
                i += 1; cli.vmDir = args[i]
            case "--memory-gb":
                i += 1; cli.memoryGb = UInt64(args[i]) ?? cli.memoryGb
            case "--cpus":
                i += 1; cli.cpus = Int(args[i]) ?? cli.cpus
            default:
                positional.append(args[i])
            }
            i += 1
        }
        cli.subcommand = positional.first ?? "run"
        return cli
    }
}

struct Paths {
    let dir: URL
    var qcowImage: URL { dir.appendingPathComponent("noble-server-cloudimg-arm64.img") }
    var rawDisk: URL { dir.appendingPathComponent("disk.raw") }
    var partialImage: URL { dir.appendingPathComponent("noble-server-cloudimg-arm64.img.partial") }
    var checksums: URL { dir.appendingPathComponent("SHA256SUMS") }
    var seedSrc: URL { dir.appendingPathComponent("seed") }
    var seedIso: URL { dir.appendingPathComponent("seed.iso") }
    var sshKey: URL { dir.appendingPathComponent("vm_ssh_key") }
    var sshPubKey: URL { dir.appendingPathComponent("vm_ssh_key.pub") }
    var machineId: URL { dir.appendingPathComponent("machine-id.bin") }
    var efiStore: URL { dir.appendingPathComponent("efi-nvram.bin") }
    var serialLog: URL { dir.appendingPathComponent("vm-serial.log") }
    var infoJson: URL { dir.appendingPathComponent("vm-info.json") }
}

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data("bpftime-vz: \(message)\n".utf8))
    exit(1)
}

func log(_ message: String) {
    FileHandle.standardOutput.write(Data("bpftime-vz: \(message)\n".utf8))
}

@discardableResult
func runProcess(_ launchPath: String, _ arguments: [String]) -> Process {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: launchPath)
    p.arguments = arguments
    do {
        try p.run()
    } catch {
        fail("failed to launch \(launchPath): \(error)")
    }
    p.waitUntilExit()
    if p.terminationStatus != 0 {
        fail("\(launchPath) \(arguments.joined(separator: " ")) exited \(p.terminationStatus)")
    }
    return p
}

func captureProcess(_ launchPath: String, _ arguments: [String]) -> String {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: launchPath)
    p.arguments = arguments
    let pipe = Pipe()
    p.standardOutput = pipe
    p.standardError = Pipe()
    do { try p.run() } catch { return "" }
    let data = pipe.fileHandleForReading.readDataToEndOfFile()
    p.waitUntilExit()
    return String(data: data, encoding: .utf8) ?? ""
}

// MARK: - Download and image conversion

func downloadFile(from urlString: String, to destination: URL) {
    guard let url = URL(string: urlString) else { fail("bad url \(urlString)") }
    let semaphore = DispatchSemaphore(value: 0)
    URLSession.shared.downloadTask(with: url) { tempUrl, _, error in
        if let tempUrl {
            try? FileManager.default.removeItem(at: destination)
            do {
                try FileManager.default.moveItem(at: tempUrl, to: destination)
            } catch {
                fail("move downloaded image failed: \(error)")
            }
        } else if let error {
            fail("download failed: \(error)")
        }
        semaphore.signal()
    }.resume()
    log("downloading \(url.lastPathComponent) ...")
    semaphore.wait()
}

func sha256(of url: URL) -> String {
    captureProcess("/usr/bin/shasum", ["-a", "256", url.path])
        .split(separator: " ").first.map(String.init) ?? ""
}

func converterPath() -> URL {
    // The converter ships next to the binary in the same source directory.
    URL(fileURLWithPath: CommandLine.arguments[0])
        .deletingLastPathComponent()
        .appendingPathComponent("qcow2_to_raw.py")
}

func ensureImage(_ paths: Paths) {
    let fm = FileManager.default
    if !fm.fileExists(atPath: paths.qcowImage.path) {
        downloadFile(from: IMAGE_URL, to: paths.partialImage)
        downloadFile(from: CHECKSUM_URL, to: paths.checksums)
        let digest = sha256(of: paths.partialImage)
        let sums = (try? String(contentsOf: paths.checksums, encoding: .utf8)) ?? ""
        guard let line = sums.split(separator: "\n")
            .first(where: { $0.contains("noble-server-cloudimg-arm64.img") })
        else { fail("checksum entry missing in SHA256SUMS") }
        let expected = line.split(separator: " ").first.map(String.init) ?? ""
        guard expected == digest else {
            fail("checksum mismatch: got \(digest), expected \(expected)")
        }
        try? fm.moveItem(at: paths.partialImage, to: paths.qcowImage)
        log("checksum verified (\(digest.prefix(16))...)")
    }
    // Virtualization.framework boots only raw images; Ubuntu ships QCOW2 and
    // stock macOS has no qemu-img, so the bundled converter runs once. The
    // 16 GiB raw disk is sparse; cloud-init grows the root partition on boot.
    if !fm.fileExists(atPath: paths.rawDisk.path) {
        log("converting QCOW2 -> raw (one time)")
        runProcess("/usr/bin/python3", [
            converterPath().path, paths.qcowImage.path,
            paths.rawDisk.path, String(DISK_SIZE_BYTES),
        ])
    }
}

// MARK: - SSH key and NoCloud seed

func ensureSshKey(_ paths: Paths) -> String {
    let fm = FileManager.default
    if !fm.fileExists(atPath: paths.sshKey.path) {
        runProcess("/usr/bin/ssh-keygen", [
            "-t", "ed25519", "-N", "", "-f", paths.sshKey.path,
            "-C", "bpftime-vz",
        ])
    }
    let pub = (try? String(contentsOf: paths.sshPubKey, encoding: .utf8))?
        .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
    if pub.isEmpty { fail("empty ssh public key") }
    return pub
}

func writeSeed(_ paths: Paths, pubKey: String) {
    let fm = FileManager.default
    try? fm.createDirectory(at: paths.seedSrc, withIntermediateDirectories: true)
    let metaData = """
    instance-id: bpftime-vz-noble
    local-hostname: bpftime-vz
    """
    // cloud-init mounts the host_users virtiofs share at the same /Users path.
    let userData = """
    #cloud-config
    hostname: bpftime-vz
    preserve_hostname: false
    ssh_pwauth: false
    chpasswd:
      expire: false
    users:
      - name: \(SSH_USER)
        sudo: ALL=(ALL) NOPASSWD:ALL
        groups: [adm, dialout, cdrom, floppy, sudo, audio, dip, video, plugdev, netdev]
        shell: /bin/bash
        ssh_authorized_keys:
          - \(pubKey)
    bootcmd:
      - mkdir -p \(HOST_USERS_PATH)
    write_files:
      - path: /etc/systemd/system/Users.mount
        permissions: '0644'
        content: |
          [Unit]
          Description=macOS host /Users via virtiofs (tag \(FS_TAG))
          DefaultDependencies=no
          After=systemd-modules-load.service
          Before=local-fs.target
          ConditionPathExists=\(HOST_USERS_PATH)
          [Mount]
          What=\(FS_TAG)
          Where=\(HOST_USERS_PATH)
          Type=virtiofs
          Options=defaults,nofail
          [Install]
          WantedBy=local-fs.target
      - path: /etc/modules-load.d/virtiofs.conf
        permissions: '0644'
        content: |
          virtiofs
    runcmd:
      - systemctl daemon-reload
      - systemctl enable --now Users.mount
      - sh -c 'echo bpftime-vz cloud-init done > /run/bpftime-vz.ready'
    """
    try! metaData.write(to: paths.seedSrc.appendingPathComponent("meta-data"),
                        atomically: true, encoding: .utf8)
    try! userData.write(to: paths.seedSrc.appendingPathComponent("user-data"),
                        atomically: true, encoding: .utf8)
    // NoCloud requires the ISO volume label CIDATA.
    try? fm.removeItem(at: paths.seedIso)
    runProcess("/usr/bin/hdiutil", [
        "makehybrid", "-iso", "-joliet",
        "-default-volume-name", "CIDATA",
        "-o", paths.seedIso.path, paths.seedSrc.path,
    ])
    log("NoCloud seed written: \(paths.seedIso.path)")
}

// MARK: - VM configuration

func stableMacAddress() -> VZMACAddress {
    // Deterministic locally-administered unicast MAC for ARP lease matching.
    VZMACAddress(string: "a6:1b:74:69:6d:65")!
}

func loadOrCreateMachineIdentifier(_ paths: Paths) -> VZGenericMachineIdentifier {
    if let data = try? Data(contentsOf: paths.machineId),
       let ident = VZGenericMachineIdentifier(dataRepresentation: data) {
        return ident
    }
    let ident = VZGenericMachineIdentifier()
    try! ident.dataRepresentation.write(to: paths.machineId)
    return ident
}

func recreateEfiStore(_ paths: Paths) -> VZEFIVariableStore {
    // Always start from a clean variable store: the Ubuntu image carries the
    // removable fallback loader (\EFI\BOOT\BOOTAA64.EFI), so empty NVRAM
    // boots reliably and stale UEFI variables can never wedge the guest.
    try? FileManager.default.removeItem(at: paths.efiStore)
    do {
        return try VZEFIVariableStore(creatingVariableStoreAt: paths.efiStore)
    } catch {
        fail("cannot create EFI variable store: \(error)")
    }
}

func buildConfiguration(_ paths: Paths, cli: CLI) throws
    -> VZVirtualMachineConfiguration
{
    let config = VZVirtualMachineConfiguration()

    let platform = VZGenericPlatformConfiguration()
    platform.machineIdentifier = loadOrCreateMachineIdentifier(paths)
    config.platform = platform

    let bootLoader = VZEFIBootLoader()
    bootLoader.variableStore = recreateEfiStore(paths)
    config.bootLoader = bootLoader

    config.cpuCount = cli.cpus
    let minMem = VZVirtualMachineConfiguration.minimumAllowedMemorySize
    let maxMem = VZVirtualMachineConfiguration.maximumAllowedMemorySize
    config.memorySize = max(minMem, min(maxMem, cli.memoryGb * 1024 * 1024 * 1024))
    log("vCPUs=\(config.cpuCount) memory=\(config.memorySize / (1024 * 1024))MiB")

    config.entropyDevices = [VZVirtioEntropyDeviceConfiguration()]

    let diskAttachment = try VZDiskImageStorageDeviceAttachment(
        url: paths.rawDisk, readOnly: false)
    let seedAttachment = try VZDiskImageStorageDeviceAttachment(
        url: paths.seedIso, readOnly: true)
    config.storageDevices = [
        VZVirtioBlockDeviceConfiguration(attachment: diskAttachment),
        VZVirtioBlockDeviceConfiguration(attachment: seedAttachment),
    ]

    let network = VZVirtioNetworkDeviceConfiguration()
    network.attachment = VZNATNetworkDeviceAttachment()
    network.macAddress = stableMacAddress()
    config.networkDevices = [network]

    let sharedDir = VZSharedDirectory(
        url: URL(fileURLWithPath: HOST_USERS_PATH), readOnly: false)
    let fsDevice = VZVirtioFileSystemDeviceConfiguration(tag: FS_TAG)
    fsDevice.share = VZSingleDirectoryShare(directory: sharedDir)
    config.directorySharingDevices = [fsDevice]

    let serial = VZVirtioConsoleDeviceSerialPortConfiguration()
    FileManager.default.createFile(atPath: paths.serialLog.path, contents: nil)
    serial.attachment = VZFileHandleSerialPortAttachment(
        fileHandleForReading: FileHandle(forReadingAtPath: "/dev/null")!,
        fileHandleForWriting: FileHandle(forWritingAtPath: paths.serialLog.path)!)
    config.serialPorts = [serial]

    try config.validate()
    return config
}

// MARK: - Guest IP discovery

func arpTableEntries() -> [(ip: String, mac: String)] {
    let out = captureProcess("/usr/sbin/arp", ["-a", "-n"])
    var entries: [(String, String)] = []
    // Example: ? (192.168.64.2) at a6:1b:74:69:6d:65 on bridge100 ...
    for line in out.split(separator: "\n") {
        let text = String(line)
        guard let ipRange = text.range(of: #"\(([0-9.]+)\)"#,
                                       options: .regularExpression),
              let macRange = text.range(
                of: #"([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}"#,
                options: .regularExpression) else { continue }
        let ip = String(text[ipRange]).trimmingCharacters(
            in: CharacterSet(charactersIn: "()"))
        entries.append((ip, String(text[macRange]).lowercased()))
    }
    return entries
}

func tcpPortOpen(_ ip: String, port: Int, timeoutSeconds: TimeInterval) -> Bool {
    let p = Process()
    p.executableURL = URL(fileURLWithPath: "/usr/bin/nc")
    p.arguments = ["-z", "-G", String(Int(timeoutSeconds)), ip, String(port)]
    do { try p.run() } catch { return false }
    p.waitUntilExit()
    return p.terminationStatus == 0
}

func nudgeSubnet() {
    // Best-effort sweep so the guest DHCP lease appears in the host ARP table.
    for suffix in 1...254 {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/sbin/ping")
        p.arguments = ["-c", "1", "-t", "1", "192.168.64.\(suffix)"]
        p.standardOutput = Pipe()
        p.standardError = Pipe()
        try? p.run()
        p.waitUntilExit()
    }
}

func discoverGuestIP(mac: String, deadline: Date) -> String? {
    DispatchQueue.global(qos: .utility).async { nudgeSubnet() }
    while Date() < deadline {
        for entry in arpTableEntries()
        where entry.mac == mac.lowercased() && tcpPortOpen(entry.ip, port: 22, timeoutSeconds: 1) {
            return entry.ip
        }
        Thread.sleep(forTimeInterval: 2)
    }
    return nil
}

func gatewayFor(ip: String) -> String {
    // vmnet NAT keeps the gateway on the guest's own /24 at .1.
    ip.split(separator: ".").dropLast().joined(separator: ".") + ".1"
}

func writeInfo(_ paths: Paths, ip: String, mac: String, gateway: String) {
    let info: [String: Any] = [
        "vm_ip": ip,
        "gateway_ip": gateway,
        "mac": mac,
        "ssh_user": SSH_USER,
        "ssh_key": paths.sshKey.path,
        "virtiofs_tag": FS_TAG,
        "host_users_path": HOST_USERS_PATH,
        "network": "nat",
    ]
    let data = try! JSONSerialization.data(
        withJSONObject: info, options: [.prettyPrinted, .sortedKeys])
    try! data.write(to: paths.infoJson)
    log("guest reachable at \(ip) (host gateway \(gateway)); info -> \(paths.infoJson.path)")
}

// MARK: - Lifecycle

func cmdSetup(_ paths: Paths) {
    let pubKey = ensureSshKey(paths)
    ensureImage(paths)
    writeSeed(paths, pubKey: pubKey)
    log("setup complete")
}

func cmdRun(_ paths: Paths, cli: CLI) {
    cmdSetup(paths)
    let mac = stableMacAddress().string
    // buildConfiguration's validate() is Apple's host-support and
    // com.apple.security.virtualization entitlement gate; report it cleanly
    // instead of trapping on an unsupported or unsigned host.
    let config: VZVirtualMachineConfiguration
    do {
        config = try buildConfiguration(paths, cli: cli)
    } catch {
        fail("virtual machine configuration unsupported: \(error)")
    }
    let machine = VZVirtualMachine(configuration: config)

    // The main thread must stay inside its RunLoop: the @MainActor start task
    // and VZ state callbacks both need it; blocking here deadlocks start().
    signal(SIGTERM, SIG_IGN)
    signal(SIGINT, SIG_IGN)

    var started = false
    let stateLock = NSLock()
    let stateObservation = machine.observe(\.state) { vm, _ in
        stateLock.lock()
        let wasStarted = started
        if vm.state == .running { started = true }
        stateLock.unlock()
        if vm.state == .stopped && wasStarted {
            log("guest stopped")
            CFRunLoopStop(CFRunLoopGetMain())
        }
    }

    Task { @MainActor in
        do {
            try await machine.start()
            log("virtual machine started")
        } catch {
            fail("machine start failed: \(error)")
        }
    }

    // Blocking pings/SSH probes stay off the main actor so the RunLoop runs.
    DispatchQueue.global(qos: .utility).async {
        if let ip = discoverGuestIP(
            mac: mac, deadline: Date().addingTimeInterval(240)) {
            writeInfo(paths, ip: ip, mac: mac, gateway: gatewayFor(ip: ip))
        } else {
            log("WARNING: guest IP discovery timed out; rerun `bpftime-vz ip`")
        }
    }

    let makeSignalSource = { (sig: Int32, label: String) in
        let src = DispatchSource.makeSignalSource(
            signal: sig, queue: DispatchQueue.global())
        src.setEventHandler {
            log("\(label) received, requesting guest stop")
            Task { @MainActor in try? await machine.requestStop() }
        }
        src.resume()
        return src
    }
    let sigTerm = makeSignalSource(SIGTERM, "SIGTERM")
    let sigInt = makeSignalSource(SIGINT, "SIGINT")
    withExtendedLifetime((sigTerm, sigInt, stateObservation)) {
        RunLoop.main.run()
    }
}

func cmdIp(_ paths: Paths) {
    let mac = stableMacAddress().string
    guard let ip = discoverGuestIP(
        mac: mac, deadline: Date().addingTimeInterval(60))
    else { fail("guest IP not found for MAC \(mac)") }
    writeInfo(paths, ip: ip, mac: mac, gateway: gatewayFor(ip: ip))
    print(ip)
}

let cli = CLI.parse(Array(CommandLine.arguments.dropFirst()))
let usage = "usage: bpftime-vz [setup|run|ip] [--vm-dir DIR] [--memory-gb N] [--cpus N]"
if ["help", "--help", "-h"].contains(cli.subcommand) {
    print(usage)
    exit(0)
}
guard ["setup", "run", "ip"].contains(cli.subcommand) else { fail(usage) }
let paths = Paths(dir: URL(fileURLWithPath: cli.vmDir, isDirectory: true))
try? FileManager.default.createDirectory(
    atPath: paths.dir.path, withIntermediateDirectories: true)

switch cli.subcommand {
case "setup":
    cmdSetup(paths)
case "run":
    cmdRun(paths, cli: cli)
case "ip":
    cmdIp(paths)
default:
    fail(usage)
}

#else
fail("bpftime-vz requires an arm64 (Apple Silicon) host")
#endif
