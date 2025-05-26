add_executable(
    bpftime-aot-cli
    main.cpp
)

set_target_properties(bpftime-aot-cli PROPERTIES OUTPUT_NAME "bpftime-aot")
llvm_map_components_to_libnames(llvm_libs ${LLVM_TARGETS_TO_BUILD} AArch64AsmParser  AArch64CodeGen  AArch64Desc  AArch64Disassembler  AArch64Info  AArch64Utils  AggressiveInstCombine  AMDGPUAsmParser  AMDGPUCodeGen  AMDGPUDesc  AMDGPUDisassembler  AMDGPUInfo  AMDGPUTargetMCA  AMDGPUUtils  Analysis  ARMAsmParser  ARMCodeGen  ARMDesc  ARMDisassembler  ARMInfo  ARMUtils  AsmParser  AsmPrinter  AVRAsmParser  AVRCodeGen  AVRDesc  AVRDisassembler  AVRInfo  BinaryFormat  BitReader  BitstreamReader  BitWriter   BPFAsmParser  BPFCodeGen  BPFDesc  BPFDisassembler  BPFInfo  CFGuard  CFIVerify  CodeGen  Core  Coroutines  Coverage  DebugInfoCodeView  Debuginfod  DebugInfoDWARF  DebugInfoGSYM  DebugInfoMSF  DebugInfoPDB  Demangle  Diff  DlltoolDriver  DWARFLinker  DWP  ExecutionEngine  Exegesis  ExegesisAArch64  ExegesisMips  ExegesisPowerPC  ExegesisX86  Extensions  FileCheck  FrontendOpenACC  FrontendOpenMP  FuzzerCLI  FuzzMutate  GlobalISel  HexagonAsmParser  HexagonCodeGen  HexagonDesc  HexagonDisassembler  HexagonInfo  InstCombine  Instrumentation  InterfaceStub  Interpreter  ipo  IRReader  JITLink  LanaiAsmParser  LanaiCodeGen  LanaiDesc  LanaiDisassembler  M68kAsmParser  M68kCodeGen  M68kDesc  M68kDisassembler  M68kInfo  MC  MCA  MCDisassembler  MCJIT  MCParser  MipsAsmParser  MipsCodeGen  MipsDesc  MipsDisassembler  MipsInfo  MIRParser  MSP430AsmParser  MSP430CodeGen  MSP430Desc  MSP430Disassembler  MSP430Info  NVPTXCodeGen  NVPTXDesc  NVPTXInfo  ObjCARCOpts  ObjCopy  Object  ObjectYAML  Option  OrcJIT  OrcShared  OrcTargetProcess  Passes  PerfJITEvents  PowerPCAsmParser  PowerPCCodeGen  PowerPCDesc  PowerPCDisassembler  PowerPCInfo  ProfileData  Remarks  RISCVAsmParser  RISCVCodeGen  RISCVDesc  RISCVDisassembler  RISCVInfo  RuntimeDyld  ScalarOpts  SelectionDAG  SparcAsmParser  SparcCodeGen  SparcDesc  SparcDisassembler  SparcInfo  Support  Symbolize  SystemZAsmParser  SystemZCodeGen  SystemZDesc  SystemZDisassembler  SystemZInfo  TableGen   Target  TextAPI  TransformUtils  VEAsmParser  VECodeGen  Vectorize  VEDesc  VEDisassembler  VEInfo  WebAssemblyAsmParser  WebAssemblyCodeGen  WebAssemblyDesc  WebAssemblyDisassembler  WebAssemblyInfo  WebAssemblyUtils  WindowsDriver  WindowsManifest  X86AsmParser  X86CodeGen  X86Desc  X86Disassembler  X86Info  X86TargetMCA  XCoreCodeGen  XCoreDesc  XCoreDisassembler  XCoreInfo  XRay)

target_include_directories(bpftime-aot-cli PRIVATE 
    ${SPDLOG_INCLUDE} 
    ${argparse_INCLUDE} 
    ${CMAKE_CURRENT_SOURCE_DIR}/../../vm/compat/include
    ../../runtime/include/
    ../../runtime/src/
    ${LIBBPF_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
)
target_link_libraries(bpftime-aot-cli PRIVATE -L/usr/lib/llvm-20/lib/ spdlog::spdlog argparse bpftime_vm_compat bpftime_llvm_vm runtime ${LIBBPF_LIBRARIES} elf z ${LLVM_LIBRARIES} ${llvm_libs})
set_property(TARGET bpftime-aot-cli PROPERTY CXX_STANDARD 20)

target_compile_definitions(bpftime-aot-cli PRIVATE _GNU_SOURCE)

add_dependencies(bpftime-aot-cli spdlog::spdlog argparse bpftime_vm_compat bpftime_llvm_vm libbpf)

install(TARGETS bpftime-aot-cli CONFIGURATIONS Release Debug RelWithDebInfo DESTINATION ~/.bpftime)
