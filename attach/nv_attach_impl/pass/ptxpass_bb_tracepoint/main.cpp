#include "json.hpp"
#include "ptxpass/core.hpp"
#include <cstdio>
#include <cstring>
#include <vector>
#include <exception>
#include <iostream>
#include <string>
#include <regex>
#include <set>
#include <sstream>

namespace bb_kprobe_params
{
struct BBKprobeParams {
	bool emit_nops_for_alignment = false;
	int pad_nops = 0;
	bool auto_inject_labels = true;  // Automatically inject labels for unlabeled BBs
};
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(BBKprobeParams,
						emit_nops_for_alignment,
						pad_nops,
						auto_inject_labels);
} // namespace bb_kprobe_params

static ptxpass::pass_config::PassConfig get_default_config()
{
	ptxpass::pass_config::PassConfig cfg;
	cfg.name = "bb_kprobe";
	cfg.description =
		"Instrument PTX at basic block level kprobes matching kprobe/<kernel>__BB<n> "
		"with optional register capture via kprobe/<kernel>__BB<n>__reg1__reg2__... "
		"Uses sequential BB numbering (BB0, BB1, BB2...) with automatic identification "
		"of all basic blocks including those not explicitly labeled by nvcc compiler. "
		"Register names should be PTX register names without the % prefix (e.g., rd1, r1, f1). "
		"Captured registers can be read via bpf_get_ptx_reg(idx, ctx) helper.";
	// Match kprobe/<kernel_name>__BB<n> where:
	// - <kernel_name> is the mangled kernel function name (e.g., _Z9vectorAddPKfS0_Pf)
	// - __BB<n> is the sequential basic block number (0, 1, 2...)
    // - Optional: __reg1__reg2__... for register capture
	// Pattern allows alphanumerics, underscores, and other C++ mangling characters
	cfg.attach_points.includes = { 
		"^kprobe/[a-zA-Z0-9_$@.]*__BB[0-9]+(__[a-zA-Z0-9_]+)*$"  // Sequential BB numbering
	};
	cfg.attach_points.excludes = {};
	
	// Default parameters
	bb_kprobe_params::BBKprobeParams default_params;
	nlohmann::json params_json;
	to_json(params_json, default_params);
	cfg.parameters = params_json;
	
	cfg.attach_type = 8; // kprobe type (same as kprobe_entry)
	return cfg;
}

// Structure to hold basic block information
struct BBInfo {
	std::string ptx_label;  // Explicit label if exists, e.g., "$L__BB0_1", or empty
	std::string line_content; // The actual first line of this BB
	size_t position;        // Position in the kernel section (char offset)
	size_t line_number;     // Line number in the kernel
	bool has_explicit_label; // Whether this BB has an explicit PTX label
};

// Helper: Check if a line is blank or a comment
static bool is_blank_or_comment(const std::string &line)
{
	// Trim leading whitespace
	size_t start = line.find_first_not_of(" \t");
	if (start == std::string::npos) {
		return true; // All whitespace
	}
	
	// Check if it's a comment
	if (line[start] == '/' && start + 1 < line.length() && line[start + 1] == '/') {
		return true;
	}
	
	return false;
}

// Helper: Check if a line is a non-instruction (directives, declarations, labels)
static bool is_non_instruction(const std::string &line)
{
	if (is_blank_or_comment(line)) {
		return true;
	}
	
	size_t start = line.find_first_not_of(" \t");
	if (start == std::string::npos) {
		return true;
	}
	
	std::string trimmed = line.substr(start);
	
	// Skip ALL directives that start with '.' (including .visible, .entry, etc.)
	if (trimmed[0] == '.') {
		return true;
	}
	
	// Skip opening and closing braces
	if (trimmed[0] == '{' || trimmed[0] == '}') {
		return true;
	}
	
	// Skip lines that end with opening parenthesis (function declarations)
	if (trimmed.back() == '(' || trimmed.back() == ')') {
		return true;
	}
	
	// Skip only-label lines (labels are on lines by themselves)
	// But keep them if they're PTX BB labels like $L__BB
	if (trimmed.find(':') != std::string::npos) {
		size_t colon_pos = trimmed.find(':');
		// Check if it's a PTX BB label
		if (trimmed.find("$L__BB") == 0) {
			// This is a BB label, check if there's an instruction after
			std::string after_colon = trimmed.substr(colon_pos + 1);
			size_t after_start = after_colon.find_first_not_of(" \t\r\n");
			if (after_start == std::string::npos) {
				return true; // Only a label, no instruction
			}
		} else {
			// Other labels (not BB labels) - skip if no instruction after
			std::string after_colon = trimmed.substr(colon_pos + 1);
			size_t after_start = after_colon.find_first_not_of(" \t\r\n");
			if (after_start == std::string::npos) {
				return true;
			}
		}
	}
	
	return false;
}

// Helper: Check if a line contains a branch/control flow instruction
// These instructions end a basic block
static bool is_control_flow_instruction(const std::string &line)
{
	// Trim leading whitespace
	size_t start = line.find_first_not_of(" \t");
	if (start == std::string::npos) {
		return false;
	}
	
	std::string trimmed = line.substr(start);
	
	// Check for branch instructions: bra, ret, exit, bar.sync (barrier can affect control flow)
	// Also check for conditional branches: @pred bra
	if (trimmed.find("bra ") == 0 || trimmed.find("bra\t") == 0 ||
	    trimmed.find("ret;") == 0 || trimmed.find("ret ") == 0 ||
	    trimmed.find("exit;") == 0 || trimmed.find("exit ") == 0) {
		return true;
	}
	
	// Check for predicated branches: @%p1 bra
	if (trimmed[0] == '@') {
		size_t space_pos = trimmed.find(' ');
		if (space_pos != std::string::npos) {
			std::string after_pred = trimmed.substr(space_pos + 1);
			size_t instr_start = after_pred.find_first_not_of(" \t");
			if (instr_start != std::string::npos) {
				std::string instr = after_pred.substr(instr_start);
				if (instr.find("bra ") == 0 || instr.find("bra\t") == 0) {
					return true;
				}
			}
		}
	}
	
	return false;
}

// Helper: Check if a line has an explicit BB label
static std::string extract_bb_label(const std::string &line)
{
	// Look for pattern: $L__BB<digits>_<digits>:
	static std::regex bb_pattern(R"(\$L__BB[0-9]+_[0-9]+:)");
	std::smatch match;
	
	if (std::regex_search(line, match, bb_pattern)) {
		std::string label = match.str();
		// Remove trailing ':'
		return label.substr(0, label.length() - 1);
	}
	
	return "";
}

// Helper: Extract BB label from a line that contains only a label (no instruction)
// Returns the label without the colon, or empty string if not a standalone BB label
static std::string extract_standalone_bb_label(const std::string &line)
{
	size_t start = line.find_first_not_of(" \t");
	if (start == std::string::npos) {
		return "";
	}
	
	std::string trimmed = line.substr(start);
	
	// Check if it's a BB label pattern: $L__BB<digits>_<digits>:
	if (trimmed.find("$L__BB") != 0) {
		return "";
	}
	
	size_t colon_pos = trimmed.find(':');
	if (colon_pos == std::string::npos) {
		return "";
	}
	
	// Check that there's nothing after the colon (standalone label)
	std::string after_colon = trimmed.substr(colon_pos + 1);
	size_t after_start = after_colon.find_first_not_of(" \t\r\n");
	if (after_start != std::string::npos) {
		return ""; // There's an instruction after, not standalone
	}
	
	// Return label without the colon
	return trimmed.substr(0, colon_pos);
}

// Find all basic blocks in a kernel by analyzing control flow
// BB0 is always the first actual instruction (skipping declarations, registers, etc.)
// Subsequent BBs start after control flow instructions or at explicit labels
// The last BB (closing brace) is excluded
static std::map<int, BBInfo>
find_all_basic_blocks(const std::string &kernel_section)
{
	std::map<int, BBInfo> bb_map;
	std::istringstream iss(kernel_section);
	std::string line;
	size_t line_num = 0;
	size_t char_pos = 0;
	int bb_num = 0;
	bool after_control_flow = false;
	bool found_first_instruction = false;
	
	// Track pending standalone BB label (label on its own line)
	std::string pending_bb_label;
	size_t pending_label_line_num = 0;
	size_t pending_label_char_pos = 0;
	
	// First pass: collect all BBs
	std::vector<std::tuple<int, BBInfo>> temp_bbs;
	
	while (std::getline(iss, line)) {
		// Check for standalone BB label (label on its own line, no instruction)
		std::string standalone_label = extract_standalone_bb_label(line);
		if (!standalone_label.empty()) {
			// Remember this label for the next instruction
			pending_bb_label = standalone_label;
			pending_label_line_num = line_num;
			pending_label_char_pos = char_pos;
			char_pos += line.length() + 1;
			line_num++;
			continue;
		}
		
		// Skip non-instructions (declarations, directives, braces, label-only lines)
		if (is_non_instruction(line)) {
			char_pos += line.length() + 1;
			line_num++;
			continue;
		}
		
		// Check for explicit BB label on the same line as instruction
		std::string label = extract_bb_label(line);
		
		// If we have a pending standalone label, use it
		if (!pending_bb_label.empty()) {
			BBInfo info;
			info.ptx_label = pending_bb_label;
			info.line_content = line;
			info.position = char_pos; // Position of the first instruction, not the label
			info.line_number = line_num;
			info.has_explicit_label = true;
			temp_bbs.push_back({bb_num++, info});
			
			pending_bb_label.clear();
			after_control_flow = false;
			found_first_instruction = true;
			
			// Check if this line is also a control flow instruction
			if (is_control_flow_instruction(line)) {
				after_control_flow = true;
			}
		} else if (!label.empty()) {
			// Explicit label found on same line - this starts a new BB
			BBInfo info;
			info.ptx_label = label;
			info.line_content = line;
			info.position = char_pos;
			info.line_number = line_num;
			info.has_explicit_label = true;
			temp_bbs.push_back({bb_num++, info});
			
			after_control_flow = false;
			found_first_instruction = true;
			
			// Check if this line is also a control flow instruction
			if (is_control_flow_instruction(line)) {
				after_control_flow = true;
			}
		} else if (!found_first_instruction) {
			// First actual instruction = BB0
			BBInfo info;
			info.ptx_label = ""; // No explicit label
			info.line_content = line;
			info.position = char_pos;
			info.line_number = line_num;
			info.has_explicit_label = false;
			temp_bbs.push_back({0, info});
			bb_num = 1;
			
			found_first_instruction = true;
			
			// Check if this first line is also a control flow instruction
			if (is_control_flow_instruction(line)) {
				after_control_flow = true;
			}
		} else if (after_control_flow) {
			// Previous instruction was control flow, this starts a new BB
			BBInfo info;
			info.ptx_label = ""; // No explicit label
			info.line_content = line;
			info.position = char_pos;
			info.line_number = line_num;
			info.has_explicit_label = false;
			temp_bbs.push_back({bb_num++, info});
			
			after_control_flow = false;
			
			// Check if this line is also a control flow instruction
			if (is_control_flow_instruction(line)) {
				after_control_flow = true;
			}
		} else {
			// Regular instruction - check if it's control flow
			if (is_control_flow_instruction(line)) {
				after_control_flow = true;
			}
		}
		
		char_pos += line.length() + 1; // +1 for newline
		line_num++;
	}
	
	// Filter out the last BB if it's just a closing brace or empty
	// (Already filtered by is_non_instruction, but be extra safe)
	for (const auto &[num, info] : temp_bbs) {
		bb_map[num] = info;
	}
	
	return bb_map;
}




// Extract kernel name and bb number from attach point name
// e.g., "kprobe/_Z9vectorAddPKfS0_Pf__BB0" -> {"_Z9vectorAddPKfS0_Pf", 0, {}, true}
// e.g., "kprobe/_Z9vectorAddPKfS0_Pf__BB5__rd1__r1" -> {"_Z9vectorAddPKfS0_Pf", 5, {"rd1", "r1"}, true}
// Returns: {kernel_name, bb_number, register_list, is_valid}
static std::tuple<std::string, int, std::vector<std::string>, bool>
parse_kprobe_name(const std::string &attach_point)
{
	// First, strip the "kprobe/" prefix if present
	std::string name = attach_point;
	if (name.find("kprobe/") == 0) {
		name = name.substr(7);
	}
	
	// Find __BB<n> pattern
	static std::regex bb_pattern(R"(__BB([0-9]+))");
	std::smatch bb_match;
	
	if (!std::regex_search(name, bb_match, bb_pattern)) {
		return { "", -1, {}, false };
	}
	
	// Extract kernel name (everything before __BB)
	size_t bb_pos = name.find("__BB");
	if (bb_pos == std::string::npos || bb_pos == 0) {
		return { "", -1, {}, false };
	}
	std::string kernel_name = name.substr(0, bb_pos);
	
	// Extract BB number
	int bb_num = std::stoi(bb_match[1].str());
	
	// Extract register list (everything after __BB<n>)
	std::vector<std::string> registers;
	size_t after_bb = bb_match.position() + bb_match.length();
	if (after_bb < name.length()) {
		std::string reg_part = name.substr(after_bb);
		// Parse __reg1__reg2__... format
		size_t pos = 0;
		while (pos < reg_part.length()) {
			// Skip leading "__"
			if (reg_part.substr(pos, 2) == "__") {
				pos += 2;
			} else {
				break; // Invalid format
			}
			// Find next "__" or end
			size_t next = reg_part.find("__", pos);
			std::string reg;
			if (next == std::string::npos) {
				reg = reg_part.substr(pos);
				pos = reg_part.length();
			} else {
				reg = reg_part.substr(pos, next - pos);
				pos = next;
			}
			if (!reg.empty()) {
				registers.push_back(reg);
			}
		}
	}
	
	return { kernel_name, bb_num, registers, true };
}

// Determine PTX register type from register name prefix
// Returns: {"u64", "rd"} for 64-bit, {"u32", "r"} for 32-bit, {"f32", "f"} for float, etc.
static std::pair<std::string, std::string> get_reg_type_info(const std::string &reg_name)
{
	if (reg_name.empty()) {
		return {"u64", "rd"};
	}
	
	// Check prefix to determine type
	if (reg_name[0] == 'r' && reg_name.length() > 1 && reg_name[1] == 'd') {
		// rd* = 64-bit integer
		return {"u64", "rd"};
	} else if (reg_name[0] == 'r') {
		// r* = 32-bit integer
		return {"u32", "r"};
	} else if (reg_name[0] == 'f' && reg_name.length() > 1 && reg_name[1] == 'd') {
		// fd* = 64-bit float (double)
		return {"f64", "fd"};
	} else if (reg_name[0] == 'f') {
		// f* = 32-bit float
		return {"f32", "f"};
	} else if (reg_name[0] == 'p') {
		// p* = predicate (1-bit, but we store as u32)
		return {"pred", "p"};
	} else if (reg_name[0] == 'b') {
		// b* = bit (could be various sizes, assume 32)
		return {"b32", "b"};
	}
	
	// Default to 64-bit
	return {"u64", "rd"};
}

// Generate PTX code to save registers to a context buffer
// The context layout is:
//   [0]: number of registers (u64)
//   [8]: register 0 value (u64)
//   [16]: register 1 value (u64)
//   ...
static std::string generate_reg_save_code(const std::vector<std::string> &registers,
					   const std::string &ctx_reg,
					   const std::string &temp_reg)
{
	std::ostringstream oss;
	
	// Store register count at offset 0
	oss << "\t// BB kprobe: save " << registers.size() << " registers to context\n";
	oss << "\tmov.u64 " << temp_reg << ", " << registers.size() << ";\n";
	oss << "\tst.local.u64 [" << ctx_reg << "], " << temp_reg << ";\n";
	
	// Store each register value
	for (size_t i = 0; i < registers.size(); i++) {
		const std::string &reg = registers[i];
		auto [type_str, prefix] = get_reg_type_info(reg);
		size_t offset = 8 + i * 8; // Each slot is 8 bytes
		
		if (type_str == "pred") {
			// Predicate registers need special handling - select 1 or 0
			oss << "\tselp.u64 " << temp_reg << ", 1, 0, %" << reg << ";\n";
			oss << "\tst.local.u64 [" << ctx_reg << "+" << offset << "], " << temp_reg << ";\n";
		} else if (type_str == "u32" || type_str == "b32") {
			// 32-bit integer - zero-extend to 64-bit
			oss << "\tcvt.u64.u32 " << temp_reg << ", %" << reg << ";\n";
			oss << "\tst.local.u64 [" << ctx_reg << "+" << offset << "], " << temp_reg << ";\n";
		} else if (type_str == "f32") {
			// 32-bit float - bitcast to u32, then zero-extend
			oss << "\tmov.b32 %__bb_tmp_r32, %" << reg << ";\n";
			oss << "\tcvt.u64.u32 " << temp_reg << ", %__bb_tmp_r32;\n";
			oss << "\tst.local.u64 [" << ctx_reg << "+" << offset << "], " << temp_reg << ";\n";
		} else if (type_str == "f64") {
			// 64-bit float - bitcast to u64
			oss << "\tmov.b64 " << temp_reg << ", %" << reg << ";\n";
			oss << "\tst.local.u64 [" << ctx_reg << "+" << offset << "], " << temp_reg << ";\n";
		} else {
			// 64-bit integer - direct store
			oss << "\tmov.u64 " << temp_reg << ", %" << reg << ";\n";
			oss << "\tst.local.u64 [" << ctx_reg << "+" << offset << "], " << temp_reg << ";\n";
		}
	}
	
	return oss.str();
}

static std::pair<std::string, bool>
patch_bb_kprobe(const std::string &ptx, const std::string &kernel,
		 int bb_num, const std::vector<std::string> &registers,
		 const std::vector<uint64_t> &ebpf_words)
{
	if (ebpf_words.empty()) {
		return { ptx, false };
	}

	// Find the kernel body in PTX
	auto body = ptxpass::find_kernel_body(ptx, kernel);
	if (body.first == std::string::npos) {
		std::cerr << "Kernel " << kernel << " not found in PTX\n";
		return { ptx, false };
	}

	std::string out = ptx;
	std::string section = out.substr(body.first, body.second - body.first);
	
	// Find all basic blocks by analyzing control flow
	auto bb_map = find_all_basic_blocks(section);
	
	if (bb_map.empty()) {
		std::cerr << "No basic blocks found in kernel " << kernel << "\n";
		return { ptx, false };
	}
	
	// Print BB map for debugging
	std::cerr << "Basic block map for kernel " << kernel << ":\n";
	for (const auto &[num, info] : bb_map) {
		if (info.has_explicit_label) {
			std::cerr << "  BB" << num << " -> line " << info.line_number 
			          << ": " << info.ptx_label << "\n";
		} else {
			// Trim line content for display
			std::string trimmed = info.line_content;
			if (trimmed.length() > 60) {
				trimmed = trimmed.substr(0, 57) + "...";
			}
			std::cerr << "  BB" << num << " -> line " << info.line_number 
			          << ": " << trimmed << "\n";
		}
	}

    if (!registers.empty()) {
		std::cerr << "Registers to capture: ";
		for (const auto &r : registers) {
			std::cerr << "%" << r << " ";
		}
		std::cerr << "\n";
	}
	
	// Check if requested BB number exists
	if (bb_map.find(bb_num) == bb_map.end()) {
		std::cerr << "Basic block BB" << bb_num << " not found in kernel " 
		          << kernel << "\n";
		std::cerr << "Available basic blocks: ";
		for (const auto &[num, info] : bb_map) {
			std::cerr << "BB" << num << " ";
		}
		std::cerr << "\n";
		return { ptx, false };
	}
	
	// Get the BB info
	const BBInfo &bb_info = bb_map[bb_num];
	
	// Generate unique function name for this BB kprobe
	// Include register hash if registers are specified
	std::string fname = kernel + "__bb_kprobe_BB" + std::to_string(bb_num);
	if (!registers.empty()) {
		fname += "_regs";
	}
	
	// Compile eBPF to PTX function
	// Use with_arguments=true when we have registers to capture (so we can pass ctx)
	// Use with_arguments=false when no registers (simple probe call)
	bool needs_args = !registers.empty();
	auto func_ptx = ptxpass::compile_ebpf_to_ptx_from_words(
		ebpf_words, "sm_86", fname, true, needs_args);

	// Find the insertion point based on line number and position
	size_t insert_pos;
	
	if (bb_info.has_explicit_label) {
		// For explicit labels, insert after the label line
		std::string label_pattern = bb_info.ptx_label + ":";
		size_t label_pos = section.find(label_pattern);
		if (label_pos == std::string::npos) {
			std::cerr << "Label " << bb_info.ptx_label << " not found\n";
			return { ptx, false };
		}
		insert_pos = label_pos + label_pattern.length();
		// Skip to next line
		size_t newline_pos = section.find('\n', insert_pos);
		if (newline_pos != std::string::npos) {
			insert_pos = newline_pos + 1;
		}
	} else {
		// For implicit BBs, insert at the position of the first instruction
		insert_pos = bb_info.position;
	}

	// Generate the instrumentation code
	std::ostringstream instr_code;
	
	if (registers.empty()) {
		// No register capture - simple call
		instr_code << "\tcall " << fname << ";\n";
	} else {
		// With register capture - need to:
		// 1. Allocate local memory for context
		// 2. Save registers to context
		// 3. Call function with context pointer as parameter
		
		// Calculate context size: 8 bytes for count + 8 bytes per register
		size_t ctx_size = 8 + registers.size() * 8;
		
		// We need to add register declarations to the kernel
		// Find where to insert register declarations (after .reg declarations)
		size_t reg_decl_pos = section.find(".reg");
		if (reg_decl_pos != std::string::npos) {
			// Find end of register declarations block
			size_t decl_end = reg_decl_pos;
			while (true) {
				size_t next_line = section.find('\n', decl_end);
				if (next_line == std::string::npos) break;
				size_t next_start = section.find_first_not_of(" \t\n", next_line + 1);
				if (next_start == std::string::npos) break;
				if (section.substr(next_start, 4) == ".reg") {
					decl_end = next_line + 1;
				} else {
					decl_end = next_line;
					break;
				}
			}
			
			// Insert our temporary register and local memory declarations
			std::ostringstream decls;
			decls << "\n\t// BB kprobe temporary registers and context\n";
			decls << "\t.reg .u64 %__bb_ctx_ptr;\n";
			decls << "\t.reg .u64 %__bb_tmp_rd;\n";
			
			// Check if we need a 32-bit temp register for float conversion
			bool need_r32 = false;
			for (const auto &r : registers) {
				auto [type_str, prefix] = get_reg_type_info(r);
				if (type_str == "f32") {
					need_r32 = true;
					break;
				}
			}
			if (need_r32) {
				decls << "\t.reg .u32 %__bb_tmp_r32;\n";
			}
			
			decls << "\t.local .align 8 .b8 __bb_reg_ctx[" << ctx_size << "];\n";
			
			section.insert(decl_end + 1, decls.str());
			
			// Adjust insert_pos since we added content before it
			insert_pos += decls.str().length();
		}
		
		// Generate the instrumentation code at the BB entry
		instr_code << "\t// BB" << bb_num << " kprobe with register capture\n";
		// Get local address for storing registers
		instr_code << "\tmov.u64 %__bb_ctx_ptr, __bb_reg_ctx;\n";
		instr_code << generate_reg_save_code(registers, "%__bb_ctx_ptr", "%__bb_tmp_rd");
		// Convert local address to generic address for function call
		instr_code << "\tcvta.local.u64 %__bb_ctx_ptr, %__bb_ctx_ptr;\n";
		
		// Call the probe function with context pointer
		// The eBPF function signature is: (void *ctx, uint64_t ctx_len)
		// We pass the context pointer and the context size
		instr_code << "\t{\n";
		instr_code << "\t\t.param .u64 __bb_param_ctx;\n";
		instr_code << "\t\t.param .u64 __bb_param_ctx_len;\n";
		instr_code << "\t\tst.param.u64 [__bb_param_ctx], %__bb_ctx_ptr;\n";
		instr_code << "\t\tst.param.u64 [__bb_param_ctx_len], " << ctx_size << ";\n";
		instr_code << "\t\tcall " << fname << ", (__bb_param_ctx, __bb_param_ctx_len);\n";
		instr_code << "\t}\n";
	}

	section.insert(insert_pos, instr_code.str());

	// Replace the kernel body with modified version
	out.replace(body.first, body.second - body.first, section);
	// Prepend the eBPF function PTX
	out = func_ptx + "\n" + out;

	ptxpass::log_transform_stats("bb_kprobe", 1, ptx.size(), out.size());
	return { out, true };
}

extern "C" void print_config(int length, char *out)
{
	auto cfg = get_default_config();
	nlohmann::json output_json;
	ptxpass::pass_config::to_json(output_json, cfg);
	snprintf(out, length, "%s", output_json.dump().c_str());
}

extern "C" int process_input(const char *input, int length, char *output)
{
	using namespace ptxpass;
	auto cfg = get_default_config();
	try {
		auto runtime_request = pass_runtime_request_from_string(input);
		if (!validate_input(runtime_request.input.full_ptx,
				    cfg.validation)) {
			return ExitCode::TransformFailed;
		}

		// Parse the attach point to extract kernel name, BB number, and registers
		// The to_patch_kernel contains attach point like:
		//   "kprobe/_Z9vectorAddPKfS0_Pf__BB0" (no registers)
		//   "kprobe/_Z9vectorAddPKfS0_Pf__BB1__rd1__r1__f1" (with registers)
		auto [kernel_name, bb_num, registers, is_valid] =
			parse_kprobe_name(runtime_request.input.to_patch_kernel);

		if (!is_valid || kernel_name.empty() || bb_num < 0) {
			std::cerr << "Failed to parse BB kprobe name: "
				  << runtime_request.input.to_patch_kernel
				  << "\n";
			std::cerr << "Expected format: kprobe/<kernel_name>__BB<number>[__reg1__reg2__...]\n";
			return ExitCode::ConfigError;
		}

		// For BB kprobes with kernel-specific names, we use the kernel_name
		// from the attach point to find the correct kernel in PTX
		auto [out, modified] = patch_bb_kprobe(
			runtime_request.input.full_ptx, kernel_name, bb_num,
			registers,
			runtime_request.get_uint64_ebpf_instructions());

		snprintf(
			output, length, "%s",
			emit_runtime_response_and_return(out, modified).c_str());
		return ExitCode::Success;
	} catch (const std::runtime_error &e) {
		std::cerr << "Runtime error: " << e.what() << "\n";
		return ExitCode::ConfigError;
	} catch (const std::exception &e) {
		std::cerr << "Exception: " << e.what() << "\n";
		return ExitCode::UnknownError;
	} catch (...) {
		std::cerr << "Unknown error\n";
		return ExitCode::UnknownError;
	}
}
