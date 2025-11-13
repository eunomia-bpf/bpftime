#include <string>
#include <vector>
#include <regex>
#include <sstream>
#include <fstream>
#include <algorithm> // Required for std::find, std::find_if
#include <iostream> // For potential debugging, can be removed
#include <stdexcept> // Required for std::invalid_argument, std::out_of_range

namespace bpftime::attach
{

struct RegisterInfo {
	std::string type;
	std::string baseName; // Will store the clean base, e.g., "%rd" for
			      // "%rd<14>"
	int count;
	std::vector<std::string> names; // Will store individual names, e.g.,
					// "%rd0", "%rd1"
	int sizeInBytes;
	std::string ptxTypeModifier;
};

int getRegisterSizeInBytes(const std::string &type)
{
	if (type == ".b8" || type == ".s8" || type == ".u8")
		return 1;
	if (type == ".b16" || type == ".s16" || type == ".u16" ||
	    type == ".f16")
		return 2;
	if (type == ".b32" || type == ".s32" || type == ".u32" ||
	    type == ".f32")
		return 4;
	if (type == ".b64" || type == ".s64" || type == ".u64" ||
	    type == ".f64" || type == ".f64x2")
		return 8;
	if (type == ".pred")
		return 1;
	return 0;
}

std::string getPtxStorageTypeModifier(const RegisterInfo &regInfo)
{
	if (regInfo.sizeInBytes == 1)
		return ".u8";
	if (regInfo.sizeInBytes == 2)
		return ".u16";
	if (regInfo.sizeInBytes == 4)
		return ".u32";
	if (regInfo.sizeInBytes == 8)
		return ".u64";
	return regInfo.type;
}

std::string add_register_guard_for_ebpf_ptx_func(const std::string &ptxCode)
{
	std::string resultPtx;
	std::stringstream ptxStream(ptxCode);
	std::string line;

	// CORRECTED Regex: Group 2 now only captures the base name before <N>
	std::regex funcDefRegex(
		R"(^\s*(?:\.visible\s+)?(?:\.func|\.entry)\s+([a-zA-Z_][a-zA-Z0-9_@$.]*)(?:\s*\(|\s*$))");
	std::regex regDeclRegex(
		R"(^\s*\.reg\s+(\.\w+)\s+([%a-zA-Z_][a-zA-Z0-9_@$.]*)(?:<(\d+)>)?\s*;)");
	std::regex localDeclRegex(R"(^\s*\.local\s+)");
	std::regex commentOrEmptyRegex(R"(^\s*(//.*|\s*$))");
	std::regex openingBraceRegex(R"(^\s*\{\s*$)"); // Matches standalone {
	std::regex closingBraceRegex(R"(^\s*\}\s*$)"); // Matches standalone }
	std::regex hasOpeningBraceRegex(R"(\{)"); // Matches any { in the line
	std::regex hasClosingBraceRegex(R"(\})"); // Matches any } in the line
	std::regex retRegex(R"(^\s*(?:@%p\d+\s+)?ret\s*;)");

	std::vector<std::string> currentFunctionLines;
	bool inFunctionDefinition = false;
	bool inFunctionBody = false;
	int braceDepth = 0; // Track brace nesting depth

	const std::string tempBaseReg = "%rd_ptx_instr_base";
	const std::string tempAddrReg = "%rd_ptx_instr_addr";
	const std::string registerSaveAreaName = "__ptx_register_save_area";

	while (std::getline(ptxStream, line)) {
		std::smatch match;

		if (!inFunctionBody && !inFunctionDefinition) {
			if (std::regex_search(line, match, funcDefRegex)) {
				inFunctionDefinition = true;
				currentFunctionLines.push_back(line);
			} else {
				resultPtx += line + "\n";
			}
		} else if (inFunctionDefinition) {
			currentFunctionLines.push_back(line);
			// Check if this line contains opening brace(s)
			// Count them properly in case there are multiple
			int openBracesInLine = 0;
			int closeBracesInLine = 0;
			for (char c : line) {
				if (c == '{')
					openBracesInLine++;
				else if (c == '}')
					closeBracesInLine++;
			}
			if (openBracesInLine > 0) {
				inFunctionBody = true;
				inFunctionDefinition = false;
				// Initialize braceDepth with net braces from
				// this line
				braceDepth =
					openBracesInLine - closeBracesInLine;
			}
		} else if (inFunctionBody) {
			// Count ALL braces on this line by scanning each
			// character Cannot use regex_search because it only
			// checks existence, not count
			for (char c : line) {
				if (c == '{') {
					braceDepth++;
				} else if (c == '}') {
					braceDepth--;
				}
			}

			currentFunctionLines.push_back(line);

			// Only process the function when we reach the matching
			// closing brace
			if (braceDepth == 0) {
				std::vector<RegisterInfo> registersToSaveInFunc;

				// 1. Determine the actual insertion point for
				// declarations
				size_t currentFuncActualOpeningBraceIdx =
					std::string::npos;
				size_t searchStartOffsetBrace = 0;
				if (!currentFunctionLines.empty() &&
				    std::regex_search(currentFunctionLines[0],
						      funcDefRegex)) {
					searchStartOffsetBrace = 1;
				}
				for (size_t i = searchStartOffsetBrace;
				     i < currentFunctionLines.size(); ++i) {
					// Look for ANY line with opening brace,
					// not just standalone
					if (std::regex_search(
						    currentFunctionLines[i],
						    hasOpeningBraceRegex)) {
						currentFuncActualOpeningBraceIdx =
							i;
						break;
					}
					// This part ensures we don't
					// accidentally break early if there are
					// comments/decls before the brace
					if (!std::regex_search(
						    currentFunctionLines[i],
						    commentOrEmptyRegex) &&
					    !std::regex_search(
						    currentFunctionLines[i],
						    regDeclRegex) &&
					    !std::regex_search(
						    currentFunctionLines[i],
						    localDeclRegex)) {
						// If we hit a
						// non-declaration/non-comment
						// line before the brace,
						// something is off, but for
						// now, break. This case is less
						// common for well-formed PTX.
						break;
					}
				}

				size_t declsInsertionPointInOriginalLines = 0;
				if (currentFuncActualOpeningBraceIdx !=
				    std::string::npos) {
					declsInsertionPointInOriginalLines =
						currentFuncActualOpeningBraceIdx +
						1; // Start search *after* the
						   // opening brace
					for (size_t i =
						     declsInsertionPointInOriginalLines;
					     i < currentFunctionLines.size();
					     ++i) {
						const auto &l =
							currentFunctionLines[i];
						if (std::regex_search(
							    l,
							    commentOrEmptyRegex) ||
						    std::regex_search(
							    l, regDeclRegex) ||
						    std::regex_search(
							    l,
							    localDeclRegex)) {
							declsInsertionPointInOriginalLines =
								i +
								1; // Move
								   // insertion
								   // point past
								   // this
								   // declaration/comment
						} else {
							break; // Found first
							       // non-declaration/non-comment
							       // line, stop
						}
					}
				} else {
					// Should not happen for valid function
					// definitions
					for (const auto &l :
					     currentFunctionLines) {
						resultPtx += l + "\n";
					}
					currentFunctionLines.clear();
					inFunctionBody = false;
					continue;
				}

				// 2. Collect registers only from the initial
				// declaration block Iterate only through the
				// lines that are part of the initial
				// declaration block This assumes that
				// function-scoped registers are declared at the
				// top of the function, before any actual
				// instructions or nested blocks like callseq.
				// The declsInsertionPointInOriginalLines is the
				// index *after* the last
				// declaration/comment/empty line. So we should
				// iterate up to this index.
				for (size_t i =
					     currentFuncActualOpeningBraceIdx +
					     1; // Start after the opening brace
				     i < declsInsertionPointInOriginalLines &&
				     i < currentFunctionLines.size();
				     ++i) {
					const std::string &funcLine =
						currentFunctionLines[i];
					std::smatch regMatch;
					if (std::regex_search(funcLine,
							      regMatch,
							      regDeclRegex)) {
						RegisterInfo regInfo;
						regInfo.type =
							regMatch[1].str();
						regInfo.baseName =
							regMatch[2]
								.str(); // Now
									// this
									// is
									// the
									// clean
									// base,
									// e.g.,
									// "%rd"
						regInfo.sizeInBytes =
							getRegisterSizeInBytes(
								regInfo.type);

						// Exclude special registers
						// like predicates, stack
						// pointers, and our own
						// temporary registers This
						// implicitly excludes
						// 'temp_param_reg' if it's
						// declared later in a callseq
						// block.
						if (regInfo.type == ".pred" ||
						    regInfo.baseName == "%SP" ||
						    regInfo.baseName ==
							    "%SPL" ||
						    regInfo.sizeInBytes == 0 ||
						    regInfo.baseName ==
							    tempBaseReg ||
						    regInfo.baseName ==
							    tempAddrReg) {
							continue;
						}
						regInfo.ptxTypeModifier =
							getPtxStorageTypeModifier(
								regInfo);

						bool name_conflict = false;
						// Check conflict against the
						// clean baseName
						for (const auto &r :
						     registersToSaveInFunc) {
							if (r.baseName ==
								    regInfo.baseName &&
							    regMatch[3].matched ==
								    r.count >
									    1) {
								name_conflict =
									true;
								break;
							}
						}
						if (name_conflict)
							continue;

						// regMatch[3] contains the
						// number N if <N> was present
						if (regMatch[3].matched &&
						    !regMatch[3].str().empty()) {
							try {
								regInfo.count = std::stoi(
									regMatch[3]
										.str());
								for (int k = 0;
								     k <
								     regInfo.count;
								     ++k) {
									// regInfo.baseName
									// is
									// already
									// the
									// clean
									// base
									// (e.g.,
									// "%rd")
									regInfo.names
										.push_back(
											regInfo.baseName +
											std::to_string(
												k));
								}
							} catch (
								const std::invalid_argument
									&ia) { // Should not happen if regex \d+ matches
								regInfo.count =
									1;
								regInfo.names.push_back(
									regInfo.baseName); // Fallback to use baseName as is
							} catch (
								const std::out_of_range
									&oor) { // Should not happen
								regInfo.count =
									1;
								regInfo.names.push_back(
									regInfo.baseName);
							}
						} else { // Not a vector
							 // register like %r<N>
							regInfo.count = 1;
							regInfo.names.push_back(
								regInfo.baseName); // Use the baseName directly (e.g., "%r1")
						}
						registersToSaveInFunc.push_back(
							regInfo);
					}
				}

				std::string pushCodeBlockStr;
				std::string popCodeBlockStr;
				std::string newLocalAndUtilRegDecls;

				if (!registersToSaveInFunc.empty()) {
					int totalBytesForSavedRegs = 0;
					std::vector<std::pair<std::string, int>>
						regOffsetMap; // Stores
							      // individual
							      // names like
							      // "%rd0"
					int currentStackOffset = 0;
					for (const auto &regInfo :
					     registersToSaveInFunc) {
						for (const auto &regName :
						     regInfo.names) { // regInfo.names
								      // contains
								      // "%rd0",
								      // "%rd1",
								      // etc.
							regOffsetMap.push_back(
								{ regName,
								  currentStackOffset });
							currentStackOffset +=
								regInfo.sizeInBytes; // Use actual size, though example uses 8 for alignment
										     // For simplicity and consistency with example, stick to 8 if all are 8-byte slots
										     // currentStackOffset += 8;
						}
					}
					// Recalculate currentStackOffset
					// ensuring 8-byte alignment for each
					// slot if necessary,
					// or sum of actual sizes if mixed. The
					// example implies 8-byte slots.
					currentStackOffset = 0;
					for (const auto &regInfo :
					     registersToSaveInFunc) {
						currentStackOffset +=
							regInfo.names.size() *
							8; // Each name gets an
							   // 8-byte slot
					}
					totalBytesForSavedRegs =
						currentStackOffset;

					int saveAreaSize = std::max(
						totalBytesForSavedRegs, 8);
					if (saveAreaSize % 8 != 0)
						saveAreaSize =
							(saveAreaSize + 7) & ~7;

					newLocalAndUtilRegDecls =
						"\t.local .align 8 .b8 \t" +
						registerSaveAreaName + "[" +
						std::to_string(saveAreaSize) +
						"];\n";
					newLocalAndUtilRegDecls +=
						"\t.reg .b64 " + tempBaseReg +
						";\n";
					newLocalAndUtilRegDecls +=
						"\t.reg .b64 " + tempAddrReg +
						";\n";

					if (totalBytesForSavedRegs > 0) {
						std::stringstream pushSs, popSs;
						currentStackOffset =
							0; // Reset for
							   // generating
							   // push/pop code
							   // offsets

						pushSs << "\t// --- BEGIN REGISTER SAVING (PUSH to "
						       << registerSaveAreaName
						       << ") ---\n";
						// Convert local symbol to a
						// local-space pointer
						pushSs << "\tcvta.local.u64 "
						       << tempBaseReg << ", "
						       << registerSaveAreaName
						       << ";\n";
						for (const auto &regInfo_outer :
						     registersToSaveInFunc) { // Iterate to maintain order if needed
							for (const auto &
								     individualRegName :
							     regInfo_outer
								     .names) {
								pushSs << "\tadd.u64 "
								       << tempAddrReg
								       << ", "
								       << tempBaseReg
								       << ", "
								       << currentStackOffset
								       << ";\n";
								pushSs << "\tst.local"
								       << regInfo_outer
										  .ptxTypeModifier
								       << " ["
								       << tempAddrReg
								       << "], "
								       << individualRegName
								       << ";\n";
								currentStackOffset +=
									8; // Each
									   // slot
									   // is
									   // 8
									   // bytes
							}
						}
						pushSs << "\t// --- END REGISTER SAVING (PUSH to "
						       << registerSaveAreaName
						       << ") ---\n";
						pushCodeBlockStr = pushSs.str();

						currentStackOffset = 0; // Reset
									// for
									// pop
						std::vector<std::pair<
							std::string,
							const RegisterInfo *>>
							popOrderMap;
						for (const auto &regInfo_outer :
						     registersToSaveInFunc) {
							for (const auto &
								     individualRegName :
							     regInfo_outer
								     .names) {
								popOrderMap.push_back(
									{ individualRegName,
									  &regInfo_outer });
							}
						}

						popSs << "\n\t// --- BEGIN REGISTER RESTORING (POP from "
						      << registerSaveAreaName
						      << ") ---\n";
						// Convert local symbol to a
						// local-space pointer
						popSs << "\tcvta.local.u64 "
						      << tempBaseReg << ", "
						      << registerSaveAreaName
						      << ";\n";
						// Iterate in reverse order of
						// saving
						currentStackOffset =
							totalBytesForSavedRegs -
							8; // Start from the
							   // last saved
							   // register's offset
						for (auto rit =
							     popOrderMap
								     .rbegin();
						     rit != popOrderMap.rend();
						     ++rit) {
							popSs << "\tadd.u64 "
							      << tempAddrReg
							      << ", "
							      << tempBaseReg
							      << ", "
							      << currentStackOffset
							      << ";\n";
							popSs << "\tld.local"
							      << rit->second
									 ->ptxTypeModifier
							      << " "
							      << rit->first
							      << ", ["
							      << tempAddrReg
							      << "];\n";
							currentStackOffset -= 8;
						}
						popSs << "\t// --- END REGISTER RESTORING (POP from "
						      << registerSaveAreaName
						      << ") ---\n";
						popCodeBlockStr = popSs.str();
					}
				}

				std::vector<std::string> instrumentedLines;
				bool newDeclsActuallyInserted = false;
				bool pushCodeActuallyInserted = false;

				// The 'currentFuncActualOpeningBraceIdx' and
				// 'declsInsertionPointInOriginalLines' are
				// already calculated above.
				size_t pushInsertionPointInOriginalLines =
					declsInsertionPointInOriginalLines; // Push code goes right after declarations

				for (size_t i = 0;
				     i < currentFunctionLines.size(); ++i) {
					if (i == declsInsertionPointInOriginalLines &&
					    !newLocalAndUtilRegDecls.empty() &&
					    !newDeclsActuallyInserted) {
						instrumentedLines.push_back(
							newLocalAndUtilRegDecls);
						newDeclsActuallyInserted = true;
					}

					if (i == pushInsertionPointInOriginalLines &&
					    !pushCodeBlockStr.empty() &&
					    !pushCodeActuallyInserted) {
						// This condition handles the
						// case where decls and push are
						// at the same point
						if (declsInsertionPointInOriginalLines ==
							    pushInsertionPointInOriginalLines &&
						    !newDeclsActuallyInserted &&
						    !newLocalAndUtilRegDecls
							     .empty()) {
							// If new decls were not
							// inserted yet, but
							// they should be at
							// this point, insert
							// them before push
							// code. This should be
							// covered by the first
							// 'if' but as a
							// safeguard.
							if (!newLocalAndUtilRegDecls
								     .empty()) {
								instrumentedLines
									.push_back(
										newLocalAndUtilRegDecls);
								newDeclsActuallyInserted =
									true;
							}
						}
						instrumentedLines.push_back(
							pushCodeBlockStr);
						pushCodeActuallyInserted = true;
					}

					if (!popCodeBlockStr.empty() &&
					    std::regex_search(
						    currentFunctionLines[i],
						    retRegex)) {
						instrumentedLines.push_back(
							popCodeBlockStr);
					}
					instrumentedLines.push_back(
						currentFunctionLines[i]);
				}

				// Handle cases where insertion points are at
				// the very end of the function lines
				if (declsInsertionPointInOriginalLines ==
					    currentFunctionLines.size() &&
				    !newLocalAndUtilRegDecls.empty() &&
				    !newDeclsActuallyInserted) {
					instrumentedLines.push_back(
						newLocalAndUtilRegDecls);
				}
				if (pushInsertionPointInOriginalLines ==
					    currentFunctionLines.size() &&
				    !pushCodeBlockStr.empty() &&
				    !pushCodeActuallyInserted) {
					instrumentedLines.push_back(
						pushCodeBlockStr);
				}

				for (const auto &instrLine :
				     instrumentedLines) {
					resultPtx +=
						instrLine +
						(instrLine.empty() ||
								 instrLine.back() ==
									 '\n' ?
							 "" :
							 "\n");
				}
				currentFunctionLines.clear();
				inFunctionBody = false;
				braceDepth = 0; // Reset brace depth for next
						// function
			}
		}
	}
	if (inFunctionDefinition || inFunctionBody) {
		// If the stream ended while still in a function, append
		// remaining lines
		for (const auto &l : currentFunctionLines) {
			resultPtx += l + "\n";
		}
	}

	// Debug: Save result to a temp file for inspection (always enabled for
	// debugging)
	static int debug_counter = 0;
	std::string debug_filename = "/tmp/ptx_register_guard_output_" +
				     std::to_string(debug_counter++) + ".ptx";
	std::ofstream debug_out(debug_filename);
	if (debug_out.is_open()) {
		debug_out << resultPtx;
		debug_out.close();
		// Log to stderr so it appears in CI logs
		std::cerr << "[PTX_DEBUG] Saved register-guarded PTX (#"
			  << (debug_counter - 1) << ") to " << debug_filename
			  << " (" << resultPtx.size() << " bytes)" << std::endl;
	}

	return resultPtx;
}

} // namespace bpftime::attach
