#include <string>
#include <vector>
#include <regex>
#include <sstream>
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
	std::regex openingBraceRegex(R"(^\s*\{\s*$)");
	std::regex closingBraceRegex(R"(^\s*\}\s*$)");
	std::regex retRegex(R"(^\s*(?:@%p\d+\s+)?ret\s*;)");

	std::vector<std::string> currentFunctionLines;
	bool inFunctionDefinition = false;
	bool inFunctionBody = false;

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
			if (std::regex_search(line, openingBraceRegex)) {
				inFunctionBody = true;
				inFunctionDefinition = false;
			}
		} else if (inFunctionBody) {
			currentFunctionLines.push_back(line);
			if (std::regex_search(line, closingBraceRegex)) {
				std::vector<RegisterInfo> registersToSaveInFunc;

				for (size_t i = 0;
				     i < currentFunctionLines.size(); ++i) {
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
						pushSs << "\tmov.u64 "
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
						popSs << "\tmov.u64 "
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
					if (std::regex_search(
						    currentFunctionLines[i],
						    openingBraceRegex)) {
						currentFuncActualOpeningBraceIdx =
							i;
						break;
					}
					if (!std::regex_search(
						    currentFunctionLines[i],
						    commentOrEmptyRegex) &&
					    !std::regex_search(
						    currentFunctionLines[i],
						    regDeclRegex) &&
					    !std::regex_search(
						    currentFunctionLines[i],
						    localDeclRegex)) {
						break;
					}
				}

				size_t declsInsertionPointInOriginalLines = 0;
				size_t pushInsertionPointInOriginalLines = 0;

				if (currentFuncActualOpeningBraceIdx !=
				    std::string::npos) {
					declsInsertionPointInOriginalLines =
						currentFuncActualOpeningBraceIdx +
						1;
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
								i + 1;
						} else {
							break;
						}
					}

					pushInsertionPointInOriginalLines =
						declsInsertionPointInOriginalLines;
					for (size_t i =
						     pushInsertionPointInOriginalLines;
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
							pushInsertionPointInOriginalLines =
								i + 1;
						} else {
							break;
						}
					}
				} else {
					for (const auto &l :
					     currentFunctionLines) {
						resultPtx += l + "\n";
					}
					currentFunctionLines.clear();
					inFunctionBody = false;
					continue;
				}

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
						if (declsInsertionPointInOriginalLines ==
							    pushInsertionPointInOriginalLines &&
						    !newDeclsActuallyInserted &&
						    !newLocalAndUtilRegDecls
							     .empty()) {
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
			}
		}
	}
	if (inFunctionDefinition || inFunctionBody) {
		for (const auto &l : currentFunctionLines) {
			resultPtx += l + "\n";
		}
	}
	return resultPtx;
}

} // namespace bpftime::attach
