#include <string>
#include <vector>
#include <regex>
namespace bpftime::attach
{
// Done by Gemini

// (RegisterInfo struct and getRegisterSizeInBytes, getPtxTypeModifier functions
// remain the same)
struct RegisterInfo {
	std::string type;
	std::string baseName;
	int count;
	std::vector<std::string> names;
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
		return 1; // Typically 1 bit, often handled as a byte for saving
	return 0;
}

std::string getPtxTypeModifier(const std::string &regType, int sizeInBytes)
{
	if (regType == ".pred")
		return ".b8";
	switch (sizeInBytes) {
	case 1:
		return ".u8";
	case 2:
		return ".u16";
	case 4:
		return ".u32";
	case 8:
		return ".u64";
	default:
		return regType;
	}
}

std::string add_register_guard_for_ebpf_ptx_func(const std::string &ptxCode)
{
	std::string resultPtx;
	std::stringstream ptxStream(ptxCode);
	std::string line;

	std::regex funcDefRegex(
		R"(^\s*(?:\.visible\s+)?(?:\.func|\.entry)\s+([a-zA-Z_][a-zA-Z0-9_@$.]*)(?:\s*\(|\s*$))");
	std::regex regDeclRegex(
		R"(^\s*\.reg\s+(\.\w+)\s+([%a-zA-Z_][a-zA-Z0-9_]*)(?:<(\d+)>)?\s*;)");
	std::regex localDeclRegex(R"(^\s*\.local\s+)");
	std::regex commentOrEmptyRegex(R"(^\s*(//.*|\s*$))");
	std::regex openingBraceRegex(R"(^\s*\{\s*$)");
	std::regex closingBraceRegex(R"(^\s*\}\s*$)");
	std::regex retRegex(R"(^\s*(?:@%p\d+\s+)?ret\s*;)");

	std::vector<std::string> currentFunctionLines;
	bool inFunctionDefinition = false;
	bool inFunctionBody = false;

	const std::string savedRegOffsetBaseReg = "%rd_saver_base";
	const std::string savedRegTempAddrReg = "%rd_saver_addr";

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
				std::vector<RegisterInfo> registersInFunc;
				size_t lastDeclarationLineIndex =
					0; // Index of the line *after* all
					   // decls (.reg, .local)
				bool foundOpeningBraceForBody = false;

				for (size_t i = 0;
				     i < currentFunctionLines.size(); ++i) {
					const std::string &funcLine =
						currentFunctionLines[i];
					std::smatch regMatch;

					if (std::regex_search(
						    funcLine,
						    openingBraceRegex)) {
						foundOpeningBraceForBody = true;
						lastDeclarationLineIndex =
							i + 1; // Potential
							       // start after
							       // this line
						continue; // Process next line
					}

					if (!foundOpeningBraceForBody)
						continue; // Skip lines before
							  // '{' in the
							  // collected function
							  // lines

					if (std::regex_search(funcLine,
							      regMatch,
							      regDeclRegex)) {
						lastDeclarationLineIndex =
							i + 1; // Next line
							       // could be an
							       // instruction
						RegisterInfo regInfo;
						regInfo.type =
							regMatch[1].str();
						regInfo.baseName =
							regMatch[2].str();
						regInfo.sizeInBytes =
							getRegisterSizeInBytes(
								regInfo.type);
						regInfo.ptxTypeModifier =
							getPtxTypeModifier(
								regInfo.type,
								regInfo.sizeInBytes);

						if (regInfo.baseName == "%SP" ||
						    regInfo.baseName ==
							    "%SPL" ||
						    regInfo.sizeInBytes == 0 ||
						    regInfo.baseName ==
							    savedRegOffsetBaseReg ||
						    regInfo.baseName ==
							    savedRegTempAddrReg) {
							continue;
						}

						if (regMatch[3].matched) {
							regInfo.count = std::stoi(
								regMatch[3]
									.str());
							for (int k = 0;
							     k < regInfo.count;
							     ++k) {
								regInfo.names.push_back(
									regInfo.baseName
										.substr(0,
											regInfo.baseName
												.length()) +
									std::to_string(
										k));
							}
						} else {
							regInfo.count = 1;
							regInfo.names.push_back(
								regInfo.baseName);
						}
						registersInFunc.push_back(
							regInfo);
					} else if (std::regex_search(
							   funcLine,
							   localDeclRegex)) {
						lastDeclarationLineIndex =
							i + 1; // Next line
							       // could be an
							       // instruction
					} else if (!std::regex_search(
							   funcLine,
							   commentOrEmptyRegex)) {
						// This is likely the first
						// instruction if no earlier
						// instruction was found
						// However,
						// lastDeclarationLineIndex
						// should point *after* all
						// decls. The logic to find the
						// *actual* first instruction
						// for push code insertion will
						// happen when rebuilding the
						// function string.
					}
				}
				// Ensure lastDeclarationLineIndex is at least
				// after the opening brace if no other decls
				// found
				if (foundOpeningBraceForBody &&
				    lastDeclarationLineIndex == 0 &&
				    currentFunctionLines.size() > 0) {
					for (size_t i = 0;
					     i < currentFunctionLines.size();
					     ++i) {
						if (std::regex_search(
							    currentFunctionLines
								    [i],
							    openingBraceRegex)) {
							lastDeclarationLineIndex =
								i + 1;
							break;
						}
					}
				}

				std::string pushCodeBlock;
				std::string popCodeBlock;
				std::string utilityRegDeclsForFunc;
				int totalPushSizeBytes = 0;

				if (!registersInFunc.empty()) {
					utilityRegDeclsForFunc =
						"\t.reg .b64 " +
						savedRegOffsetBaseReg + ";\n";
					utilityRegDeclsForFunc +=
						"\t.reg .b64 " +
						savedRegTempAddrReg + ";\n";

					std::stringstream pushSs, popSs;
					std::vector<std::pair<std::string, int>>
						regSaveOrderWithOffsets;
					int currentOffsetForCalc = 0;

					for (const auto &regInfo :
					     registersInFunc) {
						for (const auto &regName :
						     regInfo.names) {
							regSaveOrderWithOffsets.push_back(
								{ regName,
								  currentOffsetForCalc });
							currentOffsetForCalc +=
								regInfo.sizeInBytes;
							if (regInfo.sizeInBytes <
								    8 &&
							    currentOffsetForCalc %
									    regInfo.sizeInBytes !=
								    0) { // Basic
									 // alignment
									 // for
									 // smaller
									 // types
									 // if
									 // needed
								if (regInfo.sizeInBytes ==
									    4 &&
								    currentOffsetForCalc %
										    4 !=
									    0)
									currentOffsetForCalc =
										(currentOffsetForCalc +
										 3) &
										~3;
								else if (
									regInfo.sizeInBytes ==
										2 &&
									currentOffsetForCalc %
											2 !=
										0)
									currentOffsetForCalc =
										(currentOffsetForCalc +
										 1) &
										~1;
							} else if (
								regInfo.sizeInBytes ==
									8 &&
								currentOffsetForCalc %
										8 !=
									0) { // Ensure 8-byte align for 64-bit
								currentOffsetForCalc =
									(currentOffsetForCalc +
									 7) &
									~7;
							}
						}
					}
					totalPushSizeBytes =
						currentOffsetForCalc;
					if (totalPushSizeBytes > 0 &&
					    totalPushSizeBytes % 8 !=
						    0) { // Align total space to
							 // 8 bytes
						totalPushSizeBytes =
							(totalPushSizeBytes +
							 7) &
							~7;
					}

					if (totalPushSizeBytes > 0) {
						pushSs << "\n\t// --- BEGIN REGISTER SAVING (PUSH) ---\n";
						pushSs << "\tsub.u64 %SP, %SP, "
						       << totalPushSizeBytes
						       << "; // Allocate stack space\n";
						pushSs << "\tmov.u64 "
						       << savedRegOffsetBaseReg
						       << ", %SP; // Base for saving\n";
						for (const auto &regSaveEntry :
						     regSaveOrderWithOffsets) {
							const std::string &regName =
								regSaveEntry
									.first;
							int regOffset =
								regSaveEntry
									.second;
							const RegisterInfo
								*pRegInfo =
									nullptr;
							for (const auto &ri :
							     registersInFunc) {
								if (std::find(
									    ri.names.begin(),
									    ri.names.end(),
									    regName) !=
								    ri.names.end()) {
									pRegInfo =
										&ri;
									break;
								}
							}
							if (pRegInfo) {
								pushSs << "\tadd.u64 "
								       << savedRegTempAddrReg
								       << ", "
								       << savedRegOffsetBaseReg
								       << ", "
								       << regOffset
								       << ";\n";
								pushSs << "\tst.local"
								       << pRegInfo->ptxTypeModifier
								       << " ["
								       << savedRegTempAddrReg
								       << "], "
								       << regName
								       << ";\n";
							}
						}
						pushSs << "\t// --- END REGISTER SAVING (PUSH) ---\n";
						pushCodeBlock = pushSs.str();

						popSs << "\n\t// --- BEGIN REGISTER RESTORING (POP) ---\n";
						popSs << "\tmov.u64 "
						      << savedRegOffsetBaseReg
						      << ", %SP; // Base for restoring (before SP adjustment)\n";
						for (auto rit =
							     regSaveOrderWithOffsets
								     .rbegin();
						     rit !=
						     regSaveOrderWithOffsets
							     .rend();
						     ++rit) {
							const std::string &regName =
								rit->first;
							int regOffset =
								rit->second;
							const RegisterInfo
								*pRegInfo =
									nullptr;
							for (const auto &ri :
							     registersInFunc) {
								if (std::find(
									    ri.names.begin(),
									    ri.names.end(),
									    regName) !=
								    ri.names.end()) {
									pRegInfo =
										&ri;
									break;
								}
							}
							if (pRegInfo) {
								popSs << "\tadd.u64 "
								      << savedRegTempAddrReg
								      << ", "
								      << savedRegOffsetBaseReg
								      << ", "
								      << regOffset
								      << ";\n";
								popSs << "\tld.local"
								      << pRegInfo->ptxTypeModifier
								      << " "
								      << regName
								      << ", ["
								      << savedRegTempAddrReg
								      << "];\n";
							}
						}
						popSs << "\tadd.u64 %SP, %SP, "
						      << totalPushSizeBytes
						      << "; // Deallocate stack space\n";
						popSs << "\t// --- END REGISTER RESTORING (POP) ---\n";
						popCodeBlock = popSs.str();
					}
				}

				// Rebuild the function with instrumentation
				std::vector<std::string> instrumentedLines;
				bool utilityDeclsPlaced = false;
				bool pushBlockPlaced = false;
				size_t actualFirstInstructionLineIndex =
					std::string::npos;

				// First pass to find the actual first
				// instruction line after all declarations
				// (including our utility register declarations
				// if they were hypothetically placed)
				size_t tempLastDeclLineIndex = 0;
				bool tempFoundOpeningBrace = false;
				for (size_t i = 0;
				     i < currentFunctionLines.size(); ++i) {
					const auto &fLine =
						currentFunctionLines[i];
					if (std::regex_search(
						    fLine, openingBraceRegex)) {
						tempFoundOpeningBrace = true;
						tempLastDeclLineIndex =
							i + 1; // Declarations
							       // can start
							       // after this
						continue;
					}
					if (!tempFoundOpeningBrace)
						continue;

					if (std::regex_search(fLine,
							      regDeclRegex) ||
					    std::regex_search(fLine,
							      localDeclRegex)) {
						tempLastDeclLineIndex = i + 1;
					} else if (!std::regex_search(
							   fLine,
							   commentOrEmptyRegex)) {
						actualFirstInstructionLineIndex =
							i;
						break;
					}
				}
				if (actualFirstInstructionLineIndex ==
					    std::string::npos &&
				    tempLastDeclLineIndex <
					    currentFunctionLines.size()) {
					actualFirstInstructionLineIndex =
						tempLastDeclLineIndex; // If
								       // only
								       // decls,
								       // push
								       // after
								       // them
				} else if (actualFirstInstructionLineIndex ==
						   std::string::npos &&
					   !currentFunctionLines.empty()) {
					// Fallback if completely empty or only
					// comments after brace
					for (size_t i = 0;
					     i < currentFunctionLines.size();
					     ++i) {
						if (std::regex_search(
							    currentFunctionLines
								    [i],
							    openingBraceRegex)) {
							actualFirstInstructionLineIndex =
								i + 1;
							break;
						}
					}
					if (actualFirstInstructionLineIndex ==
					    std::string::npos)
						actualFirstInstructionLineIndex =
							currentFunctionLines
								.size(); // Append
									 // if
									 // no
									 // brace
				}

				for (size_t i = 0;
				     i < currentFunctionLines.size(); ++i) {
					// Insert utility register declarations
					// after all original .reg and .local
					if (!utilityRegDeclsForFunc.empty() &&
					    !utilityDeclsPlaced &&
					    i == lastDeclarationLineIndex) {
						instrumentedLines.push_back(
							utilityRegDeclsForFunc);
						utilityDeclsPlaced = true;
					}
					// Insert push block after all
					// declarations (original + utility)
					if (!pushCodeBlock.empty() &&
					    !pushBlockPlaced &&
					    i == actualFirstInstructionLineIndex) {
						// If utilityDecls were meant to
						// be placed here but haven't
						// been, place them first
						if (!utilityRegDeclsForFunc
							     .empty() &&
						    !utilityDeclsPlaced &&
						    lastDeclarationLineIndex ==
							    actualFirstInstructionLineIndex) {
							instrumentedLines.push_back(
								utilityRegDeclsForFunc);
							utilityDeclsPlaced =
								true;
						}
						instrumentedLines.push_back(
							pushCodeBlock);
						pushBlockPlaced = true;
					}

					std::smatch retMatch;
					if (!popCodeBlock.empty() &&
					    std::regex_search(
						    currentFunctionLines[i],
						    retMatch, retRegex)) {
						instrumentedLines.push_back(
							popCodeBlock);
					}
					instrumentedLines.push_back(
						currentFunctionLines[i]);
				}

				// If declarations were empty, utility decls and
				// push block might not have been inserted.
				if (!utilityRegDeclsForFunc.empty() &&
				    !utilityDeclsPlaced) {
					size_t insertAt = 0;
					for (size_t i = 0;
					     i < instrumentedLines.size();
					     ++i) {
						if (std::regex_search(
							    instrumentedLines[i],
							    openingBraceRegex)) {
							insertAt = i + 1;
							break;
						}
						if (std::regex_search(
							    instrumentedLines[i],
							    funcDefRegex) &&
						    i == 0) { // if .func is
							      // first line
							insertAt = i + 1;
						}
					}
					if (insertAt < instrumentedLines.size())
						instrumentedLines.insert(
							instrumentedLines.begin() +
								insertAt,
							utilityRegDeclsForFunc);
					else
						instrumentedLines.push_back(
							utilityRegDeclsForFunc);
					utilityDeclsPlaced = true; // Mark as
								   // placed
				}
				if (!pushCodeBlock.empty() &&
				    !pushBlockPlaced) {
					size_t insertAt = 0;
					for (size_t i = 0;
					     i < instrumentedLines.size();
					     ++i) { // Find where utility decls
						    // ended or '{'
						if (instrumentedLines[i] ==
							    utilityRegDeclsForFunc ||
						    (!utilityRegDeclsForFunc
							      .empty() &&
						     instrumentedLines[i].find(
							     savedRegTempAddrReg) !=
							     std::string::
								     npos)) { // after utility decls
							insertAt = i + 1;
							break;
						} else if (
							utilityRegDeclsForFunc
								.empty() &&
							std::regex_search(
								instrumentedLines
									[i],
								openingBraceRegex)) {
							insertAt = i + 1;
							break;
						} else if (utilityRegDeclsForFunc
								   .empty() &&
							   i == 0 &&
							   std::regex_search(
								   instrumentedLines
									   [0],
								   funcDefRegex)) {
							insertAt = i + 1;
						}
					}
					if (insertAt == 0 &&
					    instrumentedLines.size() > 0 &&
					    std::regex_search(
						    instrumentedLines[0],
						    funcDefRegex))
						insertAt = 1;

					// If still not found, find 'ret' or '}'
					if (insertAt >=
						    instrumentedLines.size() ||
					    insertAt == 0 &&
						    !std::regex_search(
							    instrumentedLines[0],
							    funcDefRegex)) {
						bool retFound = false;
						for (size_t i = 0;
						     i <
						     instrumentedLines.size();
						     ++i) {
							if (std::regex_search(
								    instrumentedLines
									    [i],
								    retRegex)) {
								insertAt = i;
								retFound = true;
								break;
							}
						}
						if (!retFound) {
							for (size_t i = 0;
							     i <
							     instrumentedLines
								     .size();
							     ++i) {
								if (std::regex_search(
									    instrumentedLines
										    [i],
									    closingBraceRegex)) {
									insertAt =
										i;
									break;
								}
							}
						}
						if (insertAt >=
						    instrumentedLines.size())
							insertAt =
								instrumentedLines
									.size(); // append if truly empty
					}

					if (insertAt < instrumentedLines.size())
						instrumentedLines.insert(
							instrumentedLines.begin() +
								insertAt,
							pushCodeBlock);
					else
						instrumentedLines.push_back(
							pushCodeBlock);
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
