#include <string>
#include <vector>
#include <regex>
namespace bpftime::attach
{
// Done by Gemini


// (RegisterInfo struct and getRegisterSizeInBytes, getPtxTypeModifier functions remain the same)
struct RegisterInfo {
    std::string type;
    std::string baseName;
    int count;
    std::vector<std::string> names;
    int sizeInBytes;
    std::string ptxTypeModifier; // For st.local/ld.local e.g. .u16, .u64
};

int getRegisterSizeInBytes(const std::string& type) {
    if (type == ".b8" || type == ".s8" || type == ".u8") return 1;
    if (type == ".b16" || type == ".s16" || type == ".u16" || type == ".f16") return 2;
    if (type == ".b32" || type == ".s32" || type == ".u32" || type == ".f32") return 4;
    if (type == ".b64" || type == ".s64" || type == ".u64" || type == ".f64" || type == ".f64x2") return 8;
    if (type == ".pred") return 1;
    return 0;
}

// Get PTX type modifier for st/ld, ensuring we handle predicates and smaller types correctly
// when storing them in 8-byte aligned slots if needed.
// For your reference, st.local will store the register's value.
// We'll use the register's actual type for st/ld, but ensure offsets are 8-byte aligned.
std::string getPtxStorageTypeModifier(const RegisterInfo& regInfo) {
    if (regInfo.type == ".pred") return ".b8"; // Store predicate as a byte
    // For direct storage of the register's type
    if (regInfo.sizeInBytes == 1) return ".u8";
    if (regInfo.sizeInBytes == 2) return ".u16";
    if (regInfo.sizeInBytes == 4) return ".u32";
    if (regInfo.sizeInBytes == 8) return ".u64";
    return regInfo.type; // Fallback, should be specific
}


std::string add_register_guard_for_ebpf_ptx_func(const std::string& ptxCode) {
    std::string resultPtx;
    std::stringstream ptxStream(ptxCode);
    std::string line;

    std::regex funcDefRegex(R"(^\s*(?:\.visible\s+)?(?:\.func|\.entry)\s+([a-zA-Z_][a-zA-Z0-9_@$.]*)(?:\s*\(|\s*$))");
    std::regex regDeclRegex(R"(^\s*\.reg\s+(\.\w+)\s+([%a-zA-Z_][a-zA-Z0-9_]*)(?:<(\d+)>)?\s*;)");
    std::regex localDeclRegex(R"(^\s*\.local\s+)");
    std::regex commentOrEmptyRegex(R"(^\s*(//.*|\s*$))");
    std::regex openingBraceRegex(R"(^\s*\{\s*$)");
    std::regex closingBraceRegex(R"(^\s*\}\s*$)");
    std::regex retRegex(R"(^\s*(?:@%p\d+\s+)?ret\s*;)");

    std::vector<std::string> currentFunctionLines;
    bool inFunctionDefinition = false;
    bool inFunctionBody = false;

    const std::string tempBaseReg = "%rd_ptx_instr_base"; // For SPL copy
    const std::string tempAddrReg = "%rd_ptx_instr_addr";   // For address calculation
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
                size_t lastOriginalRegDeclLineIndex = 0;
                bool funcHasOpeningBrace = false;
                size_t funcOpeningBraceIndex = 0;
                bool originalRegsFound = false;

                for (size_t i = 0; i < currentFunctionLines.size(); ++i) {
                    const std::string& funcLine = currentFunctionLines[i];
                    std::smatch regMatch;
                    if (std::regex_search(funcLine, openingBraceRegex)) {
                        funcHasOpeningBrace = true;
                        funcOpeningBraceIndex = i;
                        lastOriginalRegDeclLineIndex = i +1; // Declarations usually after {
                        continue;
                    }
                    if (!funcHasOpeningBrace && !std::regex_search(currentFunctionLines[0], funcDefRegex)) continue;

                    if (std::regex_search(funcLine, regMatch, regDeclRegex)) {
                        originalRegsFound = true;
                        lastOriginalRegDeclLineIndex = i; // The line itself is the last reg decl
                        RegisterInfo regInfo;
                        regInfo.type = regMatch[1].str();
                        regInfo.baseName = regMatch[2].str();
                        regInfo.sizeInBytes = getRegisterSizeInBytes(regInfo.type);
                        regInfo.ptxTypeModifier = getPtxStorageTypeModifier(regInfo); // Use storage type

                        if (regInfo.baseName == "%SP" || regInfo.baseName == "%SPL" || regInfo.sizeInBytes == 0 ||
                            regInfo.baseName == tempBaseReg || regInfo.baseName == tempAddrReg) {
                            continue;
                        }
                        bool name_conflict = false; // Basic check against our temp regs
                        for(const auto& r : registersToSaveInFunc) { if (r.baseName == regInfo.baseName) { name_conflict = true; break; } }
                        if (name_conflict) continue;

                        if (regMatch[3].matched) {
                            regInfo.count = std::stoi(regMatch[3].str());
                            for (int k = 0; k < regInfo.count; ++k) {
                                regInfo.names.push_back(regInfo.baseName.substr(0, regInfo.baseName.length()) + std::to_string(k));
                            }
                        } else {
                            regInfo.count = 1;
                            regInfo.names.push_back(regInfo.baseName);
                        }
                        registersToSaveInFunc.push_back(regInfo);
                    } else if (std::regex_search(funcLine, localDeclRegex)) {
                        // Ensure our new .local decl is after all original .local and .reg
                        lastOriginalRegDeclLineIndex = i;
                    }
                }
                // Adjust lastOriginalDeclLineIndex to be *after* the line
                if (originalRegsFound || funcHasOpeningBrace) {
                    lastOriginalRegDeclLineIndex++;
                } else if (!currentFunctionLines.empty() && std::regex_search(currentFunctionLines[0], funcDefRegex)) {
                    lastOriginalRegDeclLineIndex = 1; // After .func line if no { or .reg
                }


                std::string pushCodeBlockStr;
                std::string popCodeBlockStr;
                std::string newLocalAndUtilRegDecls;
                int totalBytesForSavedRegs = 0;
                std::vector<std::pair<std::string, int>> regOffsetMap; // name, offset in save area

                if (!registersToSaveInFunc.empty()) {
                    int currentStackOffset = 0;
                    for (const auto& regInfo : registersToSaveInFunc) {
                        for (const auto& regName : regInfo.names) {
                            regOffsetMap.push_back({regName, currentStackOffset});
                            currentStackOffset += 8; // Each register slot is 8 bytes as per example
                        }
                    }
                    totalBytesForSavedRegs = currentStackOffset;
                    // No need to further align totalBytesForSavedRegs if each slot is 8-byte aligned

                    // Declare the new local memory area and utility registers
                    // Ensure this size is sufficient and reasonable.
                    int saveAreaSize = std::max(totalBytesForSavedRegs, 8); // Min size 8 for alignment
                    if (saveAreaSize % 8 != 0) saveAreaSize = (saveAreaSize + 7) & ~7;

                    newLocalAndUtilRegDecls = "\t.local .align 8 .b8 \t" + registerSaveAreaName + "[" + std::to_string(saveAreaSize) + "];\n";
                    newLocalAndUtilRegDecls += "\t.reg .b64 " + tempBaseReg + ";\n";
                    newLocalAndUtilRegDecls += "\t.reg .b64 " + tempAddrReg + ";\n";


                    if (totalBytesForSavedRegs > 0) {
                        std::stringstream pushSs, popSs;
                        pushSs << "\t// --- BEGIN REGISTER SAVING (PUSH to " << registerSaveAreaName << ") ---\n";
                        pushSs << "\tmov.u64 " << tempBaseReg << ", " << registerSaveAreaName << "; // Use tempBaseReg for the save area's SPL\n";

                        for (const auto& entry : regOffsetMap) {
                            const std::string& regName = entry.first;
                            int offset = entry.second;
                            const RegisterInfo* pRegInfo = nullptr;
                            for(const auto& ri : registersToSaveInFunc) { if (std::find(ri.names.begin(), ri.names.end(), regName) != ri.names.end()) { pRegInfo = &ri; break; } }
                            if (pRegInfo) {
                                pushSs << "\tadd.u64 " << tempAddrReg << ", " << tempBaseReg << ", " << offset << ";\n";
                                // Store using the register's actual type, but into an 8-byte aligned slot
                                pushSs << "\tst.local" << pRegInfo->ptxTypeModifier << " [" << tempAddrReg << "], " << regName << ";\n";
                            }
                        }
                        pushSs << "\t// --- END REGISTER SAVING (PUSH to " << registerSaveAreaName << ") ---\n";
                        pushCodeBlockStr = pushSs.str();

                        popSs << "\n\t// --- BEGIN REGISTER RESTORING (POP from " << registerSaveAreaName << ") ---\n";
                        popSs << "\tmov.u64 " << tempBaseReg << ", " << registerSaveAreaName << "; // Base for restoring\n";
                        for (auto rit = regOffsetMap.rbegin(); rit != regOffsetMap.rend(); ++rit) {
                            const std::string& regName = rit->first;
                            int offset = rit->second;
                            const RegisterInfo* pRegInfo = nullptr;
                            for(const auto& ri : registersToSaveInFunc) { if (std::find(ri.names.begin(), ri.names.end(), regName) != ri.names.end()) { pRegInfo = &ri; break; } }
                            if (pRegInfo) {
                                popSs << "\tadd.u64 " << tempAddrReg << ", " << tempBaseReg << ", " << offset << ";\n";
                                popSs << "\tld.local" << pRegInfo->ptxTypeModifier << " " << regName << ", [" << tempAddrReg << "];\n";
                            }
                        }
                        popSs << "\t// --- END REGISTER RESTORING (POP from " << registerSaveAreaName << ") ---\n";
                        popCodeBlockStr = popSs.str();
                    }
                }

                // Reconstruct the function
                std::vector<std::string> instrumentedLines;
                bool newDeclsInserted = false;
                bool pushCodeInserted = false;

                // Find insertion point for new declarations (after all original .reg and .local)
                // And for push code (after all declarations including new ones)
                size_t newDeclsInsertIdx = lastOriginalRegDeclLineIndex;
                size_t pushBlockInsertIdxAfterNewDecls = newDeclsInsertIdx;
                if (!newLocalAndUtilRegDecls.empty()) {
                     std::stringstream tempSS(newLocalAndUtilRegDecls); std::string tempL; int countL = 0;
                     while(std::getline(tempSS, tempL)) if(!tempL.empty()) countL++;
                     pushBlockInsertIdxAfterNewDecls += countL;
                }
                // Find the actual first instruction after all potential declarations
                for(size_t k = pushBlockInsertIdxAfterNewDecls; k < currentFunctionLines.size(); ++k) {
                    if (!std::regex_search(currentFunctionLines[k], commentOrEmptyRegex) &&
                        !std::regex_search(currentFunctionLines[k], openingBraceRegex) && // Should be past these
                        !std::regex_search(currentFunctionLines[k], regDeclRegex) &&
                        !std::regex_search(currentFunctionLines[k], localDeclRegex) ){
                        pushBlockInsertIdxAfterNewDecls = k;
                        break;
                    }
                     if (k == currentFunctionLines.size() -1 && pushBlockInsertIdxAfterNewDecls < currentFunctionLines.size()) { // If only comments/empty lines remain
                         pushBlockInsertIdxAfterNewDecls = k + 1;
                     } else if (k == currentFunctionLines.size() -1) { // if pushBlockInsertIdxAfterNewDecls was already at the end
                         pushBlockInsertIdxAfterNewDecls = currentFunctionLines.size();
                     }
                }
                 if (pushBlockInsertIdxAfterNewDecls >= currentFunctionLines.size() && !currentFunctionLines.empty() && std::regex_search(currentFunctionLines.back(), closingBraceRegex)) {
                    pushBlockInsertIdxAfterNewDecls = currentFunctionLines.size() -1;
                }


                for (size_t i = 0; i < currentFunctionLines.size(); ++i) {
                    if (i == newDeclsInsertIdx && !newLocalAndUtilRegDecls.empty() && !newDeclsInserted) {
                        instrumentedLines.push_back(newLocalAndUtilRegDecls);
                        newDeclsInserted = true;
                    }
                    if (i == pushBlockInsertIdxAfterNewDecls && !pushCodeBlockStr.empty() && !pushCodeInserted) {
                        // Ensure utility decls are placed if they were meant for this spot
                         if (!newLocalAndUtilRegDecls.empty() && !newDeclsInserted && newDeclsInsertIdx == pushBlockInsertIdxAfterNewDecls) {
                            instrumentedLines.push_back(newLocalAndUtilRegDecls);
                            newDeclsInserted = true;
                        }
                        instrumentedLines.push_back(pushCodeBlockStr);
                        pushCodeInserted = true;
                    }

                    std::smatch retMatch;
                    if (!popCodeBlockStr.empty() && std::regex_search(currentFunctionLines[i], retMatch, retRegex)) {
                        instrumentedLines.push_back(popCodeBlockStr);
                    }
                    instrumentedLines.push_back(currentFunctionLines[i]);
                }
                 // Fallback insertions if primary logic missed (e.g. empty function body)
                if (!newLocalAndUtilRegDecls.empty() && !newDeclsInserted) {
                    size_t targetIdx = (funcHasOpeningBrace ? funcOpeningBraceIndex + 1 : (currentFunctionLines.empty() ? 0 : (std::regex_search(currentFunctionLines[0], funcDefRegex) ? 1 : 0) ));
                    if (targetIdx > instrumentedLines.size()) targetIdx = instrumentedLines.size();
                    instrumentedLines.insert(instrumentedLines.begin() + targetIdx, newLocalAndUtilRegDecls);
                    newDeclsInserted = true;
                }
                if (!pushCodeBlockStr.empty() && !pushCodeInserted) {
                    size_t targetIdx = 0; bool found = false;
                    for(size_t i=0; i < instrumentedLines.size(); ++i) {
                        if (instrumentedLines[i] == newLocalAndUtilRegDecls ||
                           (newLocalAndUtilRegDecls.empty() && std::regex_search(instrumentedLines[i], openingBraceRegex)) ||
                           (newLocalAndUtilRegDecls.empty() && !funcHasOpeningBrace && i==0 && std::regex_search(instrumentedLines[0], funcDefRegex))) {
                            targetIdx = i + 1; found = true; break;
                        }
                    }
                    if (!found) { // Fallback: before ret or '}'
                         for(size_t i = 0; i < instrumentedLines.size(); ++i) {
                           if(std::regex_search(instrumentedLines[i], retRegex) || std::regex_search(instrumentedLines[i], closingBraceRegex)) { targetIdx = i; found = true; break; }
                        }
                    }
                    if (!found) targetIdx = instrumentedLines.size();
                    if (targetIdx > instrumentedLines.size()) targetIdx = instrumentedLines.size();
                    instrumentedLines.insert(instrumentedLines.begin() + targetIdx, pushCodeBlockStr);
                }


                for(const auto& instrLine : instrumentedLines) {
                     resultPtx += instrLine + (instrLine.empty() || instrLine.back() == '\n' ? "" : "\n");
                }
                currentFunctionLines.clear();
                inFunctionBody = false;
            }
        }
    }
    if (inFunctionDefinition || inFunctionBody) {
        for (const auto& l : currentFunctionLines) {
            resultPtx += l + "\n";
        }
    }
    return resultPtx;
}
} // namespace bpftime::attach
