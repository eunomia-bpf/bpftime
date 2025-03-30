use anyhow;
use std::{collections::HashMap, vec};

use crate::parser::{Block, CallInst, Global, Ptx, Stmt, StoreInst};

#[allow(unused)]
macro_rules! strings {
    ($($s:expr), *) => {
        vec![$($s.to_string()), *]
    };
}

fn get_store_patch(s: &StoreInst) -> Option<String> {
    // could be global store ?
    if s.sspace.unwrap_or(".global") != ".global" {
        return None;
    }
    let (neg, pred) = s.pred.unwrap_or((false, "pt"));
    // invert neg
    let neg = if neg { "" } else { "!" };
    Some(format!(include_str!("access_check.ptx"), pred, s.addr, neg))
}

fn get_call_patch(
    c: &CallInst,
    patched_funcs: &HashMap<String, FuncMeta>,
) -> Option<(&'static str, Vec<&'static str>)> {
    assert!(
        c.call_targets.is_none(),
        "indirect call detected in {}",
        c.func
    );
    assert!(c.func.contains(c.func), "unknown func {}", c.func);
    let f = patched_funcs.get(c.func).unwrap();
    // skip extern func call
    if f.external {
        return None;
    }
    Some((
        include_str!("call_param.ptx"),
        vec![
            "call_range_min_param",
            "call_range_max_param",
            "call_range_cnt_param",
            "call_check_passed_param",
        ],
    ))
}

fn patch_block<'a>(
    block: &mut Block<'a>,
    patched_funcs: &HashMap<String, FuncMeta>,
) -> anyhow::Result<()> {
    let mut patches = Vec::<(usize, String)>::new();
    for (i, s) in block.stmts.iter_mut().enumerate() {
        match s {
            Stmt::Store(s) => {
                if let Some(patch) = get_store_patch(s) {
                    patches.push((i, patch));
                }
            }
            Stmt::Call(c) => {
                if let Some((patch, extra_params)) = get_call_patch(c, patched_funcs) {
                    patches.push((i, patch.to_string()));
                    c.params = Some(
                        c.params
                            .clone()
                            .unwrap_or(vec![])
                            .into_iter()
                            .chain(extra_params)
                            .collect(),
                    );
                }
            }
            Stmt::Blk(b) => {
                patch_block(b, patched_funcs)?;
            }
            _ => (),
        }
    }
    for (i, patch) in patches.into_iter().rev() {
        block.stmts.insert(i, Stmt::Patch(patch));
    }
    Ok(())
}

struct FuncMeta {
    external: bool,
}

pub fn patch_ptx<'a>(ptx: &mut Ptx<'a>) -> anyhow::Result<()> {
    let mut patched_funcs = HashMap::<String, FuncMeta>::new();
    for g in &mut ptx.globals.iter_mut() {
        match g {
            Global::Func(f) => {
                // maintain a symbol table
                patched_funcs.insert(
                    f.name.to_string(),
                    FuncMeta {
                        external: f.linkage.external,
                    },
                );

                // do not patch extern func decl
                if f.linkage.external {
                    assert!(f.body.is_none(), "extern func must be decl");
                    continue;
                }

                // patch the function
                // add extra params
                f.params.append(
                    vec![
                        ".param .u64 range_min_param[64]",
                        ".param .u64 range_max_param[64]",
                        ".param .u32 range_cnt_param",
                        ".param .u64 check_passed_param",
                    ]
                    .as_mut(),
                );

                // patch global store inst / call inst in the body block recursively
                if let Some(body) = &mut f.body {
                    patch_block(body, &patched_funcs)?;
                }
            }
            _ => (),
        }
    }
    Ok(())
}
