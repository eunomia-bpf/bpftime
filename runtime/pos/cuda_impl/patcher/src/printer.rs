use anyhow;
use std::{borrow::Borrow, io::Write};

use crate::parser::{self, Block, Global, Linkage, Stmt};

fn print_option<T: Write>(set: bool, option: &str, writer: &mut T) -> anyhow::Result<()> {
    if set {
        write!(writer, "{}", option)?;
    }
    Ok(())
}

fn print_parenthesized<T: Write>(s: &str, writer: &mut T) -> anyhow::Result<()> {
    write!(writer, "({})", s)?;
    Ok(())
}

fn print_pred<T: Write>(pred: &Option<(bool, &str)>, writer: &mut T) -> anyhow::Result<()> {
    if let Some((neg, pred)) = pred {
        write!(writer, "@")?;
        print_option(*neg, "!", writer)?;
        write!(writer, "{} ", pred)?;
    }
    Ok(())
}

fn print_linkage<T: Write>(linkage: &Linkage, writer: &mut T) -> anyhow::Result<()> {
    print_option(linkage.external, ".extern ", writer)?;
    print_option(linkage.visible, ".visible ", writer)?;
    print_option(linkage.weak, ".weak ", writer)?;
    Ok(())
}

fn print_block<T: Write>(block: &Block, writer: &mut T) -> anyhow::Result<()> {
    write!(writer, "{{\n")?;
    for i in &block.stmts {
        match i {
            Stmt::Label(l) => {
                write!(writer, "{}:\n", l.label)?;
            }
            Stmt::Inst(i) => {
                print_pred(&i.pred, writer)?;
                write!(writer, "{} {};\n", i.op, i.args)?;
            }
            Stmt::Blk(b) => {
                print_block(b, writer)?;
            }
            Stmt::Patch(i) => {
                write!(writer, "{}\n", i)?;
            }
            Stmt::Store(s) => {
                print_pred(&s.pred, writer)?;
                write!(writer, "{} [{}], {};\n", s.op, s.addr, s.args)?;
            }
            Stmt::Call(c) => {
                print_pred(&c.pred, writer)?;
                write!(writer, "{} ", c.op)?;
                if let Some(retval) = c.retval {
                    print_parenthesized(retval, writer)?;
                    write!(writer, ", ")?;
                }
                write!(writer, "{}", c.func)?;
                if let Some(params) = c.params.borrow() {
                    write!(writer, ", ")?;
                    print_parenthesized(&params.join(", "), writer)?;
                }
                if let Some(call_targets) = c.call_targets {
                    write!(writer, ", {}", call_targets)?;
                }
                write!(writer, ";\n")?;
            }
        }
    }
    write!(writer, "}}\n")?;
    Ok(())
}

pub fn print_ptx<T: Write>(ptx: &parser::Ptx, writer: &mut T) -> anyhow::Result<()> {
    // FIXME: a quirk to work around cuda driver jit issue
    // jit would fail without adding the leading empty line
    write!(writer, "\n")?;

    for g in &ptx.globals {
        match g {
            Global::Func(f) => {
                print_linkage(&f.linkage, writer)?;
                if f.entry {
                    write!(writer, ".entry ")?;
                } else {
                    write!(writer, ".func ")?;
                }
                if let Some(retval) = f.retval {
                    print_parenthesized(retval, writer)?;
                }
                write!(writer, "{}", f.name)?;
                print_parenthesized(&f.params.join(", "), writer)?;
                write!(
                    writer,
                    "{}",
                    f.directives
                        .iter()
                        .map(|d| format!("{} {}", d.directive, d.args))
                        .fold(String::from("\n"), |acc, x| acc + &x + "\n")
                )?;
                if let Some(body) = &f.body {
                    print_block(body, writer)?;
                } else {
                    write!(writer, ";\n")?;
                }
            }
            Global::Dir(d) => {
                write!(writer, "{} {}\n", d.directive, d.args)?;
            }
            Global::Var(v) => {
                print_linkage(&v.linkage, writer)?;
                write!(writer, "{} {};\n", v.var_type, v.args)?;
            }
        }
    }
    Ok(())
}
