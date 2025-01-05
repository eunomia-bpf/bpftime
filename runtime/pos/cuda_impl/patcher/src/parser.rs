#![allow(unused)]

use std::{fs, io::Read};

use nom::{
    branch::alt,
    bytes::complete::{tag, take_till1, take_until, take_until1, take_while1},
    character::complete::{char, multispace1},
    combinator::{eof, map, opt, peek, value},
    error::{Error, ErrorKind},
    multi::{many0, many1, separated_list0},
    sequence::{delimited, pair, preceded, terminated, tuple},
    IResult, Parser,
};

// FIXME: consider comments
fn parse_parenthesized(input: &str) -> IResult<&str, &str> {
    delimited(char('('), take_until(")"), char(')'))(input)
}

fn parse_braced_balanced(input: &str) -> IResult<&str, &str> {
    let (input, _) = char('{')(input)?;
    let mut depth = 1;
    let mut end = None;
    for (i, c) in input.chars().enumerate() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(i);
                    break;
                }
            }
            _ => (),
        }
    }
    match end {
        Some(end) => return Ok((&input[end + 1..], &input[0..end])),
        None => return Err(nom::Err::Error(Error::new(input, ErrorKind::Char))),
    }
}

fn comment(input: &str) -> IResult<&str, &str> {
    delimited(tag("//"), take_until("\n"), char('\n'))(input)
}

fn whitespace0(input: &str) -> IResult<&str, &str> {
    let input_save = input;
    let (input, ss) = many0(alt((comment, multispace1)))(input)?;
    let len = ss.iter().fold(0, |acc: usize, s: &&str| acc + s.len());
    Ok((input, &input_save[..len]))
}

fn whitespace1(input: &str) -> IResult<&str, &str> {
    let input_save = input;
    let (input, ss) = many1(alt((comment, multispace1)))(input)?;
    let len = ss.iter().fold(0, |acc: usize, s: &&str| acc + s.len());
    Ok((input, &input_save[..len]))
}

macro_rules! words {
    ($($element:expr),*) => {
        ($(terminated(tag($element), whitespace1)),*)
    };
}

macro_rules! tags {
    ($($element:expr),*) => {
        ($(tag($element)),*)
    };
}

// FIXME: consider comments, do not split by line
fn parse_directive(input: &str) -> IResult<&str, Directive> {
    let (input, (directive, args)) = terminated(
        pair(
            alt(words!(
                ".version",
                ".target",
                ".address_size",
                ".maxntid",
                ".minnctapersm"
            )),
            take_until("\n"),
        ),
        char('\n'),
    )(input)?;
    Ok((input, Directive { directive, args }))
}

// FIXME: consider comments
fn rest_of_stmt(input: &str) -> IResult<&str, &str> {
    terminated(take_until(";"), char(';'))(input)
}

fn parse_global_var(input: &str) -> IResult<&str, GlobalVar> {
    let (input, linkage) = parse_linkage(input)?;
    let (input, var_type) = terminated(
        alt((tag(".shared"), tag(".global"), tag(".const"))),
        whitespace1,
    )(input)?;
    let (input, args) = rest_of_stmt(input)?;
    Ok((
        input,
        GlobalVar {
            linkage,
            var_type,
            args,
        },
    ))
}

#[test]
fn test_parse_global_var() {
    let input =
        ".visible .global .align 1 .b8 __unnamed_1[11] = {111, 112, 101, 114, 97, 116, 111, 114, 40, 41};";
    let (_, global) = parse_global_var(input).unwrap();
}

fn parse_ident(input: &str) -> IResult<&str, &str> {
    return take_while1(|c: char| c.is_ascii_alphanumeric() || c == '_')(input);
}

#[derive(Debug)]
pub struct Linkage {
    // external symbol
    pub external: bool,
    // the symbol is visible to other compilation unit
    pub visible: bool,
    // visible but allow re-definition
    pub weak: bool,
}

fn parse_linkage(input: &str) -> IResult<&str, Linkage> {
    let (input, attrs) = many0(alt(words!(".extern", ".visible", ".weak")))(input)?;
    let mut ret = Linkage {
        external: false,
        visible: false,
        weak: false,
    };
    for attr in attrs {
        match attr {
            ".extern" => ret.external = true,
            ".visible" => ret.visible = true,
            ".weak" => ret.weak = true,
            _ => unreachable!(),
        }
    }
    Ok((input, ret))
}

#[derive(Debug)]
pub struct Function<'a> {
    pub linkage: Linkage,
    pub entry: bool,
    pub retval: Option<&'a str>,
    pub name: &'a str,
    pub params: Vec<&'a str>,
    pub directives: Vec<Directive<'a>>,
    pub body: Option<Block<'a>>,
}

#[derive(Debug)]
pub struct Block<'a> {
    pub stmts: Vec<Stmt<'a>>,
}

#[derive(Debug)]
pub struct Label<'a> {
    pub label: &'a str,
}

#[derive(Debug)]
pub struct Instruction<'a> {
    pub pred: Option<(bool, &'a str)>,
    pub op: &'a str,
    pub args: &'a str,
}

#[derive(Debug)]
pub struct StoreInst<'a> {
    pub pred: Option<(bool, &'a str)>,
    pub sspace: Option<&'a str>,
    pub op: &'a str,
    pub addr: &'a str,
    // for simplicy, avoid parsing {reg1, reg2}
    pub args: &'a str,
}

#[derive(Debug)]
pub struct CallInst<'a> {
    pub pred: Option<(bool, &'a str)>,
    pub op: &'a str,
    pub retval: Option<&'a str>,
    pub func: &'a str,
    pub params: Option<Vec<&'a str>>,
    // indirect call
    pub call_targets: Option<&'a str>,
}

#[derive(Debug)]
pub enum Stmt<'a> {
    Blk(Block<'a>),
    Label(Label<'a>),
    Inst(Instruction<'a>),
    Store(StoreInst<'a>),
    Call(CallInst<'a>),
    // patching
    Patch(String),
}

#[derive(Debug)]
pub struct Directive<'a> {
    pub directive: &'a str,
    pub args: &'a str,
}

#[derive(Debug)]
pub struct GlobalVar<'a> {
    pub linkage: Linkage,
    pub var_type: &'a str,
    pub args: &'a str,
}

#[derive(Debug)]
pub enum Global<'a> {
    Func(Function<'a>),
    Dir(Directive<'a>),
    Var(GlobalVar<'a>),
}

#[derive(Debug)]
pub struct Ptx<'a> {
    pub globals: Vec<Global<'a>>,
}

// FIXME: consider comments
fn parse_param(input: &str) -> IResult<&str, Vec<&str>> {
    let (input, params) =
        terminated(separated_list0(tag(","), take_while1(|c| c != ',')), eof)(input)?;
    let params = params.iter().map(|p| p.trim()).collect();
    Ok((input, params))
}

fn parse_label(input: &str) -> IResult<&str, Label> {
    let (input, label) = take_while1(|c: char| c.is_alphanumeric() || "_$".contains(c))(input)?;
    let (input, _) = preceded(whitespace0, char(':'))(input)?;
    Ok((input, Label { label }))
}

fn parse_reg(input: &str) -> IResult<&str, &str> {
    take_while1(|c: char| c.is_ascii_alphanumeric() || c == '%')(input)
}

fn parse_general_inst(input: &str) -> IResult<&str, Instruction> {
    let (input, pred) = opt(delimited(
        char('@'),
        pair(map(opt(char('!')), |x| x.is_some()), parse_reg),
        whitespace1,
    ))(input)?;
    let (input, op) = terminated(
        take_while1(|c: char| c.is_ascii_alphanumeric() || ".:_".contains(c)),
        peek(alt((whitespace1, tag(";")))),
    )(input)?;
    let (input, mut args) = rest_of_stmt(input)?;
    args = args.trim();
    Ok((input, Instruction { pred, op, args }))
}

fn parse_store_inst(input: &str) -> IResult<&str, StoreInst> {
    let (input, inst) = parse_general_inst(input)?;
    let (_, sspace) = preceded(
        tag("st"),
        preceded(
            opt(alt(tags!(
                ".weak",
                ".volatile",
                ".relaxed.scope",
                ".mmio.relaxed.sys"
            ))),
            opt(alt((tags!(".global", ".local", ".shared", ".param")))),
        ),
    )(inst.op)?;
    let (_, (addr, mut args)) = pair(
        terminated(
            delimited(
                whitespace0.and(char('[')),
                take_until("]"),
                char(']').and(whitespace0),
            ),
            char(','),
        ),
        take_while1(|_| true),
    )(inst.args)?;
    args = args.trim();

    Ok((
        input,
        StoreInst {
            pred: inst.pred,
            sspace,
            op: inst.op,
            addr,
            args,
        },
    ))
}

fn parse_call_inst(input: &str) -> IResult<&str, CallInst> {
    let (input, inst) = parse_general_inst(input)?;
    tag("call")(inst.op)?;
    let (args, retval) = opt(terminated(
        parse_parenthesized,
        tuple((whitespace0, char(','), whitespace0)),
    ))(inst.args)?;
    let (args, func) = take_till1(|c: char| c == ',' || c.is_whitespace())(args)?;
    let (args, params) = opt(preceded(
        tuple((whitespace0, char(','), whitespace0)),
        parse_parenthesized.and_then(parse_param),
    ))(args)?;
    let (_, call_targets) = opt(preceded(
        tuple((whitespace0, char(','), whitespace0)),
        take_while1(|_| true),
    ))(args)?;

    Ok((
        input,
        CallInst {
            pred: inst.pred,
            op: inst.op,
            retval,
            func,
            params,
            call_targets,
        },
    ))
}

#[test]
fn test_parse_store_inst() {
    let (_, inst) = parse_store_inst("st.global.f64 [%rd37], %fd166;").unwrap();
    print!("{:#?}", inst);
}

fn parse_inst(input: &str) -> IResult<&str, Stmt> {
    alt((
        map(parse_store_inst, |i| Stmt::Store(i)),
        map(parse_call_inst, |i| Stmt::Call(i)),
        map(parse_general_inst, |i| Stmt::Inst(i)),
    ))(input)
}

fn parse_block(input: &str) -> IResult<&str, Block> {
    let (input, _) = char('{')(input)?;
    // FIXME: parse directives in the block
    let (input, stmts) = many0(preceded(
        whitespace0,
        alt((
            map(parse_label, |l| Stmt::Label(l)),
            parse_inst,
            map(parse_block, |b: Block| Stmt::Blk(b)),
        )),
    ))(input)?;
    let (input, _) = preceded(whitespace0, char('}'))(input)?;
    Ok((input, Block { stmts }))
}

#[test]
fn test_parse_block() {
    let (_, block) = parse_block("{ret;\n // \n ret; {// \n ret;} ret;\n}").unwrap();
    print!("{:#?}", block);
}

fn parse_function(input: &str) -> IResult<&str, Function> {
    let (input, linkage) = parse_linkage(input)?;
    // FIXME: for func with retval, there could be no witespace
    let (input, entry) = terminated(
        alt((value(true, tag(".entry")), value(false, tag(".func")))),
        whitespace1,
    )(input)?;
    let (input, retval) = opt(terminated(parse_parenthesized, whitespace0))(input)?;
    let (input, name) = terminated(parse_ident, whitespace0)(input)?;
    let (input, param) = terminated(parse_parenthesized, whitespace0)(input)?;
    let (_, params) = parse_param(param)?;
    let (input, directives) = terminated(many0(parse_directive), whitespace0)(input)?;
    let (input, body) = alt((map(parse_block, |b| Some(b)), map(char(';'), |_| None)))(input)?;
    Ok((
        input,
        Function {
            linkage,
            entry,
            retval,
            name,
            params,
            directives,
            body,
        },
    ))
}

#[test]
fn test_parse_function() {
    let (_, func) = parse_function(concat!(
        ".visible .entry __foo1(.param .u32 param1, .param .align 8 .b8 param2)",
        ".maxntid 128, 1, 1\n",
        ".minnctapersm 4\n",
        "{\n {}ret;\n}"
    ))
    .unwrap();
    print!("{:#?}", func);
    let (_, func) = parse_function(concat!(
        ".weak .func (.param .b64 func_retval0) __internal_accurate_pow\n",
        "(\n",
        ".param .b64 __internal_accurate_pow_param_0,\n",
        ".param .b64 __internal_accurate_pow_param_1\n",
        ")\n",
        ";"
    ))
    .unwrap();
    print!("{:#?}", func);
}

fn check_insts_in_block(blk: &Block) {
    for stmt in &blk.stmts {
        match stmt {
            Stmt::Inst(i) => {
                assert!(
                    !(i.op.starts_with("st") || i.op.starts_with("call")),
                    "{} {}",
                    i.op,
                    i.args
                );
            }
            Stmt::Blk(b) => {
                check_insts_in_block(b);
            }
            _ => (),
        }
    }
}

fn sanitize_check(ptx: &Ptx) {
    // do we parse all call & st inst correctly?
    for g in &ptx.globals {
        match g {
            Global::Func(f) => {
                if let Some(b) = &f.body {
                    check_insts_in_block(b);
                }
            }
            _ => (),
        }
    }
}

pub fn parse_ptx(input: &str) -> IResult<&str, Ptx> {
    let (input, _) = whitespace0(input)?;
    let (input, globals) = terminated(
        many0(terminated(
            alt((
                map(parse_directive, |d| Global::Dir(d)),
                map(parse_function, |f| Global::Func(f)),
                map(parse_global_var, |v| Global::Var(v)),
            )),
            whitespace0,
        )),
        eof,
    )(input)?;

    let ptx = Ptx { globals };
    sanitize_check(&ptx);

    Ok((input, ptx))
}

#[test]
fn test_parse_ptx() {
    let mut file = fs::File::open("ptx/torch.ptx").unwrap();
    let mut ptx_str = String::new();
    file.read_to_string(&mut ptx_str).unwrap();
    let (_, ptx) = parse_ptx(&ptx_str).unwrap();
    print!("{:#?}", ptx);
}
