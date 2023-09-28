# This file is part of libFirm.
# Copyright (C) 2018 Christoph Mallon.

$arch = "riscv";

my $mode_gp = "mode_rv_gp";

%reg_classes = (
	gp => {
		mode => $mode_gp,
		registers => [
			# Registers are reordered for improved code compression in RISC-V C

			# The register list has been reordered as the RISC-V C extension
			# which defines a subgroup of registers with shortened register
			# specifiers suitable for use in more compressed instructions.
			# However, this subgroup does not consist of the first
			# general-purpose registers (e.g. due to the RISC-V calling convention),
			# but is comprised of registers x8 through x15.
			#
			# To improve compression of code, this register list has been reordered
			# to start with the registers suitable for compression. This may allow
			# the register allocator to assign registers suitable for compression
			# with more instructions in cases where no other constraints take effect
			# (e.g. in cases with low register pressure).

			# Registers x8 through x15
			{ name => "s0",   encoding =>  8, type => "compressible" },
			{ name => "s1",   encoding =>  9, type => "compressible" },
			{ name => "a0",   encoding => 10, type => "compressible" },
			{ name => "a1",   encoding => 11, type => "compressible" },
			{ name => "a2",   encoding => 12, type => "compressible" },
			{ name => "a3",   encoding => 13, type => "compressible" },
			{ name => "a4",   encoding => 14, type => "compressible" },
			{ name => "a5",   encoding => 15, type => "compressible" },

			# Registers x0 through x7
			{ name => "zero", encoding =>  0 },
			{ name => "ra",   encoding =>  1 },
			{ name => "sp",   encoding =>  2 },
			{ name => "gp",   encoding =>  3 },
			{ name => "tp",   encoding =>  4 },
			{ name => "t0",   encoding =>  5 },
			{ name => "t1",   encoding =>  6 },
			{ name => "t2",   encoding =>  7 },

			{ name => "a6",   encoding => 16 },
			{ name => "a7",   encoding => 17 },
			{ name => "s2",   encoding => 18 },
			{ name => "s3",   encoding => 19 },
			{ name => "s4",   encoding => 20 },
			{ name => "s5",   encoding => 21 },
			{ name => "s6",   encoding => 22 },
			{ name => "s7",   encoding => 23 },
			{ name => "s8",   encoding => 24 },
			{ name => "s9",   encoding => 25 },
			{ name => "s10",  encoding => 26 },
			{ name => "s11",  encoding => 27 },
			{ name => "t3",   encoding => 28 },
			{ name => "t4",   encoding => 29 },
			{ name => "t5",   encoding => 30 },
			{ name => "t6",   encoding => 31 },
		]
	},
);

%init_attr = (
	riscv_attr_t => "",
	riscv_cond_attr_t =>
		"attr->cond = cond;",
	riscv_immediate_attr_t =>
		"attr->ent = ent;\n".
		"\tattr->val = val;",
	riscv_switch_attr_t =>
		"be_switch_attr_init(res, &attr->swtch, table, table_entity);",
);

my $binOp = {
	irn_flags => [ "rematerializable" ],
	in_reqs   => [ "cls-gp", "cls-gp" ],
	out_reqs  => [ "cls-gp" ],
	ins       => [ "left", "right" ],
	outs      => [ "res" ],
	emit      => "{name}\t%D0, %S0, %S1",
};

my $callOp = {
  state     => "exc_pinned",
  in_reqs   => "...",
  out_reqs  => "...",
  ins       => [ "mem", "stack", "first_argument" ],
  outs      => [ "M",   "stack", "first_result" ],
};

my $immediateOp = {
	irn_flags => [ "rematerializable" ],
	in_reqs   => [ "cls-gp" ],
	out_reqs  => [ "cls-gp" ],
	ins       => [ "left" ],
	outs      => [ "res" ],
	attr_type => "riscv_immediate_attr_t",
	attr      => "ir_entity *const ent, int64_t const val",
	emit      => "{name}\t%D0, %S0, %I",
};

my $loadOp = {
	state     => "exc_pinned",
	in_reqs   => [ "mem", "cls-gp" ],
	out_reqs  => [ "mem", "cls-gp" ],
	ins       => [ "mem", "base" ],
	outs      => [ "M", "res" ],
	attr_type => "riscv_immediate_attr_t",
	attr      => "ir_entity *const ent, int32_t const val",
	emit      => "{name}\t%D1, %A",
};

my $storeOp = {
	state     => "exc_pinned",
	in_reqs   => [ "mem", "cls-gp", "cls-gp" ],
	out_reqs  => [ "mem" ],
	ins       => [ "mem", "base", "value" ],
	outs      => [ "M" ],
	attr_type => "riscv_immediate_attr_t",
	attr      => "ir_entity *const ent, int32_t const val",
	emit      => "{name}\t%S2, %A",
};

%nodes = (

add => { template => $binOp },

addw => { template => $binOp },

addi => { template => $immediateOp },

addiw => { template => $immediateOp },

and => { template => $binOp },

andi => { template => $immediateOp },

bcc => {
	state     => "pinned",
	op_flags  => [ "cfopcode", "forking" ],
	in_reqs   => [ "cls-gp", "cls-gp" ],
	ins       => [ "left", "right" ],
	out_reqs  => [ "exec", "exec" ],
	outs      => [ "false", "true" ],
	attr_type => "riscv_cond_attr_t",
	attr      => "riscv_cond_t const cond",
},

div => { template => $binOp, },

divw => { template => $binOp, },

divu => { template => $binOp, },

divuw => { template => $binOp, },

ijmp => {
	state    => "pinned",
	op_flags => [ "cfopcode", "unknown_jump" ],
	in_reqs  => [ "cls-gp" ],
	out_reqs => [ "exec" ],
	emit     => "jr\t%S0",
},

j => {
	state     => "pinned",
	irn_flags => [ "simple_jump", "fallthrough" ],
	op_flags  => [ "cfopcode" ],
	out_reqs  => [ "exec" ],
},

jal => {
	template  => $callOp,
	attr_type => "riscv_immediate_attr_t",
	attr      => "ir_entity *const ent, int32_t const val",
	emit      => "jal\t%J",
},

jalr => {
	template => $callOp,
	# emit     => "jalr\t%S2",
},

lb => { template => $loadOp },

lbu => { template => $loadOp },

ld => { template => $loadOp },

lh => { template => $loadOp },

lhu => { template => $loadOp },

# Pseudoinstruction for the time being
li => {
	template => $immediateOp,
	in_reqs   => [],
	ins       => [],
	emit      => "li\t%D0, %K",
},

lui => {
	template  => $immediateOp,
	in_reqs   => [],
	ins       => [],
	emit      => "lui\t%D0, %H",
},

lw => { template => $loadOp },

lwu => { template => $loadOp },

mul => { template => $binOp, },

mulw => { template => $binOp, },

mulh => { template => $binOp, },

mulhu => { template => $binOp, },

or => { template => $binOp },

ori => { template => $immediateOp },

rem => { template => $binOp, },

remw => { template => $binOp, },

remu => { template => $binOp, },

remuw => { template => $binOp, },

ret => {
	state    => "pinned",
	op_flags => [ "cfopcode" ],
	in_reqs  => "...",
	out_reqs => [ "exec" ],
	ins      => [ "mem", "stack", "addr", "first_result" ],
	emit     => "ret",
},

sb => { template => $storeOp },

sd => { template => $storeOp },

sh => { template => $storeOp },

sll => { template => $binOp },

sllw => { template => $binOp },

slli => { template => $immediateOp },

slliw => { template => $immediateOp },

slt => { template => $binOp },

sltiu => { template => $immediateOp },

sltu => { template => $binOp },

sra => { template => $binOp },

sraw => { template => $binOp },

srai => { template => $immediateOp },

sraiw => { template => $immediateOp },

srl => { template => $binOp },

srlw => { template => $binOp },

srli => { template => $immediateOp },

srliw => { template => $immediateOp },

sub => { template => $binOp },

subw => { template => $binOp },

sw => { template => $storeOp },

switch => {
	op_flags  => [ "cfopcode", "forking" ],
	state     => "pinned",
	in_reqs   => [ "cls-gp" ],
	out_reqs  => "...",
	attr_type => "riscv_switch_attr_t",
	attr      => "const ir_switch_table *table, ir_entity *table_entity",
},

xor => { template => $binOp },

xori => { template => $immediateOp },

FrameAddr => {
	op_flags  => [ "constlike" ],
	irn_flags => [ "rematerializable" ],
	attr      => "ir_entity *ent, int32_t val",
	in_reqs   => [ "cls-gp" ],
	out_reqs  => [ "cls-gp" ],
	ins       => [ "base" ],
	attr_type => "riscv_immediate_attr_t",
},

SubSP => {
	in_reqs => [ "mem", "sp", "cls-gp" ],
	ins     => [ "mem", "stack", "size" ],
	out_reqs => [ "sp:I", "cls-gp", "mem" ],
	outs     => [ "stack", "addr", "M" ],
},

SubSPimm => {
	in_reqs => [ "mem", "sp" ],
	ins      => [ "mem", "stack" ],
	out_reqs => [ "sp:I", "cls-gp", "mem" ],
	outs     => [ "stack", "addr", "M" ],
	attr_type => "riscv_immediate_attr_t",
	attr    => "ir_entity *ent, int32_t val",
},

);
