/**
 * @file
 * @brief Generated functions to emit code for assembler ir nodes.
 * @note  DO NOT EDIT THIS FILE, your changes will be lost.
 *         Edit ./ir/be/riscv/riscv_spec.pl instead.
 *         created by: ./ir/be/scripts/generate_emitter.pl ./ir/be/riscv/riscv_spec.pl build/gen/ir/be/riscv
 * @date  Thu Sep 28 13:11:31 2023
 */
#include "gen_riscv_emitter.h"

#include "beemithlp.h"
#include "gen_riscv_new_nodes.h"
#include "riscv_emitter.h"

static void emit_riscv_add(ir_node const *const node)
{
	riscv_emitf(node, "add	%D0, %S0, %S1");
}

static void emit_riscv_addi(ir_node const *const node)
{
	riscv_emitf(node, "addi	%D0, %S0, %I");
}

static void emit_riscv_addiw(ir_node const *const node)
{
	riscv_emitf(node, "addiw	%D0, %S0, %I");
}

static void emit_riscv_addw(ir_node const *const node)
{
	riscv_emitf(node, "addw	%D0, %S0, %S1");
}

static void emit_riscv_and(ir_node const *const node)
{
	riscv_emitf(node, "and	%D0, %S0, %S1");
}

static void emit_riscv_andi(ir_node const *const node)
{
	riscv_emitf(node, "andi	%D0, %S0, %I");
}

static void emit_riscv_div(ir_node const *const node)
{
	riscv_emitf(node, "div	%D0, %S0, %S1");
}

static void emit_riscv_divu(ir_node const *const node)
{
	riscv_emitf(node, "divu	%D0, %S0, %S1");
}

static void emit_riscv_divuw(ir_node const *const node)
{
	riscv_emitf(node, "divuw	%D0, %S0, %S1");
}

static void emit_riscv_divw(ir_node const *const node)
{
	riscv_emitf(node, "divw	%D0, %S0, %S1");
}

static void emit_riscv_ijmp(ir_node const *const node)
{
	riscv_emitf(node, "jr	%S0");
}

static void emit_riscv_jal(ir_node const *const node)
{
	riscv_emitf(node, "jal	%J");
}

static void emit_riscv_lb(ir_node const *const node)
{
	riscv_emitf(node, "lb	%D1, %A");
}

static void emit_riscv_lbu(ir_node const *const node)
{
	riscv_emitf(node, "lbu	%D1, %A");
}

static void emit_riscv_ld(ir_node const *const node)
{
	riscv_emitf(node, "ld	%D1, %A");
}

static void emit_riscv_lh(ir_node const *const node)
{
	riscv_emitf(node, "lh	%D1, %A");
}

static void emit_riscv_lhu(ir_node const *const node)
{
	riscv_emitf(node, "lhu	%D1, %A");
}

static void emit_riscv_li(ir_node const *const node)
{
	riscv_emitf(node, "li	%D0, %K");
}

static void emit_riscv_lui(ir_node const *const node)
{
	riscv_emitf(node, "lui	%D0, %H");
}

static void emit_riscv_lw(ir_node const *const node)
{
	riscv_emitf(node, "lw	%D1, %A");
}

static void emit_riscv_lwu(ir_node const *const node)
{
	riscv_emitf(node, "lwu	%D1, %A");
}

static void emit_riscv_mul(ir_node const *const node)
{
	riscv_emitf(node, "mul	%D0, %S0, %S1");
}

static void emit_riscv_mulh(ir_node const *const node)
{
	riscv_emitf(node, "mulh	%D0, %S0, %S1");
}

static void emit_riscv_mulhu(ir_node const *const node)
{
	riscv_emitf(node, "mulhu	%D0, %S0, %S1");
}

static void emit_riscv_mulw(ir_node const *const node)
{
	riscv_emitf(node, "mulw	%D0, %S0, %S1");
}

static void emit_riscv_or(ir_node const *const node)
{
	riscv_emitf(node, "or	%D0, %S0, %S1");
}

static void emit_riscv_ori(ir_node const *const node)
{
	riscv_emitf(node, "ori	%D0, %S0, %I");
}

static void emit_riscv_rem(ir_node const *const node)
{
	riscv_emitf(node, "rem	%D0, %S0, %S1");
}

static void emit_riscv_remu(ir_node const *const node)
{
	riscv_emitf(node, "remu	%D0, %S0, %S1");
}

static void emit_riscv_remuw(ir_node const *const node)
{
	riscv_emitf(node, "remuw	%D0, %S0, %S1");
}

static void emit_riscv_remw(ir_node const *const node)
{
	riscv_emitf(node, "remw	%D0, %S0, %S1");
}

static void emit_riscv_ret(ir_node const *const node)
{
	riscv_emitf(node, "ret");
}

static void emit_riscv_sb(ir_node const *const node)
{
	riscv_emitf(node, "sb	%S2, %A");
}

static void emit_riscv_sd(ir_node const *const node)
{
	riscv_emitf(node, "sd	%S2, %A");
}

static void emit_riscv_sh(ir_node const *const node)
{
	riscv_emitf(node, "sh	%S2, %A");
}

static void emit_riscv_sll(ir_node const *const node)
{
	riscv_emitf(node, "sll	%D0, %S0, %S1");
}

static void emit_riscv_slli(ir_node const *const node)
{
	riscv_emitf(node, "slli	%D0, %S0, %I");
}

static void emit_riscv_slliw(ir_node const *const node)
{
	riscv_emitf(node, "slliw	%D0, %S0, %I");
}

static void emit_riscv_sllw(ir_node const *const node)
{
	riscv_emitf(node, "sllw	%D0, %S0, %S1");
}

static void emit_riscv_slt(ir_node const *const node)
{
	riscv_emitf(node, "slt	%D0, %S0, %S1");
}

static void emit_riscv_sltiu(ir_node const *const node)
{
	riscv_emitf(node, "sltiu	%D0, %S0, %I");
}

static void emit_riscv_sltu(ir_node const *const node)
{
	riscv_emitf(node, "sltu	%D0, %S0, %S1");
}

static void emit_riscv_sra(ir_node const *const node)
{
	riscv_emitf(node, "sra	%D0, %S0, %S1");
}

static void emit_riscv_srai(ir_node const *const node)
{
	riscv_emitf(node, "srai	%D0, %S0, %I");
}

static void emit_riscv_sraiw(ir_node const *const node)
{
	riscv_emitf(node, "sraiw	%D0, %S0, %I");
}

static void emit_riscv_sraw(ir_node const *const node)
{
	riscv_emitf(node, "sraw	%D0, %S0, %S1");
}

static void emit_riscv_srl(ir_node const *const node)
{
	riscv_emitf(node, "srl	%D0, %S0, %S1");
}

static void emit_riscv_srli(ir_node const *const node)
{
	riscv_emitf(node, "srli	%D0, %S0, %I");
}

static void emit_riscv_srliw(ir_node const *const node)
{
	riscv_emitf(node, "srliw	%D0, %S0, %I");
}

static void emit_riscv_srlw(ir_node const *const node)
{
	riscv_emitf(node, "srlw	%D0, %S0, %S1");
}

static void emit_riscv_sub(ir_node const *const node)
{
	riscv_emitf(node, "sub	%D0, %S0, %S1");
}

static void emit_riscv_subw(ir_node const *const node)
{
	riscv_emitf(node, "subw	%D0, %S0, %S1");
}

static void emit_riscv_sw(ir_node const *const node)
{
	riscv_emitf(node, "sw	%S2, %A");
}

static void emit_riscv_xor(ir_node const *const node)
{
	riscv_emitf(node, "xor	%D0, %S0, %S1");
}

static void emit_riscv_xori(ir_node const *const node)
{
	riscv_emitf(node, "xori	%D0, %S0, %I");
}



void riscv_register_spec_emitters(void)
{
	be_set_emitter(op_riscv_add, emit_riscv_add);
	be_set_emitter(op_riscv_addi, emit_riscv_addi);
	be_set_emitter(op_riscv_addiw, emit_riscv_addiw);
	be_set_emitter(op_riscv_addw, emit_riscv_addw);
	be_set_emitter(op_riscv_and, emit_riscv_and);
	be_set_emitter(op_riscv_andi, emit_riscv_andi);
	be_set_emitter(op_riscv_div, emit_riscv_div);
	be_set_emitter(op_riscv_divu, emit_riscv_divu);
	be_set_emitter(op_riscv_divuw, emit_riscv_divuw);
	be_set_emitter(op_riscv_divw, emit_riscv_divw);
	be_set_emitter(op_riscv_ijmp, emit_riscv_ijmp);
	be_set_emitter(op_riscv_jal, emit_riscv_jal);
	be_set_emitter(op_riscv_lb, emit_riscv_lb);
	be_set_emitter(op_riscv_lbu, emit_riscv_lbu);
	be_set_emitter(op_riscv_ld, emit_riscv_ld);
	be_set_emitter(op_riscv_lh, emit_riscv_lh);
	be_set_emitter(op_riscv_lhu, emit_riscv_lhu);
	be_set_emitter(op_riscv_li, emit_riscv_li);
	be_set_emitter(op_riscv_lui, emit_riscv_lui);
	be_set_emitter(op_riscv_lw, emit_riscv_lw);
	be_set_emitter(op_riscv_lwu, emit_riscv_lwu);
	be_set_emitter(op_riscv_mul, emit_riscv_mul);
	be_set_emitter(op_riscv_mulh, emit_riscv_mulh);
	be_set_emitter(op_riscv_mulhu, emit_riscv_mulhu);
	be_set_emitter(op_riscv_mulw, emit_riscv_mulw);
	be_set_emitter(op_riscv_or, emit_riscv_or);
	be_set_emitter(op_riscv_ori, emit_riscv_ori);
	be_set_emitter(op_riscv_rem, emit_riscv_rem);
	be_set_emitter(op_riscv_remu, emit_riscv_remu);
	be_set_emitter(op_riscv_remuw, emit_riscv_remuw);
	be_set_emitter(op_riscv_remw, emit_riscv_remw);
	be_set_emitter(op_riscv_ret, emit_riscv_ret);
	be_set_emitter(op_riscv_sb, emit_riscv_sb);
	be_set_emitter(op_riscv_sd, emit_riscv_sd);
	be_set_emitter(op_riscv_sh, emit_riscv_sh);
	be_set_emitter(op_riscv_sll, emit_riscv_sll);
	be_set_emitter(op_riscv_slli, emit_riscv_slli);
	be_set_emitter(op_riscv_slliw, emit_riscv_slliw);
	be_set_emitter(op_riscv_sllw, emit_riscv_sllw);
	be_set_emitter(op_riscv_slt, emit_riscv_slt);
	be_set_emitter(op_riscv_sltiu, emit_riscv_sltiu);
	be_set_emitter(op_riscv_sltu, emit_riscv_sltu);
	be_set_emitter(op_riscv_sra, emit_riscv_sra);
	be_set_emitter(op_riscv_srai, emit_riscv_srai);
	be_set_emitter(op_riscv_sraiw, emit_riscv_sraiw);
	be_set_emitter(op_riscv_sraw, emit_riscv_sraw);
	be_set_emitter(op_riscv_srl, emit_riscv_srl);
	be_set_emitter(op_riscv_srli, emit_riscv_srli);
	be_set_emitter(op_riscv_srliw, emit_riscv_srliw);
	be_set_emitter(op_riscv_srlw, emit_riscv_srlw);
	be_set_emitter(op_riscv_sub, emit_riscv_sub);
	be_set_emitter(op_riscv_subw, emit_riscv_subw);
	be_set_emitter(op_riscv_sw, emit_riscv_sw);
	be_set_emitter(op_riscv_xor, emit_riscv_xor);
	be_set_emitter(op_riscv_xori, emit_riscv_xori);

}

void riscv_register_spec_binary_emitters(void)
{

}
