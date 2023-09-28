#include "gen_riscv_new_nodes.h"

#include "benode.h"
#include "riscv_bearch_t.h"
#include "gen_riscv_regalloc_if.h"
#include "riscv_new_nodes_t.h"
#include "fourcc.h"
#include "irgopt.h"
#include "ircons_t.h"

ir_op *op_riscv_FrameAddr = NULL;
ir_op *op_riscv_SubSP = NULL;
ir_op *op_riscv_SubSPimm = NULL;
ir_op *op_riscv_add = NULL;
ir_op *op_riscv_addi = NULL;
ir_op *op_riscv_addiw = NULL;
ir_op *op_riscv_addw = NULL;
ir_op *op_riscv_and = NULL;
ir_op *op_riscv_andi = NULL;
ir_op *op_riscv_bcc = NULL;
ir_op *op_riscv_div = NULL;
ir_op *op_riscv_divu = NULL;
ir_op *op_riscv_divuw = NULL;
ir_op *op_riscv_divw = NULL;
ir_op *op_riscv_ijmp = NULL;
ir_op *op_riscv_j = NULL;
ir_op *op_riscv_jal = NULL;
ir_op *op_riscv_jalr = NULL;
ir_op *op_riscv_lb = NULL;
ir_op *op_riscv_lbu = NULL;
ir_op *op_riscv_ld = NULL;
ir_op *op_riscv_lh = NULL;
ir_op *op_riscv_lhu = NULL;
ir_op *op_riscv_li = NULL;
ir_op *op_riscv_lui = NULL;
ir_op *op_riscv_lw = NULL;
ir_op *op_riscv_lwu = NULL;
ir_op *op_riscv_mul = NULL;
ir_op *op_riscv_mulh = NULL;
ir_op *op_riscv_mulhu = NULL;
ir_op *op_riscv_mulw = NULL;
ir_op *op_riscv_or = NULL;
ir_op *op_riscv_ori = NULL;
ir_op *op_riscv_rem = NULL;
ir_op *op_riscv_remu = NULL;
ir_op *op_riscv_remuw = NULL;
ir_op *op_riscv_remw = NULL;
ir_op *op_riscv_ret = NULL;
ir_op *op_riscv_sb = NULL;
ir_op *op_riscv_sd = NULL;
ir_op *op_riscv_sh = NULL;
ir_op *op_riscv_sll = NULL;
ir_op *op_riscv_slli = NULL;
ir_op *op_riscv_slliw = NULL;
ir_op *op_riscv_sllw = NULL;
ir_op *op_riscv_slt = NULL;
ir_op *op_riscv_sltiu = NULL;
ir_op *op_riscv_sltu = NULL;
ir_op *op_riscv_sra = NULL;
ir_op *op_riscv_srai = NULL;
ir_op *op_riscv_sraiw = NULL;
ir_op *op_riscv_sraw = NULL;
ir_op *op_riscv_srl = NULL;
ir_op *op_riscv_srli = NULL;
ir_op *op_riscv_srliw = NULL;
ir_op *op_riscv_srlw = NULL;
ir_op *op_riscv_sub = NULL;
ir_op *op_riscv_subw = NULL;
ir_op *op_riscv_sw = NULL;
ir_op *op_riscv_switch = NULL;
ir_op *op_riscv_xor = NULL;
ir_op *op_riscv_xori = NULL;


static int riscv_opcode_start = -1;

/** A tag for the riscv opcodes. */
#define riscv_op_tag FOURCC('r', 'i', 's', 'c')

/** Return 1 if the given opcode is a riscv machine op, 0 otherwise */
int is_riscv_op(const ir_op *op)
{
	return get_op_tag(op) == riscv_op_tag;
}

/** Return 1 if the given node is a riscv machine node, 0 otherwise */
int is_riscv_irn(const ir_node *node)
{
	return is_riscv_op(get_irn_op(node));
}

int get_riscv_irn_opcode(const ir_node *node)
{
	assert(is_riscv_irn(node));
	return get_irn_opcode(node) - riscv_opcode_start;
}

#undef BIT
#define BIT(x)  (1 << (x))

static const unsigned riscv_limit_gp_sp[] = { BIT(REG_GP_SP), 0 };

static const arch_register_req_t riscv_requirements_gp_sp_I = {
	.cls               = &riscv_reg_classes[CLASS_riscv_gp],
	.limited           = riscv_limit_gp_sp,
	.should_be_same    = 0,
	.must_be_different = 0,
	.width             = 1,
	.ignore = true,
};



ir_node *new_bd_riscv_FrameAddr(dbg_info *dbgi, ir_node *block, ir_node *base, ir_entity *ent, int32_t val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		base,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_FrameAddr, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_SubSP(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *stack, ir_node *size)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_single_reg_req_gp_sp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		stack,
		size,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_SubSP, mode_T, 3, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 3;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_requirements_gp_sp_I;
	out_infos[1].req = &riscv_class_reg_req_gp;
	out_infos[2].req = &arch_memory_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_SubSPimm(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *stack, ir_entity *ent, int32_t val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_single_reg_req_gp_sp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		stack,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_SubSPimm, mode_T, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 3;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_requirements_gp_sp_I;
	out_infos[1].req = &riscv_class_reg_req_gp;
	out_infos[2].req = &arch_memory_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_add(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_add, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_addi(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_addi, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_addiw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_addiw, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_addw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_addw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_and(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_and, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_andi(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_andi, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_bcc(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right, riscv_cond_t const cond)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_bcc, mode_T, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 2;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_cond_attr_t *const attr = (riscv_cond_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->cond = cond;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_exec_requirement;
	out_infos[1].req = &arch_exec_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_div(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_div, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_divu(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_divu, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_divuw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_divuw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_divw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_divw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_ijmp(dbg_info *dbgi, ir_node *block, ir_node *op0)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		op0,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_ijmp, mode_X, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_exec_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_j(dbg_info *dbgi, ir_node *block)
{
	arch_register_req_t const **const in_reqs = NULL;


	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_j, mode_X, 0, NULL);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_simple_jump;
	irn_flags |= arch_irn_flag_fallthrough;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_exec_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_jal(dbg_info *dbgi, ir_node *block, int const arity, ir_node *const *const in, arch_register_req_t const **const in_reqs, int n_res, ir_entity *const ent, int32_t const val)
{


	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_jal, mode_T, arity, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_jalr(dbg_info *dbgi, ir_node *block, int const arity, ir_node *const *const in, arch_register_req_t const **const in_reqs, int n_res)
{


	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_jalr, mode_T, arity, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_lb(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_lb, mode_T, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 2;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;
	out_infos[1].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_lbu(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_lbu, mode_T, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 2;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;
	out_infos[1].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_ld(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_ld, mode_T, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 2;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;
	out_infos[1].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_lh(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_lh, mode_T, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 2;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;
	out_infos[1].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_lhu(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_lhu, mode_T, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 2;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;
	out_infos[1].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_li(dbg_info *dbgi, ir_node *block, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
	};


	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_li, mode_rv_gp, 0, NULL);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_lui(dbg_info *dbgi, ir_node *block, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
	};


	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_lui, mode_rv_gp, 0, NULL);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_lw(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_lw, mode_T, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 2;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;
	out_infos[1].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_lwu(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_lwu, mode_T, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 2;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;
	out_infos[1].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_mul(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_mul, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_mulh(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_mulh, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_mulhu(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_mulhu, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_mulw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_mulw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_or(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_or, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_ori(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_ori, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_rem(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_rem, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_remu(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_remu, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_remuw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_remuw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_remw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_remw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_ret(dbg_info *dbgi, ir_node *block, int const arity, ir_node *const *const in, arch_register_req_t const **const in_reqs)
{


	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_ret, mode_X, arity, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_exec_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sb(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_node *value, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
		value,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sb, mode_M, 3, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sd(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_node *value, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
		value,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sd, mode_M, 3, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sh(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_node *value, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
		value,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sh, mode_M, 3, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sll(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sll, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_slli(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_slli, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_slliw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_slliw, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sllw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sllw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_slt(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_slt, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sltiu(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sltiu, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sltu(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sltu, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sra(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sra, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_srai(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_srai, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sraiw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sraiw, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sraw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sraw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_srl(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_srl, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_srli(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_srli, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_srliw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_srliw, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_srlw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_srlw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sub(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sub, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_subw(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_subw, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_sw(dbg_info *dbgi, ir_node *block, ir_node *mem, ir_node *base, ir_node *value, ir_entity *const ent, int32_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&arch_memory_requirement,
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		mem,
		base,
		value,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_sw, mode_M, 3, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &arch_memory_requirement;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_switch(dbg_info *dbgi, ir_node *block, ir_node *op0, int n_res, const ir_switch_table *table, ir_entity *table_entity)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		op0,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_switch, mode_T, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_switch_attr_t *const attr = (riscv_switch_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	be_switch_attr_init(res, &attr->swtch, table, table_entity);

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_xor(dbg_info *dbgi, ir_node *block, ir_node *left, ir_node *right)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
		right,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_xor, mode_rv_gp, 2, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_attr_t *const attr = (riscv_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}

ir_node *new_bd_riscv_xori(dbg_info *dbgi, ir_node *block, ir_node *left, ir_entity *const ent, int64_t const val)
{
	static arch_register_req_t const *in_reqs[] = {
		&riscv_class_reg_req_gp,
	};

	/* construct in array */
	ir_node *const in[] = {
		left,
	};

	ir_graph *const irg = get_irn_irg(block);
	ir_node  *const res = new_ir_node(dbgi, irg, block, op_riscv_xori, mode_rv_gp, 1, in);

	/* init node attributes */

	/* flags */
	arch_irn_flags_t irn_flags = arch_irn_flags_none;
	irn_flags |= arch_irn_flag_rematerializable;
	int const n_res = 1;
	be_info_init_irn(res, irn_flags, in_reqs, n_res);
	riscv_immediate_attr_t *const attr = (riscv_immediate_attr_t*)get_irn_generic_attr(res);
	(void)attr; /* avoid potential warning */
	attr->ent = ent;
	attr->val = val;
	reg_out_info_t *const out_infos = be_get_info(res)->out_infos;
	out_infos[0].req = &riscv_class_reg_req_gp;

	verify_new_node(res);
	return optimize_node(res);
}


/**
 * Creates the riscv specific Firm machine operations
 * needed for the assembler irgs.
 */
void riscv_create_opcodes(void)
{
	ir_op *op;
	int    cur_opcode = get_next_ir_opcodes(iro_riscv_last);

	riscv_opcode_start = cur_opcode;
	op = new_ir_op(cur_opcode + iro_riscv_FrameAddr, "riscv_FrameAddr", op_pin_state_floats, irop_flag_constlike, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_FrameAddr = op;
	op = new_ir_op(cur_opcode + iro_riscv_SubSP, "riscv_SubSP", op_pin_state_floats, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_SubSP = op;
	op = new_ir_op(cur_opcode + iro_riscv_SubSPimm, "riscv_SubSPimm", op_pin_state_floats, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_SubSPimm = op;
	op = new_ir_op(cur_opcode + iro_riscv_add, "riscv_add", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_add = op;
	op = new_ir_op(cur_opcode + iro_riscv_addi, "riscv_addi", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_addi = op;
	op = new_ir_op(cur_opcode + iro_riscv_addiw, "riscv_addiw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_addiw = op;
	op = new_ir_op(cur_opcode + iro_riscv_addw, "riscv_addw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_addw = op;
	op = new_ir_op(cur_opcode + iro_riscv_and, "riscv_and", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_and = op;
	op = new_ir_op(cur_opcode + iro_riscv_andi, "riscv_andi", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_andi = op;
	op = new_ir_op(cur_opcode + iro_riscv_bcc, "riscv_bcc", op_pin_state_pinned, irop_flag_cfopcode|irop_flag_forking, oparity_any, -1, sizeof(riscv_cond_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_cond_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_bcc = op;
	op = new_ir_op(cur_opcode + iro_riscv_div, "riscv_div", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_div = op;
	op = new_ir_op(cur_opcode + iro_riscv_divu, "riscv_divu", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_divu = op;
	op = new_ir_op(cur_opcode + iro_riscv_divuw, "riscv_divuw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_divuw = op;
	op = new_ir_op(cur_opcode + iro_riscv_divw, "riscv_divw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_divw = op;
	op = new_ir_op(cur_opcode + iro_riscv_ijmp, "riscv_ijmp", op_pin_state_pinned, irop_flag_cfopcode|irop_flag_unknown_jump, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_ijmp = op;
	op = new_ir_op(cur_opcode + iro_riscv_j, "riscv_j", op_pin_state_pinned, irop_flag_cfopcode, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_j = op;
	op = new_ir_op(cur_opcode + iro_riscv_jal, "riscv_jal", op_pin_state_exc_pinned, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_jal = op;
	op = new_ir_op(cur_opcode + iro_riscv_jalr, "riscv_jalr", op_pin_state_exc_pinned, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_jalr = op;
	op = new_ir_op(cur_opcode + iro_riscv_lb, "riscv_lb", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_lb = op;
	op = new_ir_op(cur_opcode + iro_riscv_lbu, "riscv_lbu", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_lbu = op;
	op = new_ir_op(cur_opcode + iro_riscv_ld, "riscv_ld", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_ld = op;
	op = new_ir_op(cur_opcode + iro_riscv_lh, "riscv_lh", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_lh = op;
	op = new_ir_op(cur_opcode + iro_riscv_lhu, "riscv_lhu", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_lhu = op;
	op = new_ir_op(cur_opcode + iro_riscv_li, "riscv_li", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_li = op;
	op = new_ir_op(cur_opcode + iro_riscv_lui, "riscv_lui", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_lui = op;
	op = new_ir_op(cur_opcode + iro_riscv_lw, "riscv_lw", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_lw = op;
	op = new_ir_op(cur_opcode + iro_riscv_lwu, "riscv_lwu", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_lwu = op;
	op = new_ir_op(cur_opcode + iro_riscv_mul, "riscv_mul", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_mul = op;
	op = new_ir_op(cur_opcode + iro_riscv_mulh, "riscv_mulh", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_mulh = op;
	op = new_ir_op(cur_opcode + iro_riscv_mulhu, "riscv_mulhu", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_mulhu = op;
	op = new_ir_op(cur_opcode + iro_riscv_mulw, "riscv_mulw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_mulw = op;
	op = new_ir_op(cur_opcode + iro_riscv_or, "riscv_or", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_or = op;
	op = new_ir_op(cur_opcode + iro_riscv_ori, "riscv_ori", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_ori = op;
	op = new_ir_op(cur_opcode + iro_riscv_rem, "riscv_rem", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_rem = op;
	op = new_ir_op(cur_opcode + iro_riscv_remu, "riscv_remu", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_remu = op;
	op = new_ir_op(cur_opcode + iro_riscv_remuw, "riscv_remuw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_remuw = op;
	op = new_ir_op(cur_opcode + iro_riscv_remw, "riscv_remw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_remw = op;
	op = new_ir_op(cur_opcode + iro_riscv_ret, "riscv_ret", op_pin_state_pinned, irop_flag_cfopcode, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_ret = op;
	op = new_ir_op(cur_opcode + iro_riscv_sb, "riscv_sb", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sb = op;
	op = new_ir_op(cur_opcode + iro_riscv_sd, "riscv_sd", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sd = op;
	op = new_ir_op(cur_opcode + iro_riscv_sh, "riscv_sh", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sh = op;
	op = new_ir_op(cur_opcode + iro_riscv_sll, "riscv_sll", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sll = op;
	op = new_ir_op(cur_opcode + iro_riscv_slli, "riscv_slli", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_slli = op;
	op = new_ir_op(cur_opcode + iro_riscv_slliw, "riscv_slliw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_slliw = op;
	op = new_ir_op(cur_opcode + iro_riscv_sllw, "riscv_sllw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sllw = op;
	op = new_ir_op(cur_opcode + iro_riscv_slt, "riscv_slt", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_slt = op;
	op = new_ir_op(cur_opcode + iro_riscv_sltiu, "riscv_sltiu", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sltiu = op;
	op = new_ir_op(cur_opcode + iro_riscv_sltu, "riscv_sltu", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sltu = op;
	op = new_ir_op(cur_opcode + iro_riscv_sra, "riscv_sra", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sra = op;
	op = new_ir_op(cur_opcode + iro_riscv_srai, "riscv_srai", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_srai = op;
	op = new_ir_op(cur_opcode + iro_riscv_sraiw, "riscv_sraiw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sraiw = op;
	op = new_ir_op(cur_opcode + iro_riscv_sraw, "riscv_sraw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sraw = op;
	op = new_ir_op(cur_opcode + iro_riscv_srl, "riscv_srl", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_srl = op;
	op = new_ir_op(cur_opcode + iro_riscv_srli, "riscv_srli", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_srli = op;
	op = new_ir_op(cur_opcode + iro_riscv_srliw, "riscv_srliw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_srliw = op;
	op = new_ir_op(cur_opcode + iro_riscv_srlw, "riscv_srlw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_srlw = op;
	op = new_ir_op(cur_opcode + iro_riscv_sub, "riscv_sub", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sub = op;
	op = new_ir_op(cur_opcode + iro_riscv_subw, "riscv_subw", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_subw = op;
	op = new_ir_op(cur_opcode + iro_riscv_sw, "riscv_sw", op_pin_state_exc_pinned, irop_flag_uses_memory, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	ir_op_set_memory_index(op, 0);	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_sw = op;
	op = new_ir_op(cur_opcode + iro_riscv_switch, "riscv_switch", op_pin_state_pinned, irop_flag_cfopcode|irop_flag_forking, oparity_any, -1, sizeof(riscv_switch_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_switch_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_switch = op;
	op = new_ir_op(cur_opcode + iro_riscv_xor, "riscv_xor", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_xor = op;
	op = new_ir_op(cur_opcode + iro_riscv_xori, "riscv_xori", op_pin_state_floats, irop_flag_none, oparity_any, -1, sizeof(riscv_immediate_attr_t));
	set_op_dump(op, riscv_dump_node);
	set_op_attrs_equal(op, riscv_immediate_attrs_equal);
	set_op_copy_attr(op, be_copy_attr);
	set_op_tag(op, riscv_op_tag);
	op_riscv_xori = op;

}

void riscv_free_opcodes(void)
{
	free_ir_op(op_riscv_FrameAddr); op_riscv_FrameAddr = NULL;
	free_ir_op(op_riscv_SubSP); op_riscv_SubSP = NULL;
	free_ir_op(op_riscv_SubSPimm); op_riscv_SubSPimm = NULL;
	free_ir_op(op_riscv_add); op_riscv_add = NULL;
	free_ir_op(op_riscv_addi); op_riscv_addi = NULL;
	free_ir_op(op_riscv_addiw); op_riscv_addiw = NULL;
	free_ir_op(op_riscv_addw); op_riscv_addw = NULL;
	free_ir_op(op_riscv_and); op_riscv_and = NULL;
	free_ir_op(op_riscv_andi); op_riscv_andi = NULL;
	free_ir_op(op_riscv_bcc); op_riscv_bcc = NULL;
	free_ir_op(op_riscv_div); op_riscv_div = NULL;
	free_ir_op(op_riscv_divu); op_riscv_divu = NULL;
	free_ir_op(op_riscv_divuw); op_riscv_divuw = NULL;
	free_ir_op(op_riscv_divw); op_riscv_divw = NULL;
	free_ir_op(op_riscv_ijmp); op_riscv_ijmp = NULL;
	free_ir_op(op_riscv_j); op_riscv_j = NULL;
	free_ir_op(op_riscv_jal); op_riscv_jal = NULL;
	free_ir_op(op_riscv_jalr); op_riscv_jalr = NULL;
	free_ir_op(op_riscv_lb); op_riscv_lb = NULL;
	free_ir_op(op_riscv_lbu); op_riscv_lbu = NULL;
	free_ir_op(op_riscv_ld); op_riscv_ld = NULL;
	free_ir_op(op_riscv_lh); op_riscv_lh = NULL;
	free_ir_op(op_riscv_lhu); op_riscv_lhu = NULL;
	free_ir_op(op_riscv_li); op_riscv_li = NULL;
	free_ir_op(op_riscv_lui); op_riscv_lui = NULL;
	free_ir_op(op_riscv_lw); op_riscv_lw = NULL;
	free_ir_op(op_riscv_lwu); op_riscv_lwu = NULL;
	free_ir_op(op_riscv_mul); op_riscv_mul = NULL;
	free_ir_op(op_riscv_mulh); op_riscv_mulh = NULL;
	free_ir_op(op_riscv_mulhu); op_riscv_mulhu = NULL;
	free_ir_op(op_riscv_mulw); op_riscv_mulw = NULL;
	free_ir_op(op_riscv_or); op_riscv_or = NULL;
	free_ir_op(op_riscv_ori); op_riscv_ori = NULL;
	free_ir_op(op_riscv_rem); op_riscv_rem = NULL;
	free_ir_op(op_riscv_remu); op_riscv_remu = NULL;
	free_ir_op(op_riscv_remuw); op_riscv_remuw = NULL;
	free_ir_op(op_riscv_remw); op_riscv_remw = NULL;
	free_ir_op(op_riscv_ret); op_riscv_ret = NULL;
	free_ir_op(op_riscv_sb); op_riscv_sb = NULL;
	free_ir_op(op_riscv_sd); op_riscv_sd = NULL;
	free_ir_op(op_riscv_sh); op_riscv_sh = NULL;
	free_ir_op(op_riscv_sll); op_riscv_sll = NULL;
	free_ir_op(op_riscv_slli); op_riscv_slli = NULL;
	free_ir_op(op_riscv_slliw); op_riscv_slliw = NULL;
	free_ir_op(op_riscv_sllw); op_riscv_sllw = NULL;
	free_ir_op(op_riscv_slt); op_riscv_slt = NULL;
	free_ir_op(op_riscv_sltiu); op_riscv_sltiu = NULL;
	free_ir_op(op_riscv_sltu); op_riscv_sltu = NULL;
	free_ir_op(op_riscv_sra); op_riscv_sra = NULL;
	free_ir_op(op_riscv_srai); op_riscv_srai = NULL;
	free_ir_op(op_riscv_sraiw); op_riscv_sraiw = NULL;
	free_ir_op(op_riscv_sraw); op_riscv_sraw = NULL;
	free_ir_op(op_riscv_srl); op_riscv_srl = NULL;
	free_ir_op(op_riscv_srli); op_riscv_srli = NULL;
	free_ir_op(op_riscv_srliw); op_riscv_srliw = NULL;
	free_ir_op(op_riscv_srlw); op_riscv_srlw = NULL;
	free_ir_op(op_riscv_sub); op_riscv_sub = NULL;
	free_ir_op(op_riscv_subw); op_riscv_subw = NULL;
	free_ir_op(op_riscv_sw); op_riscv_sw = NULL;
	free_ir_op(op_riscv_switch); op_riscv_switch = NULL;
	free_ir_op(op_riscv_xor); op_riscv_xor = NULL;
	free_ir_op(op_riscv_xori); op_riscv_xori = NULL;

}
