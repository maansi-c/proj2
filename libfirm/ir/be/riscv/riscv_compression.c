/*
 * This file is part of libFirm.
 */

/**
 * @file
 * @brief       This file supports compression-aware register allocation for the RISC-V backend.
 * @author      Maximilian Stemmer-Grabow
 */

#include "riscv_compression.h"

#include "firm_types.h"
#include "irop_t.h"
#include "irnode_t.h"
#include "bearch.h"
#include "becopyopt_t.h"
#include "benode.h"
#include "irprintf.h"
#include "iredges.h"

#include "riscv_nodes_attr.h"
#include "gen_riscv_new_nodes.h"
#include "gen_riscv_regalloc_if.h"

static inline bool fits_imm_n(int32_t value, int n)
{
	return value >= -(1 << (n - 1)) && value < (1 << (n - 1));
}

compression_req_t riscv_get_op_compression_requirements(ir_node const *const node)
{
	compression_req_t requirement = {
		.req_type    = comp_req_unknown,
		.n_op1       = -1,
		.n_op2       = -1,
		.n_res       =  0,  // Refers to first result by default
		.commutative = false
	};

	/* Handle generic Firm nodes */

	if (be_is_Copy(node)) {
		// mvs can always be compressed
		requirement.req_type = comp_req_always;
	}
	else if (be_is_IncSP(node)) {
		int const offset = -be_get_IncSP_offset(node);
		// Offset fits either into direct addi immediate or scaled offset for C.ADDI16SP
		bool scaled_offset_fits = ((offset & 0xF) == 0) && fits_imm_n((offset >> 4), 6);

		if (fits_imm_n(offset, 6) || scaled_offset_fits) {
			requirement.req_type = comp_req_always;
		} else {
			requirement.req_type = comp_req_never;
		}
	}
	else if (be_is_Perm(node)) {
		// Due to the way these are generated using xor instructions,
		// these always fulfill the 2-address requirement, so it is omitted here

		// TODO This should emit a register subset requirement, but among the two output values of this node
		// requirement = comp_req_register_subset;
	}

	/* Handle RISC-V-specific nodes */

	if (is_riscv_irn(node)) {

		int opcode = get_riscv_irn_opcode(node);

		switch(opcode) {
			case iro_riscv_lw: {
				requirement.req_type = comp_req_register_subset;
				requirement.n_op1    = n_riscv_lw_base;
				break;
			}

			case iro_riscv_sw: {
				requirement.req_type = comp_req_register_subset;
				requirement.n_op1    = n_riscv_sw_base;
				requirement.n_op2    = n_riscv_sw_value;
				requirement.n_res    = -1;
				break;
			}

			case iro_riscv_j:
			case iro_riscv_jal:
				// For these jumps, the offset is 11 bits long and sign-extended,
				// the LSB is omitted (like offset[11:1]), this can target a 2KiB range.

				// This offset cannot be easily checked here, so we assume that most
				// jumps are inside the range to be compressed

				// Fallthrough
			case iro_riscv_ijmp:     // This is a register jump (emitted as jr)
			case iro_riscv_jalr:
			case iro_riscv_ret:      // Pseudoinstruction for jalr
			case iro_riscv_switch: { // This is emitted as a register-based jump, so the same rules apply

				requirement.req_type = comp_req_always;
				requirement.n_res    = -1;
				break;
			}

			// Branch instructions are compressible with the C.BEQZ and C.BNEZ compressed instructions

			// These require:
			// - A value from the subset of compressible registers
			// - An offset that fits into 8 bits (the LSB of the offset is omitted, behaves like offset[8:1]),
			//   it is sign-extended, this allows for a range of +/-256B
			//   We cannot check this here, so we assume it is sufficient to indicate it might be compressible.
			case iro_riscv_bcc: {

				// Check whether we might be comparing against a zero register
				const arch_register_t* zero_reg = &riscv_registers[REG_ZERO];

				const arch_register_t* reg_right = arch_get_irn_register_in(node, n_riscv_bcc_right);
				bool right_zero = (reg_right && reg_right == zero_reg);

				if (!right_zero) {
					requirement.req_type = comp_req_never;
					break;
				}

				// Check whether we have the correct comparison operator
				riscv_cond_t cond = get_riscv_cond_attr_const(node)->cond;

				if (cond == riscv_cc_eq || cond == riscv_cc_ne) {
					// We have a compressible jump if the offset is small enough
					// We cannot check this here, so we assume it might be
					requirement.req_type = comp_req_register_subset;
					requirement.n_op1    = n_riscv_bcc_left;
					requirement.n_res    = -1;
					break;
				} else {
					requirement.req_type = comp_req_never;
					break;
				}
			}

			case iro_riscv_lui: {
				riscv_immediate_attr_t const *const imm = get_riscv_immediate_attr_const(node);

				// Check whether the value loaded here is local in the current compilation unit.
				// In this case, the value can be directly inserted before the executable is being linked,
				// and the resulting load may be compressible. If not, the value will be inserted at link-time
				// and the load instructions may never be compressed.
				ir_entity* entity = imm->ent;
				if (entity && !entity_has_definition(entity)) {
					requirement.req_type = comp_req_never;
					break;
				}

				int32_t val = imm->val;
				int32_t upper_val = val >> 12; // The lower 12 bits are ignored by lui

				requirement.req_type = fits_imm_n(upper_val, 6) ? comp_req_always : comp_req_never;
				break;
			}

			case iro_riscv_addi: {
				// addi is restricted to a 6-bit immediate
				// and requires one operand to be in same register as the target
				riscv_immediate_attr_t const *const imm = get_riscv_immediate_attr_const(node);
				int32_t val = imm->val;

				// See above for rationale
				ir_entity* entity = imm->ent;
				if (entity && !entity_has_definition(entity)) {
					requirement.req_type = comp_req_never;
					break;
				}

				if (fits_imm_n(val, 6)) {
					requirement.req_type = comp_req_2addr;
					requirement.n_op1    = n_riscv_addi_left;
				} else {
					requirement.req_type = comp_req_never;
				}
				break;
			}

			// slli is a special case as compared to the other shift instructions as it
			// can be used with any source/target register (provided they are identical)
			case iro_riscv_slli: {
				// The shift amount field has the same length as in uncompressed instructions
				// (actually, it is even 1 bit longer, but those values have special meaning in RV128;
				// for RV32C, the MSB is reserved)

				// In this case, the 2-address requirement is actually a single-address requirement,
				// indicating that the one source value must be from the same register as the result.
				requirement.req_type = comp_req_2addr;
				requirement.n_op1    = n_riscv_slli_left;
				break;
			}

			case iro_riscv_srli:
			case iro_riscv_srai: {
				// The shift amount field has the same length as in uncompressed instructions
				// (actually, it is even 1 bit longer, but those values have special meaning in RV128)
				requirement.req_type = comp_req_2addr_register_subset;
				requirement.n_op1    = n_riscv_srli_left; // == n_riscv_srai_left
				break;
			}

			case iro_riscv_andi: {
				riscv_immediate_attr_t const *const imm = get_riscv_immediate_attr_const(node);
				int32_t val = imm->val;

				if (fits_imm_n(val, 6)) {
					requirement.req_type = comp_req_2addr_register_subset;
					requirement.n_op1    = n_riscv_andi_left;
				} else {
					requirement.req_type = comp_req_never;
				}
				break;
			}

			case iro_riscv_add: {
				requirement.req_type    = comp_req_2addr;
				requirement.n_op1       = n_riscv_add_left;
				requirement.n_op2       = n_riscv_add_right;
				requirement.commutative = true;
				break;
			}

			case iro_riscv_and:
			case iro_riscv_or:
			case iro_riscv_xor:
				requirement.commutative = true;
				// Fallthrough
			case iro_riscv_sub: {
				requirement.req_type    = comp_req_2addr_register_subset;

				// Indices are he same for and, or, xor, and sub
				requirement.n_op1       = n_riscv_sub_left;
				requirement.n_op2       = n_riscv_sub_right;
				break;
			}

			/* Stack manipulation */
			case iro_riscv_SubSP: {
				requirement.req_type    = comp_req_register_subset;
				// This operand is the register with the size subtracted from the sp
				requirement.n_op1       = n_riscv_SubSP_size;
				break;
			}
			case iro_riscv_SubSPimm: {
				// Emitted as addi, so it's similar, but without free register choice
				// RVC has a C.ADDI16SP instruction for this, which scales its immediate
				// by 16 and has a 6 bit immediate field
				riscv_immediate_attr_t const *const imm = get_riscv_immediate_attr_const(node);
				int32_t val = imm->val;

				bool scaled_imm_fits = ((val & 0xF) == 0) && fits_imm_n((val >> 4), 6);

				if (fits_imm_n(val, 6) || scaled_imm_fits) {
					requirement.req_type = comp_req_always;
				} else {
					requirement.req_type = comp_req_never;
				}
				break;
			}

			/* Incompressible RISC-V operations */
			case iro_riscv_div:   // Division
			case iro_riscv_divu:
			case iro_riscv_mul:   // Multiplication
			case iro_riscv_mulh:
			case iro_riscv_mulhu:
			case iro_riscv_rem:   // Remainders
			case iro_riscv_remu:
			case iro_riscv_lb:    // Loads for <32 bits
			case iro_riscv_lbu:
			case iro_riscv_lh:
			case iro_riscv_lhu:
			case iro_riscv_sb:    // Stores for <32 bits
			case iro_riscv_sh:
			case iro_riscv_ori:
			case iro_riscv_sll:   // Shifts
			case iro_riscv_srl:
			case iro_riscv_sra:
			case iro_riscv_slt:   // Set instructions
			case iro_riscv_sltu:
			case iro_riscv_sltiu:
				requirement.req_type = comp_req_never;
				break;

			default:
				requirement.req_type = comp_req_unknown;
				break;
		}
	}

	return requirement;
}

int riscv_get_use_count(const ir_node *node)
{
	// We are using the out edges, so verify they are correct
	assure_irg_properties(node->irg, IR_GRAPH_PROPERTY_CONSISTENT_OUT_EDGES);

	int count = 0;
	foreach_out_edge(node, edge) {
		count++;
	}

	return count;
}

static bool riscv_register_is_compressible(const arch_register_t *const reg)
{
	if (!reg) return false;
	return reg->is_compressible;
}

static bool riscv_is_2addr_form(const ir_node *const node)
{
	compression_req_t compression_req = riscv_get_op_compression_requirements(node);

	const arch_register_t *rs1 = NULL;
	if (compression_req.n_op1 >= 0) rs1 = arch_get_irn_register_in(node, compression_req.n_op1);

	const arch_register_t *rs2 = NULL;
	if (compression_req.n_op2 >= 0) rs2 = arch_get_irn_register_in(node, compression_req.n_op2);

	const arch_register_t * rd = arch_get_irn_register(node);

	// First, check 2-address requirement with commutativity
	if (compression_req.commutative) {
		// Is any one of the two source registers the same as the result register?
		return (rs1 && rs1 == rd) || (rs2 && rs2 == rd);
	}

	// Without commutativity, we need to preserve the order
	if (rs1) return rs1 == rd;
	if (rs2) return rs2 == rd;

	return false;
}

static bool riscv_is_register_subset_form(const ir_node *const node)
{
	compression_req_t requirement = riscv_get_op_compression_requirements(node);

	// Check if the result is in a compressible register
	if (requirement.n_res >= 0) {
		const arch_register_t *rd = arch_get_irn_register_out(node, requirement.n_res);
		if (!rd || !riscv_register_is_compressible(rd))
			return false;
	}

	// Now also check operands
	if (requirement.n_op1 >= 0) {
		const arch_register_t *rs1 = arch_get_irn_register_in(node, requirement.n_op1);
		if (!rs1 || !riscv_register_is_compressible(rs1))
			return false;
	}
	if (requirement.n_op2 >= 0) {
		const arch_register_t *rs2 = arch_get_irn_register_in(node, requirement.n_op2);
		if (!rs2 || !riscv_register_is_compressible(rs2))
			return false;
	}

	return true;
}

bool riscv_is_compressible(const ir_node *const node)
{
	compression_req_t requirement = riscv_get_op_compression_requirements(node);

	switch (requirement.req_type) {

		case comp_req_unknown:
		case comp_req_never:
			return false;

		case comp_req_always:
			return true;

		case comp_req_register_subset:
			return riscv_is_register_subset_form(node);

		case comp_req_2addr:
			return riscv_is_2addr_form(node);

		case comp_req_2addr_register_subset:
			return riscv_is_register_subset_form(node) && riscv_is_2addr_form(node);

		default:
			return false;
	}
}
