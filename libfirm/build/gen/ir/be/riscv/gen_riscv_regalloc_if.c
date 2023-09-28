/**
 * @file
 * @brief  The generated interface for the register allocator.
 *          Contains register classes and types and register constraints
 *          for all nodes where constraints were given in spec.
 * @note    DO NOT EDIT THIS FILE, your changes will be lost.
 *          Edit ./ir/be/riscv/riscv_spec.pl instead.
 *          created by: ./ir/be/scripts/generate_regalloc_if.pl ./ir/be/riscv/riscv_spec.pl build/gen/ir/be/riscv
 * $date    Wed Sep 27 23:17:11 2023
 */
#include "gen_riscv_regalloc_if.h"

#include "riscv_bearch_t.h"

const arch_register_req_t riscv_class_reg_req_gp = {
	.cls   = &riscv_reg_classes[CLASS_riscv_gp],
	.width = 1,
};
static const unsigned riscv_limited_gp_s0[] = { (1U << REG_GP_S0) };
const arch_register_req_t riscv_single_reg_req_gp_s0 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s0,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s1[] = { (1U << REG_GP_S1) };
const arch_register_req_t riscv_single_reg_req_gp_s1 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s1,
	.width   = 1,
};
static const unsigned riscv_limited_gp_a0[] = { (1U << REG_GP_A0) };
const arch_register_req_t riscv_single_reg_req_gp_a0 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_a0,
	.width   = 1,
};
static const unsigned riscv_limited_gp_a1[] = { (1U << REG_GP_A1) };
const arch_register_req_t riscv_single_reg_req_gp_a1 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_a1,
	.width   = 1,
};
static const unsigned riscv_limited_gp_a2[] = { (1U << REG_GP_A2) };
const arch_register_req_t riscv_single_reg_req_gp_a2 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_a2,
	.width   = 1,
};
static const unsigned riscv_limited_gp_a3[] = { (1U << REG_GP_A3) };
const arch_register_req_t riscv_single_reg_req_gp_a3 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_a3,
	.width   = 1,
};
static const unsigned riscv_limited_gp_a4[] = { (1U << REG_GP_A4) };
const arch_register_req_t riscv_single_reg_req_gp_a4 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_a4,
	.width   = 1,
};
static const unsigned riscv_limited_gp_a5[] = { (1U << REG_GP_A5) };
const arch_register_req_t riscv_single_reg_req_gp_a5 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_a5,
	.width   = 1,
};
static const unsigned riscv_limited_gp_zero[] = { (1U << REG_GP_ZERO) };
const arch_register_req_t riscv_single_reg_req_gp_zero = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_zero,
	.width   = 1,
};
static const unsigned riscv_limited_gp_ra[] = { (1U << REG_GP_RA) };
const arch_register_req_t riscv_single_reg_req_gp_ra = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_ra,
	.width   = 1,
};
static const unsigned riscv_limited_gp_sp[] = { (1U << REG_GP_SP) };
const arch_register_req_t riscv_single_reg_req_gp_sp = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_sp,
	.width   = 1,
};
static const unsigned riscv_limited_gp_gp[] = { (1U << REG_GP_GP) };
const arch_register_req_t riscv_single_reg_req_gp_gp = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_gp,
	.width   = 1,
};
static const unsigned riscv_limited_gp_tp[] = { (1U << REG_GP_TP) };
const arch_register_req_t riscv_single_reg_req_gp_tp = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_tp,
	.width   = 1,
};
static const unsigned riscv_limited_gp_t0[] = { (1U << REG_GP_T0) };
const arch_register_req_t riscv_single_reg_req_gp_t0 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_t0,
	.width   = 1,
};
static const unsigned riscv_limited_gp_t1[] = { (1U << REG_GP_T1) };
const arch_register_req_t riscv_single_reg_req_gp_t1 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_t1,
	.width   = 1,
};
static const unsigned riscv_limited_gp_t2[] = { (1U << REG_GP_T2) };
const arch_register_req_t riscv_single_reg_req_gp_t2 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_t2,
	.width   = 1,
};
static const unsigned riscv_limited_gp_a6[] = { (1U << REG_GP_A6) };
const arch_register_req_t riscv_single_reg_req_gp_a6 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_a6,
	.width   = 1,
};
static const unsigned riscv_limited_gp_a7[] = { (1U << REG_GP_A7) };
const arch_register_req_t riscv_single_reg_req_gp_a7 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_a7,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s2[] = { (1U << REG_GP_S2) };
const arch_register_req_t riscv_single_reg_req_gp_s2 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s2,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s3[] = { (1U << REG_GP_S3) };
const arch_register_req_t riscv_single_reg_req_gp_s3 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s3,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s4[] = { (1U << REG_GP_S4) };
const arch_register_req_t riscv_single_reg_req_gp_s4 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s4,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s5[] = { (1U << REG_GP_S5) };
const arch_register_req_t riscv_single_reg_req_gp_s5 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s5,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s6[] = { (1U << REG_GP_S6) };
const arch_register_req_t riscv_single_reg_req_gp_s6 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s6,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s7[] = { (1U << REG_GP_S7) };
const arch_register_req_t riscv_single_reg_req_gp_s7 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s7,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s8[] = { (1U << REG_GP_S8) };
const arch_register_req_t riscv_single_reg_req_gp_s8 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s8,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s9[] = { (1U << REG_GP_S9) };
const arch_register_req_t riscv_single_reg_req_gp_s9 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s9,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s10[] = { (1U << REG_GP_S10) };
const arch_register_req_t riscv_single_reg_req_gp_s10 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s10,
	.width   = 1,
};
static const unsigned riscv_limited_gp_s11[] = { (1U << REG_GP_S11) };
const arch_register_req_t riscv_single_reg_req_gp_s11 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_s11,
	.width   = 1,
};
static const unsigned riscv_limited_gp_t3[] = { (1U << REG_GP_T3) };
const arch_register_req_t riscv_single_reg_req_gp_t3 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_t3,
	.width   = 1,
};
static const unsigned riscv_limited_gp_t4[] = { (1U << REG_GP_T4) };
const arch_register_req_t riscv_single_reg_req_gp_t4 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_t4,
	.width   = 1,
};
static const unsigned riscv_limited_gp_t5[] = { (1U << REG_GP_T5) };
const arch_register_req_t riscv_single_reg_req_gp_t5 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_t5,
	.width   = 1,
};
static const unsigned riscv_limited_gp_t6[] = { (1U << REG_GP_T6) };
const arch_register_req_t riscv_single_reg_req_gp_t6 = {
	.cls     = &riscv_reg_classes[CLASS_riscv_gp],
	.limited = riscv_limited_gp_t6,
	.width   = 1,
};


arch_register_class_t riscv_reg_classes[] = {
	{
		.name      = "riscv_gp",
		.mode      = NULL,
		.regs      = &riscv_registers[REG_S0],
		.class_req = &riscv_class_reg_req_gp,
		.index     = CLASS_riscv_gp,
		.n_regs    = 32,
	},

};

/** The array of all registers in the riscv architecture, sorted by its global index.*/
const arch_register_t riscv_registers[] = {
	{
		.name            = "s0",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s0,
		.index           = REG_GP_S0,
		.global_index    = REG_S0,
		.dwarf_number    = 0,
		.encoding        = 8,
		.is_virtual      = false,
		.is_compressible = true,
	},
	{
		.name            = "s1",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s1,
		.index           = REG_GP_S1,
		.global_index    = REG_S1,
		.dwarf_number    = 0,
		.encoding        = 9,
		.is_virtual      = false,
		.is_compressible = true,
	},
	{
		.name            = "a0",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_a0,
		.index           = REG_GP_A0,
		.global_index    = REG_A0,
		.dwarf_number    = 0,
		.encoding        = 10,
		.is_virtual      = false,
		.is_compressible = true,
	},
	{
		.name            = "a1",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_a1,
		.index           = REG_GP_A1,
		.global_index    = REG_A1,
		.dwarf_number    = 0,
		.encoding        = 11,
		.is_virtual      = false,
		.is_compressible = true,
	},
	{
		.name            = "a2",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_a2,
		.index           = REG_GP_A2,
		.global_index    = REG_A2,
		.dwarf_number    = 0,
		.encoding        = 12,
		.is_virtual      = false,
		.is_compressible = true,
	},
	{
		.name            = "a3",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_a3,
		.index           = REG_GP_A3,
		.global_index    = REG_A3,
		.dwarf_number    = 0,
		.encoding        = 13,
		.is_virtual      = false,
		.is_compressible = true,
	},
	{
		.name            = "a4",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_a4,
		.index           = REG_GP_A4,
		.global_index    = REG_A4,
		.dwarf_number    = 0,
		.encoding        = 14,
		.is_virtual      = false,
		.is_compressible = true,
	},
	{
		.name            = "a5",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_a5,
		.index           = REG_GP_A5,
		.global_index    = REG_A5,
		.dwarf_number    = 0,
		.encoding        = 15,
		.is_virtual      = false,
		.is_compressible = true,
	},
	{
		.name            = "zero",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_zero,
		.index           = REG_GP_ZERO,
		.global_index    = REG_ZERO,
		.dwarf_number    = 0,
		.encoding        = 0,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "ra",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_ra,
		.index           = REG_GP_RA,
		.global_index    = REG_RA,
		.dwarf_number    = 0,
		.encoding        = 1,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "sp",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_sp,
		.index           = REG_GP_SP,
		.global_index    = REG_SP,
		.dwarf_number    = 0,
		.encoding        = 2,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "gp",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_gp,
		.index           = REG_GP_GP,
		.global_index    = REG_GP,
		.dwarf_number    = 0,
		.encoding        = 3,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "tp",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_tp,
		.index           = REG_GP_TP,
		.global_index    = REG_TP,
		.dwarf_number    = 0,
		.encoding        = 4,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "t0",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_t0,
		.index           = REG_GP_T0,
		.global_index    = REG_T0,
		.dwarf_number    = 0,
		.encoding        = 5,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "t1",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_t1,
		.index           = REG_GP_T1,
		.global_index    = REG_T1,
		.dwarf_number    = 0,
		.encoding        = 6,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "t2",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_t2,
		.index           = REG_GP_T2,
		.global_index    = REG_T2,
		.dwarf_number    = 0,
		.encoding        = 7,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "a6",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_a6,
		.index           = REG_GP_A6,
		.global_index    = REG_A6,
		.dwarf_number    = 0,
		.encoding        = 16,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "a7",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_a7,
		.index           = REG_GP_A7,
		.global_index    = REG_A7,
		.dwarf_number    = 0,
		.encoding        = 17,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s2",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s2,
		.index           = REG_GP_S2,
		.global_index    = REG_S2,
		.dwarf_number    = 0,
		.encoding        = 18,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s3",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s3,
		.index           = REG_GP_S3,
		.global_index    = REG_S3,
		.dwarf_number    = 0,
		.encoding        = 19,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s4",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s4,
		.index           = REG_GP_S4,
		.global_index    = REG_S4,
		.dwarf_number    = 0,
		.encoding        = 20,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s5",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s5,
		.index           = REG_GP_S5,
		.global_index    = REG_S5,
		.dwarf_number    = 0,
		.encoding        = 21,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s6",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s6,
		.index           = REG_GP_S6,
		.global_index    = REG_S6,
		.dwarf_number    = 0,
		.encoding        = 22,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s7",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s7,
		.index           = REG_GP_S7,
		.global_index    = REG_S7,
		.dwarf_number    = 0,
		.encoding        = 23,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s8",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s8,
		.index           = REG_GP_S8,
		.global_index    = REG_S8,
		.dwarf_number    = 0,
		.encoding        = 24,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s9",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s9,
		.index           = REG_GP_S9,
		.global_index    = REG_S9,
		.dwarf_number    = 0,
		.encoding        = 25,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s10",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s10,
		.index           = REG_GP_S10,
		.global_index    = REG_S10,
		.dwarf_number    = 0,
		.encoding        = 26,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "s11",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_s11,
		.index           = REG_GP_S11,
		.global_index    = REG_S11,
		.dwarf_number    = 0,
		.encoding        = 27,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "t3",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_t3,
		.index           = REG_GP_T3,
		.global_index    = REG_T3,
		.dwarf_number    = 0,
		.encoding        = 28,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "t4",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_t4,
		.index           = REG_GP_T4,
		.global_index    = REG_T4,
		.dwarf_number    = 0,
		.encoding        = 29,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "t5",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_t5,
		.index           = REG_GP_T5,
		.global_index    = REG_T5,
		.dwarf_number    = 0,
		.encoding        = 30,
		.is_virtual      = false,
		.is_compressible = false,
	},
	{
		.name            = "t6",
		.cls             = &riscv_reg_classes[CLASS_riscv_gp],
		.single_req      = &riscv_single_reg_req_gp_t6,
		.index           = REG_GP_T6,
		.global_index    = REG_T6,
		.dwarf_number    = 0,
		.encoding        = 31,
		.is_virtual      = false,
		.is_compressible = false,
	},

};

/**
 * Initializes riscv register classes.
 */
void riscv_register_init(void)
{
	riscv_reg_classes[CLASS_riscv_gp].mode = mode_rv_gp;

}
