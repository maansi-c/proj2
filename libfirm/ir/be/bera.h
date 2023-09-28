/*
 * This file is part of libFirm.
 * Copyright (C) 2012 University of Karlsruhe.
 */

/**
 * @file
 * @brief       Base routines for register allocation.
 * @author      Sebastian Hack
 * @date        13.01.2005
 */
#ifndef FIRM_BE_BERA_H
#define FIRM_BE_BERA_H

#include "firm_types.h"
#include "be_types.h"

#include <stdbool.h>

/**
 * A type of compression requirement that needs to be fulfilled
 * in order for a node to be compressed.
 */
typedef enum compression_req_type_t {
	comp_req_unknown,

	comp_req_never,
	comp_req_always,

	comp_req_2addr, /**< Compressible if */
	comp_req_register_subset,
	comp_req_2addr_register_subset,
} compression_req_type_t;

/**
 * Requirements that need to be fulfilled in order for an operation to be compressible.
 */
typedef struct compression_req_t {
	compression_req_type_t req_type; /**< The type of compression requirement this describes. */

	int n_op1; /**< Index of a first operand for this requirement, ignored if <0. */
	int n_op2; /**< Index of a second operand for this requirement, ignored if <0. */
	int n_res; /**< Index of the result this requirement applies to, ignore if <0. */

	bool commutative : 1; /**< Set if the requirement can be swapped between the operands. */
} compression_req_t;

struct regalloc_if_t {
	unsigned spill_cost;  /**< cost for a spill node */
	unsigned reload_cost; /**< cost for a reload node */

	/** mark node as rematerialized */
	void (*mark_remat)(ir_node *node);

	/**
	 * Create a spill instruction. We assume that spill instructions do not need
	 * any additional registers and do not affect cpu-flags in any way.
	 * Construct a sequence of instructions after @p after (the resulting nodes
	 * are already scheduled).
	 * Returns a mode_M value which is used as input for a reload instruction.
	 */
	ir_node *(*new_spill)(ir_node *value, ir_node *after);

	/**
	 * Create a reload instruction. We assume that reload instructions do not
	 * need any additional registers and do not affect cpu-flags in any way.
	 * Constructs a sequence of instruction before @p before (the resulting
	 * nodes are already scheduled). A rewiring of users is not performed in
	 * this function.
	 * Returns a value representing the restored value.
	 */
	ir_node *(*new_reload)(ir_node *value, ir_node *spilled_value,
	                       ir_node *before);

	/**
	 * Ask the backend to fold a reload at operand @p i of @p irn. This can
	 * be done by targets that support memory addressing modes.
	 */
	void (*perform_memory_operand)(ir_node *irn, unsigned i);

    /**
     * Get the requirements for compressing the operation represented by @p irn.
     *
     * This only applies to instruction sets which include compressed variants of
     * common instructions.
     */
    compression_req_t (*get_op_compression_requirements)(const ir_node *irn);
};



/**
 * Do register allocation with currently selected register allocator
 */
void be_allocate_registers(ir_graph *irg, const regalloc_if_t *regif);

typedef void (*allocate_func)(ir_graph *irg, const regalloc_if_t *regif);

void be_register_allocator(const char *name, allocate_func allocator);

#endif
