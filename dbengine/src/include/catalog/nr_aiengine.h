#ifndef NR_AIENGINE_H
#define NR_AIENGINE_H

#include "catalog/genbki.h"
#include "catalog/nr_aiengine_d.h"

CATALOG(nr_aiengine,9600,NrAiengineRelationId)
{
	Oid			oid;
	NameData	aiename;
	int32		aieport BKI_DEFAULT(0);
	Oid			aieconn BKI_DEFAULT(0);
#ifdef CATALOG_VARLEN			/* variable-length fields start here */
	text		aieaddr BKI_DEFAULT(_null_); /* aiengine address */
#endif
} FormData_nr_aiengine;

typedef FormData_nr_aiengine * Form_nr_aiengine;

/* Indexes */
DECLARE_UNIQUE_INDEX(nr_aiengine_oid_index, 9601, NrAiengineOidIndexId, on nr_aiengine using btree(oid oid_ops));
#endif
