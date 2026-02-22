#ifndef UTIL_H
#define UTIL_H

#include "nodes/pg_list.h"
#include "nodes/value.h"

List *
split_comma_c_string(char *value)
{
	List	   *result = NIL;
	char	   *token = strtok(value, ",");

	while (token != NULL)
	{
		result = lappend(result, makeString(token));
		token = strtok(NULL, ",");
	}
	return result;
}

/* ------------------------ UDF ---------------------------- */

typedef struct
{
	Datum		value;
	bool		isnull;
}			UdfResult;

static UdfResult call_udf_function(const char *funcName, Oid *argTypes,
								   int nargs, Datum *args, bool *nulls)
{
	FmgrInfo	fmgrInfo;

	LOCAL_FCINFO(fcinfo, FUNC_MAX_ARGS);
	Oid			funcOid;
	UdfResult	result;

	funcOid = LookupFuncName(list_make1(makeString(funcName)), nargs, argTypes,
							 false);
	if (!OidIsValid(funcOid))
		elog(ERROR, "Function %s not found", funcName);

	fmgr_info(funcOid, &fmgrInfo);
	InitFunctionCallInfoData(*fcinfo, &fmgrInfo, nargs, InvalidOid, NULL, NULL);

	for (int i = 0; i < nargs; i++)
	{
		fcinfo->args[i].value = args[i];
		fcinfo->args[i].isnull = nulls ? nulls[i] : false;
	}

	result.value = FunctionCallInvoke(fcinfo);
	result.isnull = fcinfo->isnull;

	if (result.isnull)
		elog(DEBUG2, "%s returned NULL", funcName);

	return result;
}

/* ------------------------ UDF end ------------------------ */

#endif
