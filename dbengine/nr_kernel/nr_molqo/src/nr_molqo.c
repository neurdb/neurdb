/*-------------------------------------------------------------------------
 *
 * nr_molqo.c
 * 		NeurDB MoLQO (Mixture of Learned Query Optimizer) Extension
 *
 * Part of the NeurDB kernel extensions. This extension hooks into PostgreSQL's
 * parse analysis phase to send SQL queries to an external MoLQO optimization
 * server and receives back optimized SQL with hints and configurations.
 *
 * Usage:
 *   CREATE EXTENSION nr_molqo;    -- Install the extension
 *   SET enable_molqo = on;        -- Enable MoLQO optimization
 *   SET molqo.server_url = '...'; -- Configure server URL
 *
 * Copyright (c) 2025, NeurDB Project
 *
 * IDENTIFICATION
 *	  nr_kernel/nr_molqo/src/nr_molqo.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres.h"
#include "fmgr.h"
#include "access/xact.h"
#include "catalog/namespace.h"
#include "commands/dbcommands.h"
#include "executor/spi.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "nodes/print.h"
#include "parser/analyze.h"
#include "parser/parse_node.h"
#include "parser/parser.h"
#include "tcop/tcopprot.h"
#include "utils/builtins.h"
#include "utils/guc.h"
#include "utils/memutils.h"
#include "utils/snapmgr.h"
#include <time.h>
#include <unistd.h>

#ifdef PG_MODULE_MAGIC
PG_MODULE_MAGIC;
#endif

/* MoLQO GUC variables */
static bool enable_molqo = false;     /* Main enable/disable switch */
static char *molqo_server_url = NULL; /* MoLQO server URL */

/* Hook storage */
static post_parse_analyze_hook_type prev_post_parse_analyze_hook = NULL;

/* Forward declarations */
void _PG_init(void);
void _PG_fini(void);
static void query_optimizer_post_parse_analyze(ParseState *pstate, Query *query,
                                               JumbleState *jstate);
static char *call_external_optimizer(const char *original_sql);
static char *send_http_request(const char *url, const char *sql_data);
static void execute_optimized_sql(const char *optimized_sql);

/*
 * Module load callback
 */
void _PG_init(void) {
	/* Main MoLQO enable/disable switch */
    DefineCustomBoolVariable(
        "enable_molqo", "Enable MoLQO mixture of learned query optimizer",
        "When enabled, queries will be optimized using MoLQO server.",
        &enable_molqo, false, PGC_USERSET, 0, NULL, NULL, NULL);

	/* MoLQO configuration parameters */
    DefineCustomStringVariable(
        "molqo.server_url", "MoLQO server URL",
        "URL of the MoLQO (Mixture of Learned Query Optimizer) server.",
        &molqo_server_url, "http://localhost:8080/optimize", PGC_SUSET, 0, NULL,
        NULL, NULL);

	MarkGUCPrefixReserved("molqo");

	/* Install our hook */
	prev_post_parse_analyze_hook = post_parse_analyze_hook;
	post_parse_analyze_hook = query_optimizer_post_parse_analyze;

    elog(LOG, "MoLQO (Mixture of Learned Query Optimizer) extension loaded");
}

/*
 * Module unload callback
 */
void _PG_fini(void) {
	/* Restore previous hook */
	post_parse_analyze_hook = prev_post_parse_analyze_hook;
	
    elog(LOG, "MoLQO (Mixture of Learned Query Optimizer) extension unloaded");
}

/*
 * Post parse analyze hook - called after SQL parsing is complete
 */
static void query_optimizer_post_parse_analyze(ParseState *pstate, Query *query,
                                               JumbleState *jstate) {
    const char *original_sql;
	char *optimized_sql;
	
	/* Call previous hook if exists */
	if (prev_post_parse_analyze_hook)
		prev_post_parse_analyze_hook(pstate, query, jstate);

	/* Skip if MoLQO is disabled */
    if (!enable_molqo) return;

	/* Skip if no MoLQO server URL configured */
    if (!molqo_server_url || strlen(molqo_server_url) == 0) return;

    /* Only optimize SELECT queries */
    /* Skip CMD_UTILITY to avoid duplicate processing (EXPLAIN will recursively process SELECT) */
    if (query->commandType != CMD_SELECT)
		return;

	/* Get the original SQL from parse state */
	original_sql = pstate->p_sourcetext;
    if (!original_sql || strlen(original_sql) == 0) return;

    /* Debug: log that we're processing this query (removed detailed log) */

	/* Call external optimization server */
	PG_TRY();
	{
		optimized_sql = call_external_optimizer(original_sql);
		
        /* If optimization failed, log fallback to cost-based optimizer */
        if (!optimized_sql) {
            elog(INFO, "MoLQO: sql -> moqoe -> cost-based optimizer");  // Fallback on HTTP error
        }

        if (optimized_sql && strcmp(optimized_sql, original_sql) != 0) {
            /* Check format and process accordingly */
            if (strstr(optimized_sql, "/*+")) {
                /* 
                 * pg_hint_plan format: Replace source text
                 * Since our hook runs before planner_hook, pg_hint_plan will
                 * read the modified p_sourcetext and parse the hints!
                 */
                /* Extract hints from the comment for display */
                char *hint_start = strstr(optimized_sql, "/*+");
                char *hint_end = strstr(hint_start, "*/");
                if (hint_start && hint_end) {
                    int hint_len = hint_end - hint_start + 2;
                    char *hint_text = palloc(hint_len + 1);
                    memcpy(hint_text, hint_start, hint_len);
                    hint_text[hint_len] = '\0';
                    
                    elog(INFO, "┌─────────────────────────────────────────────────");
                    elog(INFO, "│ MoLQO Optimization Applied");
                    elog(INFO, "│ Server: %s", molqo_server_url);
                    elog(INFO, "└─────────────────────────────────────────────────");
                    
                    pfree(hint_text);
                }
                pstate->p_sourcetext = optimized_sql;
            } else {
                /* SET command format: Apply settings directly using GUC API */
                elog(INFO, "┌─────────────────────────────────────────────────");
                elog(INFO, "│ MoLQO Optimization Applied");
                elog(INFO, "│ Server: %s", molqo_server_url);
                elog(INFO, "└─────────────────────────────────────────────────");
                
			execute_optimized_sql(optimized_sql);
		}
		}
		
        if (optimized_sql) pfree(optimized_sql);
	}
	PG_CATCH();
	{
		/* Reset error state and continue with original query */
		FlushErrorState();
	}
	PG_END_TRY();
}

/*
 * Call external optimizer server
 */
static char *call_external_optimizer(const char *original_sql) {
	char *result;
	
    if (!original_sql) return NULL;
		
	/* Send HTTP request to MoLQO server */
	result = send_http_request(molqo_server_url, original_sql);
	
	return result;
}

/*
 * Send HTTP request to external server using curl command
 * Simple implementation using system call
 */
static char *send_http_request(const char *url, const char *sql_data) {
    char *result = NULL;
    char *json_payload = NULL;
    char *curl_cmd = NULL;
    char *escaped_sql = NULL;
    FILE *fp;
    char response_buf[8192];
    size_t response_len = 0;
    
    /* Escape SQL for JSON (replace " with \") */
	size_t sql_len = strlen(sql_data);
    escaped_sql = palloc(sql_len * 2 + 1);
    char *out = escaped_sql;
    for (const char *in = sql_data; *in; in++) {
        if (*in == '"' || *in == '\\' || *in == '\n' || *in == '\r' || *in == '\t') {
            *out++ = '\\';
            if (*in == '\n') *out++ = 'n';
            else if (*in == '\r') *out++ = 'r';
            else if (*in == '\t') *out++ = 't';
            else *out++ = *in;
        } else {
            *out++ = *in;
        }
    }
    *out = '\0';
    
    /* Build JSON payload */
    json_payload = palloc(strlen(escaped_sql) + 256);
    snprintf(json_payload, strlen(escaped_sql) + 256, "{\"sql\":\"%s\"}", escaped_sql);
    
    /* Use temporary file to avoid shell escaping issues with single quotes in SQL */
    char tmp_file[256];
    snprintf(tmp_file, sizeof(tmp_file), "/tmp/molqo_%d_%lu.json", 
             getpid(), (unsigned long)time(NULL));
    
    FILE *tmp_fp = fopen(tmp_file, "w");
    if (!tmp_fp) {
        elog(WARNING, "MoLQO: Failed to create temporary file: %s", tmp_file);
        pfree(escaped_sql);
        pfree(json_payload);
        return NULL;
    }
    fwrite(json_payload, 1, strlen(json_payload), tmp_fp);
    fclose(tmp_fp);
    
    /* Build curl command using temporary file (@filename syntax) */
    curl_cmd = palloc(strlen(url) + strlen(tmp_file) + 256);
    snprintf(curl_cmd, strlen(url) + strlen(tmp_file) + 256,
             "curl -s -X POST -H 'Content-Type: application/json' -d @%s '%s' 2>&1",
             tmp_file, url);
    
    /* Execute curl command and capture output */
    fp = popen(curl_cmd, "r");
    if (!fp) {
        elog(WARNING, "MoLQO: Failed to execute curl command");
        unlink(tmp_file);
        pfree(escaped_sql);
        pfree(json_payload);
        pfree(curl_cmd);
        return NULL;
    }
    
    response_len = fread(response_buf, 1, sizeof(response_buf) - 1, fp);
    response_buf[response_len] = '\0';
    int curl_status = pclose(fp);
    
    /* Clean up temporary file */
    unlink(tmp_file);
    
    if (curl_status != 0) {
        elog(WARNING, "MoLQO: Curl failed with status %d: %s", curl_status, response_buf);
        pfree(escaped_sql);
        pfree(json_payload);
        pfree(curl_cmd);
        return NULL;
    }
    
    if (response_len == 0) {
        elog(WARNING, "MoLQO: Empty response from server");
        pfree(escaped_sql);
        pfree(json_payload);
        pfree(curl_cmd);
        return NULL;
    }
    
    /* Parse JSON response to extract optimized_sql and expert_name */
    char *expert_name = NULL;
    char *opt_sql_start = strstr(response_buf, "\"optimized_sql\"");
    if (opt_sql_start) {
        opt_sql_start = strchr(opt_sql_start, ':');
        if (opt_sql_start) {
            opt_sql_start = strchr(opt_sql_start, '"');
            if (opt_sql_start) {
                opt_sql_start++; /* Skip opening quote */
                char *opt_sql_end = strchr(opt_sql_start, '"');
                if (opt_sql_end) {
                    size_t opt_len = opt_sql_end - opt_sql_start;
                    result = palloc(opt_len + 1);
                    memcpy(result, opt_sql_start, opt_len);
                    result[opt_len] = '\0';
                }
            }
        }
    }
    
    /* Extract expert_name from JSON response */
    char *expert_start = strstr(response_buf, "\"expert_name\"");
    if (expert_start) {
        expert_start = strchr(expert_start, ':');
        if (expert_start) {
            expert_start = strchr(expert_start, '"');
            if (expert_start) {
                expert_start++; /* Skip opening quote */
                char *expert_end = strchr(expert_start, '"');
                if (expert_end) {
                    size_t expert_len = expert_end - expert_start;
                    expert_name = palloc(expert_len + 1);
                    memcpy(expert_name, expert_start, expert_len);
                    expert_name[expert_len] = '\0';
                }
            }
        }
    }
    
    /* Log concise message: sql -> moqoe -> expertname */
    if (expert_name && strlen(expert_name) > 0) {
        elog(INFO, "MoLQO: sql -> moqoe -> %s", expert_name);
    } else {
            elog(INFO, "MoLQO: sql -> moqoe -> cost-based optimizer");  // Fallback to cost-based optimizer on error
    }
    
    /* Fallback: return original SQL if extraction failed */
    if (!result) {
        result = pstrdup(sql_data);
    }
    
    /* Clean up */
    pfree(escaped_sql);
    pfree(json_payload);
    pfree(curl_cmd);
    if (expert_name) {
        pfree(expert_name);
	}
	
	return result;
}

/*
 * Execute optimized SQL commands (typically SET commands for hints)
 * Parse SET commands and apply them directly using GUC API
 */
static void execute_optimized_sql(const char *optimized_sql) {
	char *sql_copy;
	char *current_pos;
    char *set_start, *set_end;
	
    if (!optimized_sql) return;
		
	sql_copy = pstrdup(optimized_sql);
	current_pos = sql_copy;
	
    /* Parse and execute SET commands */
    while ((set_start = strstr(current_pos, "SET ")) != NULL) {
        char *param_name, *param_value;
        char *eq_pos;
        
		/* Find the end of the SET command (semicolon) */
        set_end = strchr(set_start, ';');
        if (!set_end) break;
        
        *set_end = '\0';
        
        /* Parse: SET parameter_name = value or SET parameter_name TO value */
        param_name = set_start + 4; /* Skip "SET " */
        
        /* Skip whitespace */
        while (*param_name == ' ' || *param_name == '\t') param_name++;
        
        /* Find = or TO */
        eq_pos = strstr(param_name, " TO ");
        if (!eq_pos) eq_pos = strstr(param_name, " = ");
        if (!eq_pos) eq_pos = strchr(param_name, '=');
        
        if (eq_pos) {
            *eq_pos = '\0';
            param_value = eq_pos + 1;
            
            /* Skip " TO " or " = " or "=" */
            while (*param_value == ' ' || *param_value == '=' || 
                   *param_value == 'T' || *param_value == 'O') {
                param_value++;
            }
            
            /* Remove quotes if present */
            if (*param_value == '\'' || *param_value == '"') {
                param_value++;
                char *quote_end = strchr(param_value, '\'');
                if (!quote_end) quote_end = strchr(param_value, '"');
                if (quote_end) *quote_end = '\0';
            }
            
            /* Trim trailing whitespace */
            char *end = param_value + strlen(param_value) - 1;
            while (end > param_value && (*end == ' ' || *end == '\t')) {
                *end = '\0';
                end--;
            }
            
            /* Apply the setting using GUC API - no SPI needed! */
            SetConfigOption(param_name, param_value, PGC_USERSET, PGC_S_SESSION);
        }
        
        current_pos = set_end + 1;
	}
	
	pfree(sql_copy);
}

/*
 * SQL function to manually trigger query optimization for testing
 */
PG_FUNCTION_INFO_V1(optimize_query);

Datum optimize_query(PG_FUNCTION_ARGS) {
	text *sql_text;
	char *sql_string;
	char *optimized_sql;
	
    if (PG_ARGISNULL(0)) PG_RETURN_NULL();
		
	sql_text = PG_GETARG_TEXT_PP(0);
	sql_string = text_to_cstring(sql_text);
	
	optimized_sql = call_external_optimizer(sql_string);
	
    if (optimized_sql) {
		text *result = cstring_to_text(optimized_sql);
		pfree(optimized_sql);
		pfree(sql_string);
		PG_RETURN_TEXT_P(result);
    } else {
		pfree(sql_string);
		PG_RETURN_TEXT_P(sql_text);
	}
}
