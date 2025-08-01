/*-------------------------------------------------------------------------
 *
 * initdb --- initialize a PostgreSQL installation
 *
 * initdb creates (initializes) a PostgreSQL database cluster (site,
 * instance, installation, whatever).  A database cluster is a
 * collection of PostgreSQL databases all managed by the same server.
 *
 * To create the database cluster, we create the directory that contains
 * all its data, create the files that hold the global tables, create
 * a few other control files for it, and create three databases: the
 * template databases "template0" and "template1", and a default user
 * database "neurdb".
 *
 * The template databases are ordinary PostgreSQL databases.  template0
 * is never supposed to change after initdb, whereas template1 can be
 * changed to add site-local standard data.  Either one can be copied
 * to produce a new database.
 *
 * For largely-historical reasons, the template1 database is the one built
 * by the basic bootstrap process.  After it is complete, template0 and
 * the default database, neurdb, are made just by copying template1.
 *
 * To create template1, we run the postgres (backend) program in bootstrap
 * mode and feed it data from the postgres.bki library file.  After this
 * initial bootstrap phase, some additional stuff is created by normal
 * SQL commands fed to a standalone backend.  Some of those commands are
 * just embedded into this program (yeah, it's ugly), but larger chunks
 * are taken from script files.
 *
 *
 * Note:
 *	 The program has some memory leakage - it isn't worth cleaning it up.
 *
 * This is a C implementation of the previous shell script for setting up a
 * PostgreSQL cluster location, and should be highly compatible with it.
 * author of C translation: Andrew Dunstan	   mailto:andrew@dunslane.net
 *
 * This code is released under the terms of the PostgreSQL License.
 *
 * Portions Copyright (c) 1996-2023, PostgreSQL Global Development Group
 * Portions Copyright (c) 1994, Regents of the University of California
 *
 * src/bin/initdb/initdb.c
 *
 *-------------------------------------------------------------------------
 */

#include "postgres_fe.h"

#include <dirent.h>
#include <fcntl.h>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/stat.h>
#ifdef USE_ICU
#include <unicode/ucol.h>
#endif
#include <unistd.h>
#include <signal.h>
#include <time.h>

#ifdef HAVE_SHM_OPEN
#include "sys/mman.h"
#endif

#include "access/xlog_internal.h"
#include "catalog/pg_authid_d.h"
#include "catalog/pg_class_d.h" /* pgrminclude ignore */
#include "catalog/pg_collation_d.h"
#include "catalog/pg_database_d.h"	/* pgrminclude ignore */
#include "common/file_perm.h"
#include "common/file_utils.h"
#include "common/logging.h"
#include "common/pg_prng.h"
#include "common/restricted_token.h"
#include "common/string.h"
#include "common/username.h"
#include "fe_utils/string_utils.h"
#include "getopt_long.h"
#include "mb/pg_wchar.h"
#include "miscadmin.h"


/* Ideally this would be in a .h file, but it hardly seems worth the trouble */
extern const char *select_default_timezone(const char *share_path);

/* simple list of strings */
typedef struct _stringlist
{
	char	   *str;
	struct _stringlist *next;
} _stringlist;

static const char *const auth_methods_host[] = {
	"trust", "reject", "scram-sha-256", "md5", "password", "ident", "radius",
#ifdef ENABLE_GSS
	"gss",
#endif
#ifdef ENABLE_SSPI
	"sspi",
#endif
#ifdef USE_PAM
	"pam", "pam ",
#endif
#ifdef USE_BSD_AUTH
	"bsd",
#endif
#ifdef USE_LDAP
	"ldap",
#endif
#ifdef USE_SSL
	"cert",
#endif
	NULL
};
static const char *const auth_methods_local[] = {
	"trust", "reject", "scram-sha-256", "md5", "password", "peer", "radius",
#ifdef USE_PAM
	"pam", "pam ",
#endif
#ifdef USE_BSD_AUTH
	"bsd",
#endif
#ifdef USE_LDAP
	"ldap",
#endif
	NULL
};

/*
 * these values are passed in by makefile defines
 */
static char *share_path = NULL;

/* values to be obtained from arguments */
static char *pg_data = NULL;
static char *encoding = NULL;
static char *locale = NULL;
static char *lc_collate = NULL;
static char *lc_ctype = NULL;
static char *lc_monetary = NULL;
static char *lc_numeric = NULL;
static char *lc_time = NULL;
static char *lc_messages = NULL;
static char locale_provider = COLLPROVIDER_LIBC;
static char *icu_locale = NULL;
static char *icu_rules = NULL;
static const char *default_text_search_config = NULL;
static char *username = NULL;
static bool pwprompt = false;
static char *pwfilename = NULL;
static char *superuser_password = NULL;
static const char *authmethodhost = NULL;
static const char *authmethodlocal = NULL;
static _stringlist *extra_guc_names = NULL;
static _stringlist *extra_guc_values = NULL;
static bool debug = false;
static bool noclean = false;
static bool noinstructions = false;
static bool do_sync = true;
static bool sync_only = false;
static bool show_setting = false;
static bool data_checksums = false;
static char *xlog_dir = NULL;
static char *str_wal_segment_size_mb = NULL;
static int	wal_segment_size_mb;


/* internal vars */
static const char *progname;
static int	encodingid;
static char *bki_file;
static char *hba_file;
static char *ident_file;
static char *conf_file;
static char *dictionary_file;
static char *info_schema_file;
static char *features_file;
static char *system_constraints_file;
static char *system_functions_file;
static char *system_views_file;
static bool success = false;
static bool made_new_pgdata = false;
static bool found_existing_pgdata = false;
static bool made_new_xlogdir = false;
static bool found_existing_xlogdir = false;
static char infoversion[100];
static bool caught_signal = false;
static bool output_failed = false;
static int	output_errno = 0;
static char *pgdata_native;

/* defaults */
static int	n_connections = 10;
static int	n_buffers = 50;
static const char *dynamic_shared_memory_type = NULL;
static const char *default_timezone = NULL;

/*
 * Warning messages for authentication methods
 */
#define AUTHTRUST_WARNING \
"# CAUTION: Configuring the system for local \"trust\" authentication\n" \
"# allows any local user to connect as any PostgreSQL user, including\n" \
"# the database superuser.  If you do not trust all your local users,\n" \
"# use another authentication method.\n"
static bool authwarning = false;

/*
 * Centralized knowledge of switches to pass to backend
 *
 * Note: we run the backend with -F (fsync disabled) and then do a single
 * pass of fsync'ing at the end.  This is faster than fsync'ing each step.
 *
 * Note: in the shell-script version, we also passed PGDATA as a -D switch,
 * but here it is more convenient to pass it as an environment variable
 * (no quoting to worry about).
 */
static const char *boot_options = "-F -c log_checkpoints=false";
static const char *backend_options = "--single -F -O -j -c search_path=pg_catalog -c exit_on_error=true -c log_checkpoints=false";

/* Additional switches to pass to backend (either boot or standalone) */
static char *extra_options = "";

static const char *const subdirs[] = {
	"global",
	"pg_wal/archive_status",
	"pg_commit_ts",
	"pg_dynshmem",
	"pg_notify",
	"pg_serial",
	"pg_snapshots",
	"pg_subtrans",
	"pg_twophase",
	"pg_multixact",
	"pg_multixact/members",
	"pg_multixact/offsets",
	"base",
	"base/1",
	"pg_replslot",
	"pg_tblspc",
	"pg_stat",
	"pg_stat_tmp",
	"pg_xact",
	"pg_logical",
	"pg_logical/snapshots",
	"pg_logical/mappings"
};


/* path to 'initdb' binary directory */
static char bin_path[MAXPGPATH];
static char backend_exec[MAXPGPATH];

static char **replace_token(char **lines,
							const char *token, const char *replacement);
static char **replace_guc_value(char **lines,
								const char *guc_name, const char *guc_value,
								bool mark_as_comment);
static bool guc_value_requires_quotes(const char *guc_value);
static char **readfile(const char *path);
static void writefile(char *path, char **lines);
static FILE *popen_check(const char *command, const char *mode);
static char *get_id(void);
static int	get_encoding_id(const char *encoding_name);
static void set_input(char **dest, const char *filename);
static void check_input(char *path);
static void write_version_file(const char *extrapath);
static void set_null_conf(void);
static void test_config_settings(void);
static bool test_specific_config_settings(int test_conns, int test_buffs);
static void setup_config(void);
static void bootstrap_template1(void);
static void setup_auth(FILE *cmdfd);
static void get_su_pwd(void);
static void setup_depend(FILE *cmdfd);
static void setup_run_file(FILE *cmdfd, const char *filename);
static void setup_description(FILE *cmdfd);
static void setup_collation(FILE *cmdfd);
static void setup_privileges(FILE *cmdfd);
static void set_info_version(void);
static void setup_schema(FILE *cmdfd);
static void load_plpgsql(FILE *cmdfd);
static void vacuum_db(FILE *cmdfd);
static void make_template0(FILE *cmdfd);
static void make_postgres(FILE *cmdfd);
static void trapsig(SIGNAL_ARGS);
static void check_ok(void);
static char *escape_quotes(const char *src);
static char *escape_quotes_bki(const char *src);
static int	locale_date_order(const char *locale);
static void check_locale_name(int category, const char *locale,
							  char **canonname);
static bool check_locale_encoding(const char *locale, int user_enc);
static void setlocales(void);
static void usage(const char *progname);
void		setup_pgdata(void);
void		setup_bin_paths(const char *argv0);
void		setup_data_file_paths(void);
void		setup_locale_encoding(void);
void		setup_signals(void);
void		setup_text_search(void);
void		create_data_directory(void);
void		create_xlog_or_symlink(void);
void		warn_on_mount_point(int error);
void		initialize_data_directory(void);

/*
 * macros for running pipes to postgres
 */
#define PG_CMD_DECL		char cmd[MAXPGPATH]; FILE *cmdfd

#define PG_CMD_OPEN \
do { \
	cmdfd = popen_check(cmd, "w"); \
	if (cmdfd == NULL) \
		exit(1); /* message already printed by popen_check */ \
} while (0)

#define PG_CMD_CLOSE \
do { \
	if (pclose_check(cmdfd)) \
		exit(1); /* message already printed by pclose_check */ \
} while (0)

#define PG_CMD_PUTS(line) \
do { \
	if (fputs(line, cmdfd) < 0 || fflush(cmdfd) < 0) \
		output_failed = true, output_errno = errno; \
} while (0)

#define PG_CMD_PRINTF(fmt, ...) \
do { \
	if (fprintf(cmdfd, fmt, __VA_ARGS__) < 0 || fflush(cmdfd) < 0) \
		output_failed = true, output_errno = errno; \
} while (0)

/*
 * Escape single quotes and backslashes, suitably for insertions into
 * configuration files or SQL E'' strings.
 */
static char *
escape_quotes(const char *src)
{
	char	   *result = escape_single_quotes_ascii(src);

	if (!result)
		pg_fatal("out of memory");
	return result;
}

/*
 * Escape a field value to be inserted into the BKI data.
 * Run the value through escape_quotes (which will be inverted
 * by the backend's DeescapeQuotedString() function), then wrap
 * the value in single quotes, even if that isn't strictly necessary.
 */
static char *
escape_quotes_bki(const char *src)
{
	char	   *result;
	char	   *data = escape_quotes(src);
	char	   *resultp;
	char	   *datap;

	result = (char *) pg_malloc(strlen(data) + 3);
	resultp = result;
	*resultp++ = '\'';
	for (datap = data; *datap; datap++)
		*resultp++ = *datap;
	*resultp++ = '\'';
	*resultp = '\0';

	free(data);
	return result;
}

/*
 * Add an item at the end of a stringlist.
 */
static void
add_stringlist_item(_stringlist **listhead, const char *str)
{
	_stringlist *newentry = pg_malloc(sizeof(_stringlist));
	_stringlist *oldentry;

	newentry->str = pg_strdup(str);
	newentry->next = NULL;
	if (*listhead == NULL)
		*listhead = newentry;
	else
	{
		for (oldentry = *listhead; oldentry->next; oldentry = oldentry->next)
			 /* skip */ ;
		oldentry->next = newentry;
	}
}

/*
 * Modify the array of lines, replacing "token" by "replacement"
 * the first time it occurs on each line.
 *
 * The array must be a malloc'd array of individually malloc'd strings.
 * We free any discarded strings.
 *
 * This does most of what sed was used for in the shell script, but
 * doesn't need any regexp stuff.
 */
static char **
replace_token(char **lines, const char *token, const char *replacement)
{
	int			toklen,
				replen,
				diff;

	toklen = strlen(token);
	replen = strlen(replacement);
	diff = replen - toklen;

	for (int i = 0; lines[i]; i++)
	{
		char	   *where;
		char	   *newline;
		int			pre;

		/* nothing to do if no change needed */
		if ((where = strstr(lines[i], token)) == NULL)
			continue;

		/* if we get here a change is needed - set up new line */

		newline = (char *) pg_malloc(strlen(lines[i]) + diff + 1);

		pre = where - lines[i];

		memcpy(newline, lines[i], pre);

		memcpy(newline + pre, replacement, replen);

		strcpy(newline + pre + replen, lines[i] + pre + toklen);

		free(lines[i]);
		lines[i] = newline;
	}

	return lines;
}

/*
 * Modify the array of lines, replacing the possibly-commented-out
 * assignment of parameter guc_name with a live assignment of guc_value.
 * The value will be suitably quoted.
 *
 * If mark_as_comment is true, the replacement line is prefixed with '#'.
 * This is used for fixing up cases where the effective default might not
 * match what is in postgresql.conf.sample.
 *
 * We assume there's at most one matching assignment.  If we find no match,
 * append a new line with the desired assignment.
 *
 * The array must be a malloc'd array of individually malloc'd strings.
 * We free any discarded strings.
 */
static char **
replace_guc_value(char **lines, const char *guc_name, const char *guc_value,
				  bool mark_as_comment)
{
	int			namelen = strlen(guc_name);
	PQExpBuffer newline = createPQExpBuffer();
	int			i;

	/* prepare the replacement line, except for possible comment and newline */
	if (mark_as_comment)
		appendPQExpBufferChar(newline, '#');
	appendPQExpBuffer(newline, "%s = ", guc_name);
	if (guc_value_requires_quotes(guc_value))
		appendPQExpBuffer(newline, "'%s'", escape_quotes(guc_value));
	else
		appendPQExpBufferStr(newline, guc_value);

	for (i = 0; lines[i]; i++)
	{
		const char *where;
		const char *namestart;

		/*
		 * Look for a line assigning to guc_name.  Typically it will be
		 * preceded by '#', but that might not be the case if a -c switch
		 * overrides a previous assignment.  We allow leading whitespace too,
		 * although normally there wouldn't be any.
		 */
		where = lines[i];
		while (*where == '#' || isspace((unsigned char) *where))
			where++;
		if (pg_strncasecmp(where, guc_name, namelen) != 0)
			continue;
		namestart = where;
		where += namelen;
		while (isspace((unsigned char) *where))
			where++;
		if (*where != '=')
			continue;

		/* found it -- let's use the canonical casing shown in the file */
		memcpy(&newline->data[mark_as_comment ? 1 : 0], namestart, namelen);

		/* now append the original comment if any */
		where = strrchr(where, '#');
		if (where)
		{
			/*
			 * We try to preserve original indentation, which is tedious.
			 * oldindent and newindent are measured in de-tab-ified columns.
			 */
			const char *ptr;
			int			oldindent = 0;
			int			newindent;

			for (ptr = lines[i]; ptr < where; ptr++)
			{
				if (*ptr == '\t')
					oldindent += 8 - (oldindent % 8);
				else
					oldindent++;
			}
			/* ignore the possibility of tabs in guc_value */
			newindent = newline->len;
			/* append appropriate tabs and spaces, forcing at least one */
			oldindent = Max(oldindent, newindent + 1);
			while (newindent < oldindent)
			{
				int			newindent_if_tab = newindent + 8 - (newindent % 8);

				if (newindent_if_tab <= oldindent)
				{
					appendPQExpBufferChar(newline, '\t');
					newindent = newindent_if_tab;
				}
				else
				{
					appendPQExpBufferChar(newline, ' ');
					newindent++;
				}
			}
			/* and finally append the old comment */
			appendPQExpBufferStr(newline, where);
			/* we'll have appended the original newline; don't add another */
		}
		else
			appendPQExpBufferChar(newline, '\n');

		free(lines[i]);
		lines[i] = newline->data;

		break;					/* assume there's only one match */
	}

	if (lines[i] == NULL)
	{
		/*
		 * No match, so append a new entry.  (We rely on the bootstrap server
		 * to complain if it's not a valid GUC name.)
		 */
		appendPQExpBufferChar(newline, '\n');
		lines = pg_realloc_array(lines, char *, i + 2);
		lines[i++] = newline->data;
		lines[i] = NULL;		/* keep the array null-terminated */
	}

	free(newline);				/* but don't free newline->data */

	return lines;
}

/*
 * Decide if we should quote a replacement GUC value.  We aren't too tense
 * here, but we'd like to avoid quoting simple identifiers and numbers
 * with units, which are common cases.
 */
static bool
guc_value_requires_quotes(const char *guc_value)
{
	/* Don't use <ctype.h> macros here, they might accept too much */
#define LETTERS	"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
#define DIGITS	"0123456789"

	if (*guc_value == '\0')
		return true;			/* empty string must be quoted */
	if (strchr(LETTERS, *guc_value))
	{
		if (strspn(guc_value, LETTERS DIGITS) == strlen(guc_value))
			return false;		/* it's an identifier */
		return true;			/* nope */
	}
	if (strchr(DIGITS, *guc_value))
	{
		/* skip over digits */
		guc_value += strspn(guc_value, DIGITS);
		/* there can be zero or more unit letters after the digits */
		if (strspn(guc_value, LETTERS) == strlen(guc_value))
			return false;		/* it's a number, possibly with units */
		return true;			/* nope */
	}
	return true;				/* all else must be quoted */
}

/*
 * get the lines from a text file
 *
 * The result is a malloc'd array of individually malloc'd strings.
 */
static char **
readfile(const char *path)
{
	char	  **result;
	FILE	   *infile;
	StringInfoData line;
	int			maxlines;
	int			n;

	if ((infile = fopen(path, "r")) == NULL)
		pg_fatal("could not open file \"%s\" for reading: %m", path);

	initStringInfo(&line);

	maxlines = 1024;
	result = (char **) pg_malloc(maxlines * sizeof(char *));

	n = 0;
	while (pg_get_line_buf(infile, &line))
	{
		/* make sure there will be room for a trailing NULL pointer */
		if (n >= maxlines - 1)
		{
			maxlines *= 2;
			result = (char **) pg_realloc(result, maxlines * sizeof(char *));
		}

		result[n++] = pg_strdup(line.data);
	}
	result[n] = NULL;

	pfree(line.data);

	fclose(infile);

	return result;
}

/*
 * write an array of lines to a file
 *
 * "lines" must be a malloc'd array of individually malloc'd strings.
 * All that data is freed here.
 *
 * This is only used to write text files.  Use fopen "w" not PG_BINARY_W
 * so that the resulting configuration files are nicely editable on Windows.
 */
static void
writefile(char *path, char **lines)
{
	FILE	   *out_file;
	char	  **line;

	if ((out_file = fopen(path, "w")) == NULL)
		pg_fatal("could not open file \"%s\" for writing: %m", path);
	for (line = lines; *line != NULL; line++)
	{
		if (fputs(*line, out_file) < 0)
			pg_fatal("could not write file \"%s\": %m", path);
		free(*line);
	}
	if (fclose(out_file))
		pg_fatal("could not close file \"%s\": %m", path);
	free(lines);
}

/*
 * Open a subcommand with suitable error messaging
 */
static FILE *
popen_check(const char *command, const char *mode)
{
	FILE	   *cmdfd;

	fflush(NULL);
	errno = 0;
	cmdfd = popen(command, mode);
	if (cmdfd == NULL)
		pg_log_error("could not execute command \"%s\": %m", command);
	return cmdfd;
}

/*
 * clean up any files we created on failure
 * if we created the data directory remove it too
 */
static void
cleanup_directories_atexit(void)
{
	if (success)
		return;

	if (!noclean)
	{
		if (made_new_pgdata)
		{
			pg_log_info("removing data directory \"%s\"", pg_data);
			if (!rmtree(pg_data, true))
				pg_log_error("failed to remove data directory");
		}
		else if (found_existing_pgdata)
		{
			pg_log_info("removing contents of data directory \"%s\"",
						pg_data);
			if (!rmtree(pg_data, false))
				pg_log_error("failed to remove contents of data directory");
		}

		if (made_new_xlogdir)
		{
			pg_log_info("removing WAL directory \"%s\"", xlog_dir);
			if (!rmtree(xlog_dir, true))
				pg_log_error("failed to remove WAL directory");
		}
		else if (found_existing_xlogdir)
		{
			pg_log_info("removing contents of WAL directory \"%s\"", xlog_dir);
			if (!rmtree(xlog_dir, false))
				pg_log_error("failed to remove contents of WAL directory");
		}
		/* otherwise died during startup, do nothing! */
	}
	else
	{
		if (made_new_pgdata || found_existing_pgdata)
			pg_log_info("data directory \"%s\" not removed at user's request",
						pg_data);

		if (made_new_xlogdir || found_existing_xlogdir)
			pg_log_info("WAL directory \"%s\" not removed at user's request",
						xlog_dir);
	}
}

/*
 * find the current user
 *
 * on unix make sure it isn't root
 */
static char *
get_id(void)
{
	const char *username;

#ifndef WIN32
	if (geteuid() == 0)			/* 0 is root's uid */
	{
		pg_log_error("cannot be run as root");
		pg_log_error_hint("Please log in (using, e.g., \"su\") as the (unprivileged) user that will own the server process.");
		exit(1);
	}
#endif

	username = get_user_name_or_exit(progname);

	return pg_strdup(username);
}

static char *
encodingid_to_string(int enc)
{
	char		result[20];

	sprintf(result, "%d", enc);
	return pg_strdup(result);
}

/*
 * get the encoding id for a given encoding name
 */
static int
get_encoding_id(const char *encoding_name)
{
	int			enc;

	if (encoding_name && *encoding_name)
	{
		if ((enc = pg_valid_server_encoding(encoding_name)) >= 0)
			return enc;
	}
	pg_fatal("\"%s\" is not a valid server encoding name",
			 encoding_name ? encoding_name : "(null)");
}

/*
 * Support for determining the best default text search configuration.
 * We key this off the first part of LC_CTYPE (ie, the language name).
 */
struct tsearch_config_match
{
	const char *tsconfname;
	const char *langname;
};

static const struct tsearch_config_match tsearch_config_languages[] =
{
	{"arabic", "ar"},
	{"arabic", "Arabic"},
	{"armenian", "hy"},
	{"armenian", "Armenian"},
	{"basque", "eu"},
	{"basque", "Basque"},
	{"catalan", "ca"},
	{"catalan", "Catalan"},
	{"danish", "da"},
	{"danish", "Danish"},
	{"dutch", "nl"},
	{"dutch", "Dutch"},
	{"english", "C"},
	{"english", "POSIX"},
	{"english", "en"},
	{"english", "English"},
	{"finnish", "fi"},
	{"finnish", "Finnish"},
	{"french", "fr"},
	{"french", "French"},
	{"german", "de"},
	{"german", "German"},
	{"greek", "el"},
	{"greek", "Greek"},
	{"hindi", "hi"},
	{"hindi", "Hindi"},
	{"hungarian", "hu"},
	{"hungarian", "Hungarian"},
	{"indonesian", "id"},
	{"indonesian", "Indonesian"},
	{"irish", "ga"},
	{"irish", "Irish"},
	{"italian", "it"},
	{"italian", "Italian"},
	{"lithuanian", "lt"},
	{"lithuanian", "Lithuanian"},
	{"nepali", "ne"},
	{"nepali", "Nepali"},
	{"norwegian", "no"},
	{"norwegian", "Norwegian"},
	{"portuguese", "pt"},
	{"portuguese", "Portuguese"},
	{"romanian", "ro"},
	{"russian", "ru"},
	{"russian", "Russian"},
	{"serbian", "sr"},
	{"serbian", "Serbian"},
	{"spanish", "es"},
	{"spanish", "Spanish"},
	{"swedish", "sv"},
	{"swedish", "Swedish"},
	{"tamil", "ta"},
	{"tamil", "Tamil"},
	{"turkish", "tr"},
	{"turkish", "Turkish"},
	{"yiddish", "yi"},
	{"yiddish", "Yiddish"},
	{NULL, NULL}				/* end marker */
};

/*
 * Look for a text search configuration matching lc_ctype, and return its
 * name; return NULL if no match.
 */
static const char *
find_matching_ts_config(const char *lc_type)
{
	int			i;
	char	   *langname,
			   *ptr;

	/*
	 * Convert lc_ctype to a language name by stripping everything after an
	 * underscore (usual case) or a hyphen (Windows "locale name"; see
	 * comments at IsoLocaleName()).
	 *
	 * XXX Should ' ' be a stop character?	This would select "norwegian" for
	 * the Windows locale "Norwegian (Nynorsk)_Norway.1252".  If we do so, we
	 * should also accept the "nn" and "nb" Unix locales.
	 *
	 * Just for paranoia, we also stop at '.' or '@'.
	 */
	if (lc_type == NULL)
		langname = pg_strdup("");
	else
	{
		ptr = langname = pg_strdup(lc_type);
		while (*ptr &&
			   *ptr != '_' && *ptr != '-' && *ptr != '.' && *ptr != '@')
			ptr++;
		*ptr = '\0';
	}

	for (i = 0; tsearch_config_languages[i].tsconfname; i++)
	{
		if (pg_strcasecmp(tsearch_config_languages[i].langname, langname) == 0)
		{
			free(langname);
			return tsearch_config_languages[i].tsconfname;
		}
	}

	free(langname);
	return NULL;
}


/*
 * set name of given input file variable under data directory
 */
static void
set_input(char **dest, const char *filename)
{
	*dest = psprintf("%s/%s", share_path, filename);
}

/*
 * check that given input file exists
 */
static void
check_input(char *path)
{
	struct stat statbuf;

	if (stat(path, &statbuf) != 0)
	{
		if (errno == ENOENT)
		{
			pg_log_error("file \"%s\" does not exist", path);
			pg_log_error_hint("This might mean you have a corrupted installation or identified the wrong directory with the invocation option -L.");
		}
		else
		{
			pg_log_error("could not access file \"%s\": %m", path);
			pg_log_error_hint("This might mean you have a corrupted installation or identified the wrong directory with the invocation option -L.");
		}
		exit(1);
	}
	if (!S_ISREG(statbuf.st_mode))
	{
		pg_log_error("file \"%s\" is not a regular file", path);
		pg_log_error_hint("This might mean you have a corrupted installation or identified the wrong directory with the invocation option -L.");
		exit(1);
	}
}

/*
 * write out the PG_VERSION file in the data dir, or its subdirectory
 * if extrapath is not NULL
 */
static void
write_version_file(const char *extrapath)
{
	FILE	   *version_file;
	char	   *path;

	if (extrapath == NULL)
		path = psprintf("%s/PG_VERSION", pg_data);
	else
		path = psprintf("%s/%s/PG_VERSION", pg_data, extrapath);

	if ((version_file = fopen(path, PG_BINARY_W)) == NULL)
		pg_fatal("could not open file \"%s\" for writing: %m", path);
	if (fprintf(version_file, "%s\n", PG_MAJORVERSION) < 0 ||
		fclose(version_file))
		pg_fatal("could not write file \"%s\": %m", path);
	free(path);
}

/*
 * set up an empty config file so we can check config settings by launching
 * a test backend
 */
static void
set_null_conf(void)
{
	FILE	   *conf_file;
	char	   *path;

	path = psprintf("%s/postgresql.conf", pg_data);
	conf_file = fopen(path, PG_BINARY_W);
	if (conf_file == NULL)
		pg_fatal("could not open file \"%s\" for writing: %m", path);
	if (fclose(conf_file))
		pg_fatal("could not write file \"%s\": %m", path);
	free(path);
}

/*
 * Determine which dynamic shared memory implementation should be used on
 * this platform.  POSIX shared memory is preferable because the default
 * allocation limits are much higher than the limits for System V on most
 * systems that support both, but the fact that a platform has shm_open
 * doesn't guarantee that that call will succeed when attempted.  So, we
 * attempt to reproduce what the postmaster will do when allocating a POSIX
 * segment in dsm_impl.c; if it doesn't work, we assume it won't work for
 * the postmaster either, and configure the cluster for System V shared
 * memory instead.
 *
 * We avoid choosing Solaris's implementation of shm_open() by default.  It
 * can sleep and fail spuriously under contention.
 */
static const char *
choose_dsm_implementation(void)
{
#if defined(HAVE_SHM_OPEN) && !defined(__sun__)
	int			ntries = 10;
	pg_prng_state prng_state;

	/* Initialize prng; this function is its only user in this program. */
	pg_prng_seed(&prng_state, (uint64) (getpid() ^ time(NULL)));

	while (ntries > 0)
	{
		uint32		handle;
		char		name[64];
		int			fd;

		handle = pg_prng_uint32(&prng_state);
		snprintf(name, 64, "/PostgreSQL.%u", handle);
		if ((fd = shm_open(name, O_CREAT | O_RDWR | O_EXCL, 0600)) != -1)
		{
			close(fd);
			shm_unlink(name);
			return "posix";
		}
		if (errno != EEXIST)
			break;
		--ntries;
	}
#endif

#ifdef WIN32
	return "windows";
#else
	return "sysv";
#endif
}

/*
 * Determine platform-specific config settings
 *
 * Use reasonable values if kernel will let us, else scale back.
 */
static void
test_config_settings(void)
{
	/*
	 * This macro defines the minimum shared_buffers we want for a given
	 * max_connections value. The arrays show the settings to try.
	 */
#define MIN_BUFS_FOR_CONNS(nconns)	((nconns) * 10)

	static const int trial_conns[] = {
		100, 50, 40, 30, 20
	};
	static const int trial_bufs[] = {
		16384, 8192, 4096, 3584, 3072, 2560, 2048, 1536,
		1000, 900, 800, 700, 600, 500,
		400, 300, 200, 100, 50
	};

	const int	connslen = sizeof(trial_conns) / sizeof(int);
	const int	bufslen = sizeof(trial_bufs) / sizeof(int);
	int			i,
				test_conns,
				test_buffs,
				ok_buffers = 0;

	/*
	 * Need to determine working DSM implementation first so that subsequent
	 * tests don't fail because DSM setting doesn't work.
	 */
	printf(_("selecting dynamic shared memory implementation ... "));
	fflush(stdout);
	dynamic_shared_memory_type = choose_dsm_implementation();
	printf("%s\n", dynamic_shared_memory_type);

	/*
	 * Probe for max_connections before shared_buffers, since it is subject to
	 * more constraints than shared_buffers.
	 */
	printf(_("selecting default max_connections ... "));
	fflush(stdout);

	for (i = 0; i < connslen; i++)
	{
		test_conns = trial_conns[i];
		test_buffs = MIN_BUFS_FOR_CONNS(test_conns);

		if (test_specific_config_settings(test_conns, test_buffs))
		{
			ok_buffers = test_buffs;
			break;
		}
	}
	if (i >= connslen)
		i = connslen - 1;
	n_connections = trial_conns[i];

	printf("%d\n", n_connections);

	printf(_("selecting default shared_buffers ... "));
	fflush(stdout);

	for (i = 0; i < bufslen; i++)
	{
		/* Use same amount of memory, independent of BLCKSZ */
		test_buffs = (trial_bufs[i] * 8192) / BLCKSZ;
		if (test_buffs <= ok_buffers)
		{
			test_buffs = ok_buffers;
			break;
		}

		if (test_specific_config_settings(n_connections, test_buffs))
			break;
	}
	n_buffers = test_buffs;

	if ((n_buffers * (BLCKSZ / 1024)) % 1024 == 0)
		printf("%dMB\n", (n_buffers * (BLCKSZ / 1024)) / 1024);
	else
		printf("%dkB\n", n_buffers * (BLCKSZ / 1024));

	printf(_("selecting default time zone ... "));
	fflush(stdout);
	default_timezone = select_default_timezone(share_path);
	printf("%s\n", default_timezone ? default_timezone : "GMT");
}

/*
 * Test a specific combination of configuration settings.
 */
static bool
test_specific_config_settings(int test_conns, int test_buffs)
{
	PQExpBuffer cmd = createPQExpBuffer();
	_stringlist *gnames,
			   *gvalues;
	int			status;

	/* Set up the test postmaster invocation */
	printfPQExpBuffer(cmd,
					  "\"%s\" --check %s %s "
					  "-c max_connections=%d "
					  "-c shared_buffers=%d "
					  "-c dynamic_shared_memory_type=%s",
					  backend_exec, boot_options, extra_options,
					  test_conns, test_buffs,
					  dynamic_shared_memory_type);

	/* Add any user-given setting overrides */
	for (gnames = extra_guc_names, gvalues = extra_guc_values;
		 gnames != NULL;		/* assume lists have the same length */
		 gnames = gnames->next, gvalues = gvalues->next)
	{
		appendPQExpBuffer(cmd, " -c %s=", gnames->str);
		appendShellString(cmd, gvalues->str);
	}

	appendPQExpBuffer(cmd,
					  " < \"%s\" > \"%s\" 2>&1",
					  DEVNULL, DEVNULL);

	fflush(NULL);
	status = system(cmd->data);

	destroyPQExpBuffer(cmd);

	return (status == 0);
}

/*
 * Calculate the default wal_size with a "pretty" unit.
 */
static char *
pretty_wal_size(int segment_count)
{
	int			sz = wal_segment_size_mb * segment_count;
	char	   *result = pg_malloc(14);

	if ((sz % 1024) == 0)
		snprintf(result, 14, "%dGB", sz / 1024);
	else
		snprintf(result, 14, "%dMB", sz);

	return result;
}

/*
 * set up all the config files
 */
static void
setup_config(void)
{
	char	  **conflines;
	char		repltok[MAXPGPATH];
	char		path[MAXPGPATH];
	_stringlist *gnames,
			   *gvalues;

	fputs(_("creating configuration files ... "), stdout);
	fflush(stdout);

	/* postgresql.conf */

	conflines = readfile(conf_file);

	snprintf(repltok, sizeof(repltok), "%d", n_connections);
	conflines = replace_guc_value(conflines, "max_connections",
								  repltok, false);

	if ((n_buffers * (BLCKSZ / 1024)) % 1024 == 0)
		snprintf(repltok, sizeof(repltok), "%dMB",
				 (n_buffers * (BLCKSZ / 1024)) / 1024);
	else
		snprintf(repltok, sizeof(repltok), "%dkB",
				 n_buffers * (BLCKSZ / 1024));
	conflines = replace_guc_value(conflines, "shared_buffers",
								  repltok, false);

	conflines = replace_guc_value(conflines, "lc_messages",
								  lc_messages, false);

	conflines = replace_guc_value(conflines, "lc_monetary",
								  lc_monetary, false);

	conflines = replace_guc_value(conflines, "lc_numeric",
								  lc_numeric, false);

	conflines = replace_guc_value(conflines, "lc_time",
								  lc_time, false);

	switch (locale_date_order(lc_time))
	{
		case DATEORDER_YMD:
			strcpy(repltok, "iso, ymd");
			break;
		case DATEORDER_DMY:
			strcpy(repltok, "iso, dmy");
			break;
		case DATEORDER_MDY:
		default:
			strcpy(repltok, "iso, mdy");
			break;
	}
	conflines = replace_guc_value(conflines, "datestyle",
								  repltok, false);

	snprintf(repltok, sizeof(repltok), "pg_catalog.%s",
			 default_text_search_config);
	conflines = replace_guc_value(conflines, "default_text_search_config",
								  repltok, false);

	if (default_timezone)
	{
		conflines = replace_guc_value(conflines, "timezone",
									  default_timezone, false);
		conflines = replace_guc_value(conflines, "log_timezone",
									  default_timezone, false);
	}

	conflines = replace_guc_value(conflines, "dynamic_shared_memory_type",
								  dynamic_shared_memory_type, false);

	/* Caution: these depend on wal_segment_size_mb, they're not constants */
	conflines = replace_guc_value(conflines, "min_wal_size",
								  pretty_wal_size(DEFAULT_MIN_WAL_SEGS), false);

	conflines = replace_guc_value(conflines, "max_wal_size",
								  pretty_wal_size(DEFAULT_MAX_WAL_SEGS), false);

	/*
	 * Fix up various entries to match the true compile-time defaults.  Since
	 * these are indeed defaults, keep the postgresql.conf lines commented.
	 */
	conflines = replace_guc_value(conflines, "unix_socket_directories",
								  DEFAULT_PGSOCKET_DIR, true);

	conflines = replace_guc_value(conflines, "port",
								  DEF_PGPORT_STR, true);

#if DEFAULT_BACKEND_FLUSH_AFTER > 0
	snprintf(repltok, sizeof(repltok), "%dkB",
			 DEFAULT_BACKEND_FLUSH_AFTER * (BLCKSZ / 1024));
	conflines = replace_guc_value(conflines, "backend_flush_after",
								  repltok, true);
#endif

#if DEFAULT_BGWRITER_FLUSH_AFTER > 0
	snprintf(repltok, sizeof(repltok), "%dkB",
			 DEFAULT_BGWRITER_FLUSH_AFTER * (BLCKSZ / 1024));
	conflines = replace_guc_value(conflines, "bgwriter_flush_after",
								  repltok, true);
#endif

#if DEFAULT_CHECKPOINT_FLUSH_AFTER > 0
	snprintf(repltok, sizeof(repltok), "%dkB",
			 DEFAULT_CHECKPOINT_FLUSH_AFTER * (BLCKSZ / 1024));
	conflines = replace_guc_value(conflines, "checkpoint_flush_after",
								  repltok, true);
#endif

#ifndef USE_PREFETCH
	conflines = replace_guc_value(conflines, "effective_io_concurrency",
								  "0", true);
#endif

#ifdef WIN32
	conflines = replace_guc_value(conflines, "update_process_title",
								  "off", true);
#endif

	/*
	 * Change password_encryption setting to md5 if md5 was chosen as an
	 * authentication method, unless scram-sha-256 was also chosen.
	 */
	if ((strcmp(authmethodlocal, "md5") == 0 &&
		 strcmp(authmethodhost, "scram-sha-256") != 0) ||
		(strcmp(authmethodhost, "md5") == 0 &&
		 strcmp(authmethodlocal, "scram-sha-256") != 0))
	{
		conflines = replace_guc_value(conflines, "password_encryption",
									  "md5", false);
	}

	/*
	 * If group access has been enabled for the cluster then it makes sense to
	 * ensure that the log files also allow group access.  Otherwise a backup
	 * from a user in the group would fail if the log files were not
	 * relocated.
	 */
	if (pg_dir_create_mode == PG_DIR_MODE_GROUP)
	{
		conflines = replace_guc_value(conflines, "log_file_mode",
									  "0640", false);
	}

	/*
	 * Now replace anything that's overridden via -c switches.
	 */
	for (gnames = extra_guc_names, gvalues = extra_guc_values;
		 gnames != NULL;		/* assume lists have the same length */
		 gnames = gnames->next, gvalues = gvalues->next)
	{
		conflines = replace_guc_value(conflines, gnames->str,
									  gvalues->str, false);
	}

	/* ... and write out the finished postgresql.conf file */
	snprintf(path, sizeof(path), "%s/postgresql.conf", pg_data);

	writefile(path, conflines);
	if (chmod(path, pg_file_create_mode) != 0)
		pg_fatal("could not change permissions of \"%s\": %m", path);


	/* postgresql.auto.conf */

	conflines = pg_malloc_array(char *, 3);
	conflines[0] = pg_strdup("# Do not edit this file manually!\n");
	conflines[1] = pg_strdup("# It will be overwritten by the ALTER SYSTEM command.\n");
	conflines[2] = NULL;

	sprintf(path, "%s/postgresql.auto.conf", pg_data);

	writefile(path, conflines);
	if (chmod(path, pg_file_create_mode) != 0)
		pg_fatal("could not change permissions of \"%s\": %m", path);


	/* pg_hba.conf */

	conflines = readfile(hba_file);

	conflines = replace_token(conflines, "@remove-line-for-nolocal@", "");


	/*
	 * Probe to see if there is really any platform support for IPv6, and
	 * comment out the relevant pg_hba line if not.  This avoids runtime
	 * warnings if getaddrinfo doesn't actually cope with IPv6.  Particularly
	 * useful on Windows, where executables built on a machine with IPv6 may
	 * have to run on a machine without.
	 */
	{
		struct addrinfo *gai_result;
		struct addrinfo hints;
		int			err = 0;

#ifdef WIN32
		/* need to call WSAStartup before calling getaddrinfo */
		WSADATA		wsaData;

		err = WSAStartup(MAKEWORD(2, 2), &wsaData);
#endif

		/* for best results, this code should match parse_hba_line() */
		hints.ai_flags = AI_NUMERICHOST;
		hints.ai_family = AF_UNSPEC;
		hints.ai_socktype = 0;
		hints.ai_protocol = 0;
		hints.ai_addrlen = 0;
		hints.ai_canonname = NULL;
		hints.ai_addr = NULL;
		hints.ai_next = NULL;

		if (err != 0 ||
			getaddrinfo("::1", NULL, &hints, &gai_result) != 0)
		{
			conflines = replace_token(conflines,
									  "host    all             all             ::1",
									  "#host    all             all             ::1");
			conflines = replace_token(conflines,
									  "host    replication     all             ::1",
									  "#host    replication     all             ::1");
		}
	}

	/* Replace default authentication methods */
	conflines = replace_token(conflines,
							  "@authmethodhost@",
							  authmethodhost);
	conflines = replace_token(conflines,
							  "@authmethodlocal@",
							  authmethodlocal);

	conflines = replace_token(conflines,
							  "@authcomment@",
							  (strcmp(authmethodlocal, "trust") == 0 || strcmp(authmethodhost, "trust") == 0) ? AUTHTRUST_WARNING : "");

	snprintf(path, sizeof(path), "%s/pg_hba.conf", pg_data);

	writefile(path, conflines);
	if (chmod(path, pg_file_create_mode) != 0)
		pg_fatal("could not change permissions of \"%s\": %m", path);


	/* pg_ident.conf */

	conflines = readfile(ident_file);

	snprintf(path, sizeof(path), "%s/pg_ident.conf", pg_data);

	writefile(path, conflines);
	if (chmod(path, pg_file_create_mode) != 0)
		pg_fatal("could not change permissions of \"%s\": %m", path);

	check_ok();
}


/*
 * run the BKI script in bootstrap mode to create template1
 */
static void
bootstrap_template1(void)
{
	PG_CMD_DECL;
	char	  **line;
	char	  **bki_lines;
	char		headerline[MAXPGPATH];
	char		buf[64];

	printf(_("running bootstrap script ... "));
	fflush(stdout);

	bki_lines = readfile(bki_file);

	/* Check that bki file appears to be of the right version */

	snprintf(headerline, sizeof(headerline), "# PostgreSQL %s\n",
			 PG_MAJORVERSION);

	if (strcmp(headerline, *bki_lines) != 0)
	{
		pg_log_error("input file \"%s\" does not belong to PostgreSQL %s",
					 bki_file, PG_VERSION);
		pg_log_error_hint("Specify the correct path using the option -L.");
		exit(1);
	}

	/* Substitute for various symbols used in the BKI file */

	sprintf(buf, "%d", NAMEDATALEN);
	bki_lines = replace_token(bki_lines, "NAMEDATALEN", buf);

	sprintf(buf, "%d", (int) sizeof(Pointer));
	bki_lines = replace_token(bki_lines, "SIZEOF_POINTER", buf);

	bki_lines = replace_token(bki_lines, "ALIGNOF_POINTER",
							  (sizeof(Pointer) == 4) ? "i" : "d");

	bki_lines = replace_token(bki_lines, "FLOAT8PASSBYVAL",
							  FLOAT8PASSBYVAL ? "true" : "false");

	bki_lines = replace_token(bki_lines, "POSTGRES",
							  escape_quotes_bki(username));

	bki_lines = replace_token(bki_lines, "ENCODING",
							  encodingid_to_string(encodingid));

	bki_lines = replace_token(bki_lines, "LC_COLLATE",
							  escape_quotes_bki(lc_collate));

	bki_lines = replace_token(bki_lines, "LC_CTYPE",
							  escape_quotes_bki(lc_ctype));

	bki_lines = replace_token(bki_lines, "ICU_LOCALE",
							  icu_locale ? escape_quotes_bki(icu_locale) : "_null_");

	bki_lines = replace_token(bki_lines, "ICU_RULES",
							  icu_rules ? escape_quotes_bki(icu_rules) : "_null_");

	sprintf(buf, "%c", locale_provider);
	bki_lines = replace_token(bki_lines, "LOCALE_PROVIDER", buf);

	/* Also ensure backend isn't confused by this environment var: */
	unsetenv("PGCLIENTENCODING");

	snprintf(cmd, sizeof(cmd),
			 "\"%s\" --boot -X %d %s %s %s %s",
			 backend_exec,
			 wal_segment_size_mb * (1024 * 1024),
			 data_checksums ? "-k" : "",
			 boot_options, extra_options,
			 debug ? "-d 5" : "");


	PG_CMD_OPEN;

	for (line = bki_lines; *line != NULL; line++)
	{
		PG_CMD_PUTS(*line);
		free(*line);
	}

	PG_CMD_CLOSE;

	free(bki_lines);

	check_ok();
}

/*
 * set up the shadow password table
 */
static void
setup_auth(FILE *cmdfd)
{
	/*
	 * The authid table shouldn't be readable except through views, to ensure
	 * passwords are not publicly visible.
	 */
	PG_CMD_PUTS("REVOKE ALL ON pg_authid FROM public;\n\n");

	if (superuser_password)
		PG_CMD_PRINTF("ALTER USER \"%s\" WITH PASSWORD E'%s';\n\n",
					  username, escape_quotes(superuser_password));
}

/*
 * get the superuser password if required
 */
static void
get_su_pwd(void)
{
	char	   *pwd1;

	if (pwprompt)
	{
		/*
		 * Read password from terminal
		 */
		char	   *pwd2;

		printf("\n");
		fflush(stdout);
		pwd1 = simple_prompt("Enter new superuser password: ", false);
		pwd2 = simple_prompt("Enter it again: ", false);
		if (strcmp(pwd1, pwd2) != 0)
		{
			fprintf(stderr, _("Passwords didn't match.\n"));
			exit(1);
		}
		free(pwd2);
	}
	else
	{
		/*
		 * Read password from file
		 *
		 * Ideally this should insist that the file not be world-readable.
		 * However, this option is mainly intended for use on Windows where
		 * file permissions may not exist at all, so we'll skip the paranoia
		 * for now.
		 */
		FILE	   *pwf = fopen(pwfilename, "r");

		if (!pwf)
			pg_fatal("could not open file \"%s\" for reading: %m",
					 pwfilename);
		pwd1 = pg_get_line(pwf, NULL);
		if (!pwd1)
		{
			if (ferror(pwf))
				pg_fatal("could not read password from file \"%s\": %m",
						 pwfilename);
			else
				pg_fatal("password file \"%s\" is empty",
						 pwfilename);
		}
		fclose(pwf);

		(void) pg_strip_crlf(pwd1);
	}

	superuser_password = pwd1;
}

/*
 * set up pg_depend
 */
static void
setup_depend(FILE *cmdfd)
{
	/*
	 * Advance the OID counter so that subsequently-created objects aren't
	 * pinned.
	 */
	PG_CMD_PUTS("SELECT pg_stop_making_pinned_objects();\n\n");
}

/*
 * Run external file
 */
static void
setup_run_file(FILE *cmdfd, const char *filename)
{
	char	  **lines;

	lines = readfile(filename);

	for (char **line = lines; *line != NULL; line++)
	{
		PG_CMD_PUTS(*line);
		free(*line);
	}

	PG_CMD_PUTS("\n\n");

	free(lines);
}

/*
 * fill in extra description data
 */
static void
setup_description(FILE *cmdfd)
{
	/* Create default descriptions for operator implementation functions */
	PG_CMD_PUTS("WITH funcdescs AS ( "
				"SELECT p.oid as p_oid, o.oid as o_oid, oprname "
				"FROM pg_proc p JOIN pg_operator o ON oprcode = p.oid ) "
				"INSERT INTO pg_description "
				"  SELECT p_oid, 'pg_proc'::regclass, 0, "
				"    'implementation of ' || oprname || ' operator' "
				"  FROM funcdescs "
				"  WHERE NOT EXISTS (SELECT 1 FROM pg_description "
				"   WHERE objoid = p_oid AND classoid = 'pg_proc'::regclass) "
				"  AND NOT EXISTS (SELECT 1 FROM pg_description "
				"   WHERE objoid = o_oid AND classoid = 'pg_operator'::regclass"
				"         AND description LIKE 'deprecated%');\n\n");
}

/*
 * populate pg_collation
 */
static void
setup_collation(FILE *cmdfd)
{
	/*
	 * Set the collation version for collations defined in pg_collation.dat,
	 * but not the ones where we know that the collation behavior will never
	 * change.
	 */
	PG_CMD_PUTS("UPDATE pg_collation SET collversion = pg_collation_actual_version(oid) WHERE collname = 'unicode';\n\n");

	/* Import all collations we can find in the operating system */
	PG_CMD_PUTS("SELECT pg_import_system_collations('pg_catalog');\n\n");
}

/*
 * Set up privileges
 *
 * We mark most system catalogs as world-readable.  We don't currently have
 * to touch functions, languages, or databases, because their default
 * permissions are OK.
 *
 * Some objects may require different permissions by default, so we
 * make sure we don't overwrite privilege sets that have already been
 * set (NOT NULL).
 *
 * Also populate pg_init_privs to save what the privileges are at init
 * time.  This is used by pg_dump to allow users to change privileges
 * on catalog objects and to have those privilege changes preserved
 * across dump/reload and pg_upgrade.
 *
 * Note that pg_init_privs is only for per-database objects and therefore
 * we don't include databases or tablespaces.
 */
static void
setup_privileges(FILE *cmdfd)
{
	PG_CMD_PRINTF("UPDATE pg_class "
				  "  SET relacl = (SELECT array_agg(a.acl) FROM "
				  " (SELECT E'=r/\"%s\"' as acl "
				  "  UNION SELECT unnest(pg_catalog.acldefault("
				  "    CASE WHEN relkind = " CppAsString2(RELKIND_SEQUENCE) " THEN 's' "
				  "         ELSE 'r' END::\"char\"," CppAsString2(BOOTSTRAP_SUPERUSERID) "::oid))"
				  " ) as a) "
				  "  WHERE relkind IN (" CppAsString2(RELKIND_RELATION) ", "
				  CppAsString2(RELKIND_VIEW) ", " CppAsString2(RELKIND_MATVIEW) ", "
				  CppAsString2(RELKIND_SEQUENCE) ")"
				  "  AND relacl IS NULL;\n\n",
				  escape_quotes(username));
	PG_CMD_PUTS("GRANT USAGE ON SCHEMA pg_catalog, public TO PUBLIC;\n\n");
	PG_CMD_PUTS("REVOKE ALL ON pg_largeobject FROM PUBLIC;\n\n");
	PG_CMD_PUTS("INSERT INTO pg_init_privs "
				"  (objoid, classoid, objsubid, initprivs, privtype)"
				"    SELECT"
				"        oid,"
				"        (SELECT oid FROM pg_class WHERE relname = 'pg_class'),"
				"        0,"
				"        relacl,"
				"        'i'"
				"    FROM"
				"        pg_class"
				"    WHERE"
				"        relacl IS NOT NULL"
				"        AND relkind IN (" CppAsString2(RELKIND_RELATION) ", "
				CppAsString2(RELKIND_VIEW) ", " CppAsString2(RELKIND_MATVIEW) ", "
				CppAsString2(RELKIND_SEQUENCE) ");\n\n");
	PG_CMD_PUTS("INSERT INTO pg_init_privs "
				"  (objoid, classoid, objsubid, initprivs, privtype)"
				"    SELECT"
				"        pg_class.oid,"
				"        (SELECT oid FROM pg_class WHERE relname = 'pg_class'),"
				"        pg_attribute.attnum,"
				"        pg_attribute.attacl,"
				"        'i'"
				"    FROM"
				"        pg_class"
				"        JOIN pg_attribute ON (pg_class.oid = pg_attribute.attrelid)"
				"    WHERE"
				"        pg_attribute.attacl IS NOT NULL"
				"        AND pg_class.relkind IN (" CppAsString2(RELKIND_RELATION) ", "
				CppAsString2(RELKIND_VIEW) ", " CppAsString2(RELKIND_MATVIEW) ", "
				CppAsString2(RELKIND_SEQUENCE) ");\n\n");
	PG_CMD_PUTS("INSERT INTO pg_init_privs "
				"  (objoid, classoid, objsubid, initprivs, privtype)"
				"    SELECT"
				"        oid,"
				"        (SELECT oid FROM pg_class WHERE relname = 'pg_proc'),"
				"        0,"
				"        proacl,"
				"        'i'"
				"    FROM"
				"        pg_proc"
				"    WHERE"
				"        proacl IS NOT NULL;\n\n");
	PG_CMD_PUTS("INSERT INTO pg_init_privs "
				"  (objoid, classoid, objsubid, initprivs, privtype)"
				"    SELECT"
				"        oid,"
				"        (SELECT oid FROM pg_class WHERE relname = 'pg_type'),"
				"        0,"
				"        typacl,"
				"        'i'"
				"    FROM"
				"        pg_type"
				"    WHERE"
				"        typacl IS NOT NULL;\n\n");
	PG_CMD_PUTS("INSERT INTO pg_init_privs "
				"  (objoid, classoid, objsubid, initprivs, privtype)"
				"    SELECT"
				"        oid,"
				"        (SELECT oid FROM pg_class WHERE relname = 'pg_language'),"
				"        0,"
				"        lanacl,"
				"        'i'"
				"    FROM"
				"        pg_language"
				"    WHERE"
				"        lanacl IS NOT NULL;\n\n");
	PG_CMD_PUTS("INSERT INTO pg_init_privs "
				"  (objoid, classoid, objsubid, initprivs, privtype)"
				"    SELECT"
				"        oid,"
				"        (SELECT oid FROM pg_class WHERE "
				"         relname = 'pg_largeobject_metadata'),"
				"        0,"
				"        lomacl,"
				"        'i'"
				"    FROM"
				"        pg_largeobject_metadata"
				"    WHERE"
				"        lomacl IS NOT NULL;\n\n");
	PG_CMD_PUTS("INSERT INTO pg_init_privs "
				"  (objoid, classoid, objsubid, initprivs, privtype)"
				"    SELECT"
				"        oid,"
				"        (SELECT oid FROM pg_class WHERE relname = 'pg_namespace'),"
				"        0,"
				"        nspacl,"
				"        'i'"
				"    FROM"
				"        pg_namespace"
				"    WHERE"
				"        nspacl IS NOT NULL;\n\n");
	PG_CMD_PUTS("INSERT INTO pg_init_privs "
				"  (objoid, classoid, objsubid, initprivs, privtype)"
				"    SELECT"
				"        oid,"
				"        (SELECT oid FROM pg_class WHERE "
				"         relname = 'pg_foreign_data_wrapper'),"
				"        0,"
				"        fdwacl,"
				"        'i'"
				"    FROM"
				"        pg_foreign_data_wrapper"
				"    WHERE"
				"        fdwacl IS NOT NULL;\n\n");
	PG_CMD_PUTS("INSERT INTO pg_init_privs "
				"  (objoid, classoid, objsubid, initprivs, privtype)"
				"    SELECT"
				"        oid,"
				"        (SELECT oid FROM pg_class "
				"         WHERE relname = 'pg_foreign_server'),"
				"        0,"
				"        srvacl,"
				"        'i'"
				"    FROM"
				"        pg_foreign_server"
				"    WHERE"
				"        srvacl IS NOT NULL;\n\n");
}

/*
 * extract the strange version of version required for information schema
 * (09.08.0007abc)
 */
static void
set_info_version(void)
{
	char	   *letterversion;
	long		major = 0,
				minor = 0,
				micro = 0;
	char	   *endptr;
	char	   *vstr = pg_strdup(PG_VERSION);
	char	   *ptr;

	ptr = vstr + (strlen(vstr) - 1);
	while (ptr != vstr && (*ptr < '0' || *ptr > '9'))
		ptr--;
	letterversion = ptr + 1;
	major = strtol(vstr, &endptr, 10);
	if (*endptr)
		minor = strtol(endptr + 1, &endptr, 10);
	if (*endptr)
		micro = strtol(endptr + 1, &endptr, 10);
	snprintf(infoversion, sizeof(infoversion), "%02ld.%02ld.%04ld%s",
			 major, minor, micro, letterversion);
}

/*
 * load info schema and populate from features file
 */
static void
setup_schema(FILE *cmdfd)
{
	setup_run_file(cmdfd, info_schema_file);

	PG_CMD_PRINTF("UPDATE information_schema.sql_implementation_info "
				  "  SET character_value = '%s' "
				  "  WHERE implementation_info_name = 'DBMS VERSION';\n\n",
				  infoversion);

	PG_CMD_PRINTF("COPY information_schema.sql_features "
				  "  (feature_id, feature_name, sub_feature_id, "
				  "  sub_feature_name, is_supported, comments) "
				  " FROM E'%s';\n\n",
				  escape_quotes(features_file));
}

/*
 * load PL/pgSQL server-side language
 */
static void
load_plpgsql(FILE *cmdfd)
{
	PG_CMD_PUTS("CREATE EXTENSION plpgsql;\n\n");
}

/*
 * clean everything up in template1
 */
static void
vacuum_db(FILE *cmdfd)
{
	/* Run analyze before VACUUM so the statistics are frozen. */
	PG_CMD_PUTS("ANALYZE;\n\nVACUUM FREEZE;\n\n");
}

/*
 * copy template1 to template0
 */
static void
make_template0(FILE *cmdfd)
{
	/*
	 * pg_upgrade tries to preserve database OIDs across upgrades. It's smart
	 * enough to drop and recreate a conflicting database with the same name,
	 * but if the same OID were used for one system-created database in the
	 * old cluster and a different system-created database in the new cluster,
	 * it would fail. To avoid that, assign a fixed OID to template0 rather
	 * than letting the server choose one.
	 *
	 * (Note that, while the user could have dropped and recreated these
	 * objects in the old cluster, the problem scenario only exists if the OID
	 * that is in use in the old cluster is also used in the new cluster - and
	 * the new cluster should be the result of a fresh initdb.)
	 *
	 * We use "STRATEGY = file_copy" here because checkpoints during initdb
	 * are cheap. "STRATEGY = wal_log" would generate more WAL, which would be
	 * a little bit slower and make the new cluster a little bit bigger.
	 */
	PG_CMD_PUTS("CREATE DATABASE template0 IS_TEMPLATE = true ALLOW_CONNECTIONS = false"
				" OID = " CppAsString2(Template0DbOid)
				" STRATEGY = file_copy;\n\n");

	/*
	 * template0 shouldn't have any collation-dependent objects, so unset the
	 * collation version.  This disables collation version checks when making
	 * a new database from it.
	 */
	PG_CMD_PUTS("UPDATE pg_database SET datcollversion = NULL WHERE datname = 'template0';\n\n");

	/*
	 * While we are here, do set the collation version on template1.
	 */
	PG_CMD_PUTS("UPDATE pg_database SET datcollversion = pg_database_collation_actual_version(oid) WHERE datname = 'template1';\n\n");

	/*
	 * Explicitly revoke public create-schema and create-temp-table privileges
	 * in template1 and template0; else the latter would be on by default
	 */
	PG_CMD_PUTS("REVOKE CREATE,TEMPORARY ON DATABASE template1 FROM public;\n\n");
	PG_CMD_PUTS("REVOKE CREATE,TEMPORARY ON DATABASE template0 FROM public;\n\n");

	PG_CMD_PUTS("COMMENT ON DATABASE template0 IS 'unmodifiable empty database';\n\n");

	/*
	 * Finally vacuum to clean up dead rows in pg_database
	 */
	PG_CMD_PUTS("VACUUM pg_database;\n\n");
}

/*
 * copy template1 to neurdb
 */
static void
make_postgres(FILE *cmdfd)
{
	/*
	 * Just as we did for template0, and for the same reasons, assign a fixed
	 * OID to neurdb and select the file_copy strategy.
	 */
	PG_CMD_PUTS("CREATE DATABASE neurdb OID = " CppAsString2(PostgresDbOid)
				" STRATEGY = file_copy;\n\n");
	PG_CMD_PUTS("COMMENT ON DATABASE neurdb IS 'default administrative connection database';\n\n");
}

/*
 * signal handler in case we are interrupted.
 *
 * The Windows runtime docs at
 * https://docs.microsoft.com/en-us/cpp/c-runtime-library/reference/signal
 * specifically forbid a number of things being done from a signal handler,
 * including IO, memory allocation and system calls, and only allow jmpbuf
 * if you are handling SIGFPE.
 *
 * I avoided doing the forbidden things by setting a flag instead of calling
 * exit() directly.
 *
 * Also note the behaviour of Windows with SIGINT, which says this:
 *  SIGINT is not supported for any Win32 application. When a CTRL+C interrupt
 *  occurs, Win32 operating systems generate a new thread to specifically
 *  handle that interrupt. This can cause a single-thread application, such as
 *  one in UNIX, to become multithreaded and cause unexpected behavior.
 *
 * I have no idea how to handle this. (Strange they call UNIX an application!)
 * So this will need some testing on Windows.
 */
static void
trapsig(SIGNAL_ARGS)
{
	/* handle systems that reset the handler, like Windows (grr) */
	pqsignal(postgres_signal_arg, trapsig);
	caught_signal = true;
}

/*
 * call exit() if we got a signal, or else output "ok".
 */
static void
check_ok(void)
{
	if (caught_signal)
	{
		printf(_("caught signal\n"));
		fflush(stdout);
		exit(1);
	}
	else if (output_failed)
	{
		printf(_("could not write to child process: %s\n"),
			   strerror(output_errno));
		fflush(stdout);
		exit(1);
	}
	else
	{
		/* all seems well */
		printf(_("ok\n"));
		fflush(stdout);
	}
}

/* Hack to suppress a warning about %x from some versions of gcc */
static inline size_t
my_strftime(char *s, size_t max, const char *fmt, const struct tm *tm)
{
	return strftime(s, max, fmt, tm);
}

/*
 * Determine likely date order from locale
 */
static int
locale_date_order(const char *locale)
{
	struct tm	testtime;
	char		buf[128];
	char	   *posD;
	char	   *posM;
	char	   *posY;
	char	   *save;
	size_t		res;
	int			result;

	result = DATEORDER_MDY;		/* default */

	save = setlocale(LC_TIME, NULL);
	if (!save)
		return result;
	save = pg_strdup(save);

	setlocale(LC_TIME, locale);

	memset(&testtime, 0, sizeof(testtime));
	testtime.tm_mday = 22;
	testtime.tm_mon = 10;		/* November, should come out as "11" */
	testtime.tm_year = 133;		/* 2033 */

	res = my_strftime(buf, sizeof(buf), "%x", &testtime);

	setlocale(LC_TIME, save);
	free(save);

	if (res == 0)
		return result;

	posM = strstr(buf, "11");
	posD = strstr(buf, "22");
	posY = strstr(buf, "33");

	if (!posM || !posD || !posY)
		return result;

	if (posY < posM && posM < posD)
		result = DATEORDER_YMD;
	else if (posD < posM)
		result = DATEORDER_DMY;
	else
		result = DATEORDER_MDY;

	return result;
}

/*
 * Verify that locale name is valid for the locale category.
 *
 * If successful, and canonname isn't NULL, a malloc'd copy of the locale's
 * canonical name is stored there.  This is especially useful for figuring out
 * what locale name "" means (ie, the environment value).  (Actually,
 * it seems that on most implementations that's the only thing it's good for;
 * we could wish that setlocale gave back a canonically spelled version of
 * the locale name, but typically it doesn't.)
 *
 * this should match the backend's check_locale() function
 */
static void
check_locale_name(int category, const char *locale, char **canonname)
{
	char	   *save;
	char	   *res;

	if (canonname)
		*canonname = NULL;		/* in case of failure */

	save = setlocale(category, NULL);
	if (!save)
		pg_fatal("setlocale() failed");

	/* save may be pointing at a modifiable scratch variable, so copy it. */
	save = pg_strdup(save);

	/* for setlocale() call */
	if (!locale)
		locale = "";

	/* set the locale with setlocale, to see if it accepts it. */
	res = setlocale(category, locale);

	/* save canonical name if requested. */
	if (res && canonname)
		*canonname = pg_strdup(res);

	/* restore old value. */
	if (!setlocale(category, save))
		pg_fatal("failed to restore old locale \"%s\"", save);
	free(save);

	/* complain if locale wasn't valid */
	if (res == NULL)
	{
		if (*locale)
		{
			pg_log_error("invalid locale name \"%s\"", locale);
			pg_log_error_hint("If the locale name is specific to ICU, use --icu-locale.");
			exit(1);
		}
		else
		{
			/*
			 * If no relevant switch was given on command line, locale is an
			 * empty string, which is not too helpful to report.  Presumably
			 * setlocale() found something it did not like in the environment.
			 * Ideally we'd report the bad environment variable, but since
			 * setlocale's behavior is implementation-specific, it's hard to
			 * be sure what it didn't like.  Print a safe generic message.
			 */
			pg_fatal("invalid locale settings; check LANG and LC_* environment variables");
		}
	}
}

/*
 * check if the chosen encoding matches the encoding required by the locale
 *
 * this should match the similar check in the backend createdb() function
 */
static bool
check_locale_encoding(const char *locale, int user_enc)
{
	int			locale_enc;

	locale_enc = pg_get_encoding_from_locale(locale, true);

	/* See notes in createdb() to understand these tests */
	if (!(locale_enc == user_enc ||
		  locale_enc == PG_SQL_ASCII ||
		  locale_enc == -1 ||
#ifdef WIN32
		  user_enc == PG_UTF8 ||
#endif
		  user_enc == PG_SQL_ASCII))
	{
		pg_log_error("encoding mismatch");
		pg_log_error_detail("The encoding you selected (%s) and the encoding that the "
							"selected locale uses (%s) do not match. This would lead to "
							"misbehavior in various character string processing functions.",
							pg_encoding_to_char(user_enc),
							pg_encoding_to_char(locale_enc));
		pg_log_error_hint("Rerun %s and either do not specify an encoding explicitly, "
						  "or choose a matching combination.",
						  progname);
		return false;
	}
	return true;
}

/*
 * check if the chosen encoding matches is supported by ICU
 *
 * this should match the similar check in the backend createdb() function
 */
static bool
check_icu_locale_encoding(int user_enc)
{
	if (!(is_encoding_supported_by_icu(user_enc)))
	{
		pg_log_error("encoding mismatch");
		pg_log_error_detail("The encoding you selected (%s) is not supported with the ICU provider.",
							pg_encoding_to_char(user_enc));
		pg_log_error_hint("Rerun %s and either do not specify an encoding explicitly, "
						  "or choose a matching combination.",
						  progname);
		return false;
	}
	return true;
}

/*
 * Convert to canonical BCP47 language tag. Must be consistent with
 * icu_language_tag().
 */
static char *
icu_language_tag(const char *loc_str)
{
#ifdef USE_ICU
	UErrorCode	status;
	char	   *langtag;
	size_t		buflen = 32;	/* arbitrary starting buffer size */
	const bool	strict = true;

	/*
	 * A BCP47 language tag doesn't have a clearly-defined upper limit (cf.
	 * RFC5646 section 4.4). Additionally, in older ICU versions,
	 * uloc_toLanguageTag() doesn't always return the ultimate length on the
	 * first call, necessitating a loop.
	 */
	langtag = pg_malloc(buflen);
	while (true)
	{
		status = U_ZERO_ERROR;
		uloc_toLanguageTag(loc_str, langtag, buflen, strict, &status);

		/* try again if the buffer is not large enough */
		if (status == U_BUFFER_OVERFLOW_ERROR ||
			status == U_STRING_NOT_TERMINATED_WARNING)
		{
			buflen = buflen * 2;
			langtag = pg_realloc(langtag, buflen);
			continue;
		}

		break;
	}

	if (U_FAILURE(status))
	{
		pg_free(langtag);

		pg_fatal("could not convert locale name \"%s\" to language tag: %s",
				 loc_str, u_errorName(status));
	}

	return langtag;
#else
	pg_fatal("ICU is not supported in this build");
	return NULL;				/* keep compiler quiet */
#endif
}

/*
 * Perform best-effort check that the locale is a valid one. Should be
 * consistent with pg_locale.c, except that it doesn't need to open the
 * collator (that will happen during post-bootstrap initialization).
 */
static void
icu_validate_locale(const char *loc_str)
{
#ifdef USE_ICU
	UErrorCode	status;
	char		lang[ULOC_LANG_CAPACITY];
	bool		found = false;

	/* validate that we can extract the language */
	status = U_ZERO_ERROR;
	uloc_getLanguage(loc_str, lang, ULOC_LANG_CAPACITY, &status);
	if (U_FAILURE(status))
	{
		pg_fatal("could not get language from locale \"%s\": %s",
				 loc_str, u_errorName(status));
		return;
	}

	/* check for special language name */
	if (strcmp(lang, "") == 0 ||
		strcmp(lang, "root") == 0 || strcmp(lang, "und") == 0)
		found = true;

	/* search for matching language within ICU */
	for (int32_t i = 0; !found && i < uloc_countAvailable(); i++)
	{
		const char *otherloc = uloc_getAvailable(i);
		char		otherlang[ULOC_LANG_CAPACITY];

		status = U_ZERO_ERROR;
		uloc_getLanguage(otherloc, otherlang, ULOC_LANG_CAPACITY, &status);
		if (U_FAILURE(status))
			continue;

		if (strcmp(lang, otherlang) == 0)
			found = true;
	}

	if (!found)
		pg_fatal("locale \"%s\" has unknown language \"%s\"",
				 loc_str, lang);
#else
	pg_fatal("ICU is not supported in this build");
#endif
}

/*
 * set up the locale variables
 *
 * assumes we have called setlocale(LC_ALL, "") -- see set_pglocale_pgservice
 */
static void
setlocales(void)
{
	char	   *canonname;

	/* set empty lc_* and iculocale values to locale config if set */

	if (locale)
	{
		if (!lc_ctype)
			lc_ctype = locale;
		if (!lc_collate)
			lc_collate = locale;
		if (!lc_numeric)
			lc_numeric = locale;
		if (!lc_time)
			lc_time = locale;
		if (!lc_monetary)
			lc_monetary = locale;
		if (!lc_messages)
			lc_messages = locale;
		if (!icu_locale && locale_provider == COLLPROVIDER_ICU)
			icu_locale = locale;
	}

	/*
	 * canonicalize locale names, and obtain any missing values from our
	 * current environment
	 */
	check_locale_name(LC_CTYPE, lc_ctype, &canonname);
	lc_ctype = canonname;
	check_locale_name(LC_COLLATE, lc_collate, &canonname);
	lc_collate = canonname;
	check_locale_name(LC_NUMERIC, lc_numeric, &canonname);
	lc_numeric = canonname;
	check_locale_name(LC_TIME, lc_time, &canonname);
	lc_time = canonname;
	check_locale_name(LC_MONETARY, lc_monetary, &canonname);
	lc_monetary = canonname;
#if defined(LC_MESSAGES) && !defined(WIN32)
	check_locale_name(LC_MESSAGES, lc_messages, &canonname);
	lc_messages = canonname;
#else
	/* when LC_MESSAGES is not available, use the LC_CTYPE setting */
	check_locale_name(LC_CTYPE, lc_messages, &canonname);
	lc_messages = canonname;
#endif

	if (locale_provider == COLLPROVIDER_ICU)
	{
		char	   *langtag;

		/* acquire default locale from the environment, if not specified */
		if (icu_locale == NULL)
			pg_fatal("ICU locale must be specified");

		/* canonicalize to a language tag */
		langtag = icu_language_tag(icu_locale);
		printf(_("Using language tag \"%s\" for ICU locale \"%s\".\n"),
			   langtag, icu_locale);
		pg_free(icu_locale);
		icu_locale = langtag;

		icu_validate_locale(icu_locale);

		/*
		 * In supported builds, the ICU locale ID will be opened during
		 * post-bootstrap initialization, which will perform extra checks.
		 */
#ifndef USE_ICU
		pg_fatal("ICU is not supported in this build");
#endif
	}
}

/*
 * print help text
 */
static void
usage(const char *progname)
{
	printf(_("%s initializes a PostgreSQL database cluster.\n\n"), progname);
	printf(_("Usage:\n"));
	printf(_("  %s [OPTION]... [DATADIR]\n"), progname);
	printf(_("\nOptions:\n"));
	printf(_("  -A, --auth=METHOD         default authentication method for local connections\n"));
	printf(_("      --auth-host=METHOD    default authentication method for local TCP/IP connections\n"));
	printf(_("      --auth-local=METHOD   default authentication method for local-socket connections\n"));
	printf(_(" [-D, --pgdata=]DATADIR     location for this database cluster\n"));
	printf(_("  -E, --encoding=ENCODING   set default encoding for new databases\n"));
	printf(_("  -g, --allow-group-access  allow group read/execute on data directory\n"));
	printf(_("      --icu-locale=LOCALE   set ICU locale ID for new databases\n"));
	printf(_("      --icu-rules=RULES     set additional ICU collation rules for new databases\n"));
	printf(_("  -k, --data-checksums      use data page checksums\n"));
	printf(_("      --locale=LOCALE       set default locale for new databases\n"));
	printf(_("      --lc-collate=, --lc-ctype=, --lc-messages=LOCALE\n"
			 "      --lc-monetary=, --lc-numeric=, --lc-time=LOCALE\n"
			 "                            set default locale in the respective category for\n"
			 "                            new databases (default taken from environment)\n"));
	printf(_("      --no-locale           equivalent to --locale=C\n"));
	printf(_("      --locale-provider={libc|icu}\n"
			 "                            set default locale provider for new databases\n"));
	printf(_("      --pwfile=FILE         read password for the new superuser from file\n"));
	printf(_("  -T, --text-search-config=CFG\n"
			 "                            default text search configuration\n"));
	printf(_("  -U, --username=NAME       database superuser name\n"));
	printf(_("  -W, --pwprompt            prompt for a password for the new superuser\n"));
	printf(_("  -X, --waldir=WALDIR       location for the write-ahead log directory\n"));
	printf(_("      --wal-segsize=SIZE    size of WAL segments, in megabytes\n"));
	printf(_("\nLess commonly used options:\n"));
	printf(_("  -c, --set NAME=VALUE      override default setting for server parameter\n"));
	printf(_("  -d, --debug               generate lots of debugging output\n"));
	printf(_("      --discard-caches      set debug_discard_caches=1\n"));
	printf(_("  -L DIRECTORY              where to find the input files\n"));
	printf(_("  -n, --no-clean            do not clean up after errors\n"));
	printf(_("  -N, --no-sync             do not wait for changes to be written safely to disk\n"));
	printf(_("      --no-instructions     do not print instructions for next steps\n"));
	printf(_("  -s, --show                show internal settings\n"));
	printf(_("  -S, --sync-only           only sync database files to disk, then exit\n"));
	printf(_("\nOther options:\n"));
	printf(_("  -V, --version             output version information, then exit\n"));
	printf(_("  -?, --help                show this help, then exit\n"));
	printf(_("\nIf the data directory is not specified, the environment variable PGDATA\n"
			 "is used.\n"));
	printf(_("\nReport bugs to <%s>.\n"), PACKAGE_BUGREPORT);
	printf(_("%s home page: <%s>\n"), PACKAGE_NAME, PACKAGE_URL);
}

static void
check_authmethod_unspecified(const char **authmethod)
{
	if (*authmethod == NULL)
	{
		authwarning = true;
		*authmethod = "trust";
	}
}

static void
check_authmethod_valid(const char *authmethod, const char *const *valid_methods, const char *conntype)
{
	const char *const *p;

	for (p = valid_methods; *p; p++)
	{
		if (strcmp(authmethod, *p) == 0)
			return;
		/* with space = param */
		if (strchr(authmethod, ' '))
			if (strncmp(authmethod, *p, (authmethod - strchr(authmethod, ' '))) == 0)
				return;
	}

	pg_fatal("invalid authentication method \"%s\" for \"%s\" connections",
			 authmethod, conntype);
}

static void
check_need_password(const char *authmethodlocal, const char *authmethodhost)
{
	if ((strcmp(authmethodlocal, "md5") == 0 ||
		 strcmp(authmethodlocal, "password") == 0 ||
		 strcmp(authmethodlocal, "scram-sha-256") == 0) &&
		(strcmp(authmethodhost, "md5") == 0 ||
		 strcmp(authmethodhost, "password") == 0 ||
		 strcmp(authmethodhost, "scram-sha-256") == 0) &&
		!(pwprompt || pwfilename))
		pg_fatal("must specify a password for the superuser to enable password authentication");
}


void
setup_pgdata(void)
{
	char	   *pgdata_get_env;

	if (!pg_data)
	{
		pgdata_get_env = getenv("PGDATA");
		if (pgdata_get_env && strlen(pgdata_get_env))
		{
			/* PGDATA found */
			pg_data = pg_strdup(pgdata_get_env);
		}
		else
		{
			pg_log_error("no data directory specified");
			pg_log_error_hint("You must identify the directory where the data for this database system "
							  "will reside.  Do this with either the invocation option -D or the "
							  "environment variable PGDATA.");
			exit(1);
		}
	}

	pgdata_native = pg_strdup(pg_data);
	canonicalize_path(pg_data);

	/*
	 * we have to set PGDATA for neurdb rather than pass it on the command
	 * line to avoid dumb quoting problems on Windows, and we would especially
	 * need quotes otherwise on Windows because paths there are most likely to
	 * have embedded spaces.
	 */
	if (setenv("PGDATA", pg_data, 1) != 0)
		pg_fatal("could not set environment");
}


void
setup_bin_paths(const char *argv0)
{
	int			ret;

	if ((ret = find_other_exec(argv0, "postgres", PG_BACKEND_VERSIONSTR,
							   backend_exec)) < 0)
	{
		char		full_path[MAXPGPATH];

		if (find_my_exec(argv0, full_path) < 0)
			strlcpy(full_path, progname, sizeof(full_path));

		if (ret == -1)
			pg_fatal("program \"%s\" is needed by %s but was not found in the same directory as \"%s\"",
					 "neurdb", progname, full_path);
		else
			pg_fatal("program \"%s\" was found by \"%s\" but was not the same version as %s",
					 "neurdb", full_path, progname);
	}

	/* store binary directory */
	strcpy(bin_path, backend_exec);
	*last_dir_separator(bin_path) = '\0';
	canonicalize_path(bin_path);

	if (!share_path)
	{
		share_path = pg_malloc(MAXPGPATH);
		get_share_path(backend_exec, share_path);
	}
	else if (!is_absolute_path(share_path))
		pg_fatal("input file location must be an absolute path");

	canonicalize_path(share_path);
}

void
setup_locale_encoding(void)
{
	setlocales();

	if (locale_provider == COLLPROVIDER_LIBC &&
		strcmp(lc_ctype, lc_collate) == 0 &&
		strcmp(lc_ctype, lc_time) == 0 &&
		strcmp(lc_ctype, lc_numeric) == 0 &&
		strcmp(lc_ctype, lc_monetary) == 0 &&
		strcmp(lc_ctype, lc_messages) == 0 &&
		(!icu_locale || strcmp(lc_ctype, icu_locale) == 0))
		printf(_("The database cluster will be initialized with locale \"%s\".\n"), lc_ctype);
	else
	{
		printf(_("The database cluster will be initialized with this locale configuration:\n"));
		printf(_("  provider:    %s\n"), collprovider_name(locale_provider));
		if (icu_locale)
			printf(_("  ICU locale:  %s\n"), icu_locale);
		printf(_("  LC_COLLATE:  %s\n"
				 "  LC_CTYPE:    %s\n"
				 "  LC_MESSAGES: %s\n"
				 "  LC_MONETARY: %s\n"
				 "  LC_NUMERIC:  %s\n"
				 "  LC_TIME:     %s\n"),
			   lc_collate,
			   lc_ctype,
			   lc_messages,
			   lc_monetary,
			   lc_numeric,
			   lc_time);
	}

	if (!encoding)
	{
		int			ctype_enc;

		ctype_enc = pg_get_encoding_from_locale(lc_ctype, true);

		/*
		 * If ctype_enc=SQL_ASCII, it's compatible with any encoding. ICU does
		 * not support SQL_ASCII, so select UTF-8 instead.
		 */
		if (locale_provider == COLLPROVIDER_ICU && ctype_enc == PG_SQL_ASCII)
			ctype_enc = PG_UTF8;

		if (ctype_enc == -1)
		{
			/* Couldn't recognize the locale's codeset */
			pg_log_error("could not find suitable encoding for locale \"%s\"",
						 lc_ctype);
			pg_log_error_hint("Rerun %s with the -E option.", progname);
			pg_log_error_hint("Try \"%s --help\" for more information.", progname);
			exit(1);
		}
		else if (!pg_valid_server_encoding_id(ctype_enc))
		{
			/*
			 * We recognized it, but it's not a legal server encoding. On
			 * Windows, UTF-8 works with any locale, so we can fall back to
			 * UTF-8.
			 */
#ifdef WIN32
			encodingid = PG_UTF8;
			printf(_("Encoding \"%s\" implied by locale is not allowed as a server-side encoding.\n"
					 "The default database encoding will be set to \"%s\" instead.\n"),
				   pg_encoding_to_char(ctype_enc),
				   pg_encoding_to_char(encodingid));
#else
			pg_log_error("locale \"%s\" requires unsupported encoding \"%s\"",
						 lc_ctype, pg_encoding_to_char(ctype_enc));
			pg_log_error_detail("Encoding \"%s\" is not allowed as a server-side encoding.",
								pg_encoding_to_char(ctype_enc));
			pg_log_error_hint("Rerun %s with a different locale selection.",
							  progname);
			exit(1);
#endif
		}
		else
		{
			encodingid = ctype_enc;
			printf(_("The default database encoding has accordingly been set to \"%s\".\n"),
				   pg_encoding_to_char(encodingid));
		}
	}
	else
		encodingid = get_encoding_id(encoding);

	if (!check_locale_encoding(lc_ctype, encodingid) ||
		!check_locale_encoding(lc_collate, encodingid))
		exit(1);				/* check_locale_encoding printed the error */

	if (locale_provider == COLLPROVIDER_ICU &&
		!check_icu_locale_encoding(encodingid))
		exit(1);
}


void
setup_data_file_paths(void)
{
	set_input(&bki_file, "postgres.bki");
	set_input(&hba_file, "pg_hba.conf.sample");
	set_input(&ident_file, "pg_ident.conf.sample");
	set_input(&conf_file, "postgresql.conf.sample");
	set_input(&dictionary_file, "snowball_create.sql");
	set_input(&info_schema_file, "information_schema.sql");
	set_input(&features_file, "sql_features.txt");
	set_input(&system_constraints_file, "system_constraints.sql");
	set_input(&system_functions_file, "system_functions.sql");
	set_input(&system_views_file, "system_views.sql");

	if (show_setting || debug)
	{
		fprintf(stderr,
				"VERSION=%s\n"
				"PGDATA=%s\nshare_path=%s\nPGPATH=%s\n"
				"POSTGRES_SUPERUSERNAME=%s\nPOSTGRES_BKI=%s\n"
				"POSTGRESQL_CONF_SAMPLE=%s\n"
				"PG_HBA_SAMPLE=%s\nPG_IDENT_SAMPLE=%s\n",
				PG_VERSION,
				pg_data, share_path, bin_path,
				username, bki_file,
				conf_file,
				hba_file, ident_file);
		if (show_setting)
			exit(0);
	}

	check_input(bki_file);
	check_input(hba_file);
	check_input(ident_file);
	check_input(conf_file);
	check_input(dictionary_file);
	check_input(info_schema_file);
	check_input(features_file);
	check_input(system_constraints_file);
	check_input(system_functions_file);
	check_input(system_views_file);
}


void
setup_text_search(void)
{
	if (!default_text_search_config)
	{
		default_text_search_config = find_matching_ts_config(lc_ctype);
		if (!default_text_search_config)
		{
			pg_log_info("could not find suitable text search configuration for locale \"%s\"",
						lc_ctype);
			default_text_search_config = "simple";
		}
	}
	else
	{
		const char *checkmatch = find_matching_ts_config(lc_ctype);

		if (checkmatch == NULL)
		{
			pg_log_warning("suitable text search configuration for locale \"%s\" is unknown",
						   lc_ctype);
		}
		else if (strcmp(checkmatch, default_text_search_config) != 0)
		{
			pg_log_warning("specified text search configuration \"%s\" might not match locale \"%s\"",
						   default_text_search_config, lc_ctype);
		}
	}

	printf(_("The default text search configuration will be set to \"%s\".\n"),
		   default_text_search_config);
}


void
setup_signals(void)
{
	/* some of these are not valid on Windows */
#ifdef SIGHUP
	pqsignal(SIGHUP, trapsig);
#endif
#ifdef SIGINT
	pqsignal(SIGINT, trapsig);
#endif
#ifdef SIGQUIT
	pqsignal(SIGQUIT, trapsig);
#endif
#ifdef SIGTERM
	pqsignal(SIGTERM, trapsig);
#endif

	/* Ignore SIGPIPE when writing to backend, so we can clean up */
#ifdef SIGPIPE
	pqsignal(SIGPIPE, SIG_IGN);
#endif

	/* Prevent SIGSYS so we can probe for kernel calls that might not work */
#ifdef SIGSYS
	pqsignal(SIGSYS, SIG_IGN);
#endif
}


void
create_data_directory(void)
{
	int			ret;

	switch ((ret = pg_check_dir(pg_data)))
	{
		case 0:
			/* PGDATA not there, must create it */
			printf(_("creating directory %s ... "),
				   pg_data);
			fflush(stdout);

			if (pg_mkdir_p(pg_data, pg_dir_create_mode) != 0)
				pg_fatal("could not create directory \"%s\": %m", pg_data);
			else
				check_ok();

			made_new_pgdata = true;
			break;

		case 1:
			/* Present but empty, fix permissions and use it */
			printf(_("fixing permissions on existing directory %s ... "),
				   pg_data);
			fflush(stdout);

			if (chmod(pg_data, pg_dir_create_mode) != 0)
				pg_fatal("could not change permissions of directory \"%s\": %m",
						 pg_data);
			else
				check_ok();

			found_existing_pgdata = true;
			break;

		case 2:
		case 3:
		case 4:
			/* Present and not empty */
			pg_log_error("directory \"%s\" exists but is not empty", pg_data);
			if (ret != 4)
				warn_on_mount_point(ret);
			else
				pg_log_error_hint("If you want to create a new database system, either remove or empty "
								  "the directory \"%s\" or run %s "
								  "with an argument other than \"%s\".",
								  pg_data, progname, pg_data);
			exit(1);			/* no further message needed */

		default:
			/* Trouble accessing directory */
			pg_fatal("could not access directory \"%s\": %m", pg_data);
	}
}


/* Create WAL directory, and symlink if required */
void
create_xlog_or_symlink(void)
{
	char	   *subdirloc;

	/* form name of the place for the subdirectory or symlink */
	subdirloc = psprintf("%s/pg_wal", pg_data);

	if (xlog_dir)
	{
		int			ret;

		/* clean up xlog directory name, check it's absolute */
		canonicalize_path(xlog_dir);
		if (!is_absolute_path(xlog_dir))
			pg_fatal("WAL directory location must be an absolute path");

		/* check if the specified xlog directory exists/is empty */
		switch ((ret = pg_check_dir(xlog_dir)))
		{
			case 0:
				/* xlog directory not there, must create it */
				printf(_("creating directory %s ... "),
					   xlog_dir);
				fflush(stdout);

				if (pg_mkdir_p(xlog_dir, pg_dir_create_mode) != 0)
					pg_fatal("could not create directory \"%s\": %m",
							 xlog_dir);
				else
					check_ok();

				made_new_xlogdir = true;
				break;

			case 1:
				/* Present but empty, fix permissions and use it */
				printf(_("fixing permissions on existing directory %s ... "),
					   xlog_dir);
				fflush(stdout);

				if (chmod(xlog_dir, pg_dir_create_mode) != 0)
					pg_fatal("could not change permissions of directory \"%s\": %m",
							 xlog_dir);
				else
					check_ok();

				found_existing_xlogdir = true;
				break;

			case 2:
			case 3:
			case 4:
				/* Present and not empty */
				pg_log_error("directory \"%s\" exists but is not empty", xlog_dir);
				if (ret != 4)
					warn_on_mount_point(ret);
				else
					pg_log_error_hint("If you want to store the WAL there, either remove or empty the directory \"%s\".",
									  xlog_dir);
				exit(1);

			default:
				/* Trouble accessing directory */
				pg_fatal("could not access directory \"%s\": %m", xlog_dir);
		}

		if (symlink(xlog_dir, subdirloc) != 0)
			pg_fatal("could not create symbolic link \"%s\": %m",
					 subdirloc);
	}
	else
	{
		/* Without -X option, just make the subdirectory normally */
		if (mkdir(subdirloc, pg_dir_create_mode) < 0)
			pg_fatal("could not create directory \"%s\": %m",
					 subdirloc);
	}

	free(subdirloc);
}


void
warn_on_mount_point(int error)
{
	if (error == 2)
		pg_log_error_detail("It contains a dot-prefixed/invisible file, perhaps due to it being a mount point.");
	else if (error == 3)
		pg_log_error_detail("It contains a lost+found directory, perhaps due to it being a mount point.");

	pg_log_error_hint("Using a mount point directly as the data directory is not recommended.\n"
					  "Create a subdirectory under the mount point.");
}


void
initialize_data_directory(void)
{
	PG_CMD_DECL;
	int			i;

	setup_signals();

	/*
	 * Set mask based on requested PGDATA permissions.  pg_mode_mask, and
	 * friends like pg_dir_create_mode, are set to owner-only by default and
	 * then updated if -g is passed in by calling SetDataDirectoryCreatePerm()
	 * when parsing our options (see above).
	 */
	umask(pg_mode_mask);

	create_data_directory();

	create_xlog_or_symlink();

	/* Create required subdirectories (other than pg_wal) */
	printf(_("creating subdirectories ... "));
	fflush(stdout);

	for (i = 0; i < lengthof(subdirs); i++)
	{
		char	   *path;

		path = psprintf("%s/%s", pg_data, subdirs[i]);

		/*
		 * The parent directory already exists, so we only need mkdir() not
		 * pg_mkdir_p() here, which avoids some failure modes; cf bug #13853.
		 */
		if (mkdir(path, pg_dir_create_mode) < 0)
			pg_fatal("could not create directory \"%s\": %m", path);

		free(path);
	}

	check_ok();

	/* Top level PG_VERSION is checked by bootstrapper, so make it first */
	write_version_file(NULL);

	/* Select suitable configuration settings */
	set_null_conf();
	test_config_settings();

	/* Now create all the text config files */
	setup_config();

	/* Bootstrap template1 */
	bootstrap_template1();

	/*
	 * Make the per-database PG_VERSION for template1 only after init'ing it
	 */
	write_version_file("base/1");

	/*
	 * Create the stuff we don't need to use bootstrap mode for, using a
	 * backend running in simple standalone mode.
	 */
	fputs(_("performing post-bootstrap initialization ... "), stdout);
	fflush(stdout);

	snprintf(cmd, sizeof(cmd),
			 "\"%s\" %s %s template1 >%s",
			 backend_exec, backend_options, extra_options,
			 DEVNULL);

	PG_CMD_OPEN;

	setup_auth(cmdfd);

	setup_run_file(cmdfd, system_constraints_file);

	setup_run_file(cmdfd, system_functions_file);

	setup_depend(cmdfd);

	/*
	 * Note that no objects created after setup_depend() will be "pinned".
	 * They are all droppable at the whim of the DBA.
	 */

	setup_run_file(cmdfd, system_views_file);

	setup_description(cmdfd);

	setup_collation(cmdfd);

	setup_run_file(cmdfd, dictionary_file);

	setup_privileges(cmdfd);

	setup_schema(cmdfd);

	load_plpgsql(cmdfd);

	vacuum_db(cmdfd);

	make_template0(cmdfd);

	make_postgres(cmdfd);

	PG_CMD_CLOSE;

	check_ok();
}


int
main(int argc, char *argv[])
{
	static struct option long_options[] = {
		{"pgdata", required_argument, NULL, 'D'},
		{"encoding", required_argument, NULL, 'E'},
		{"locale", required_argument, NULL, 1},
		{"lc-collate", required_argument, NULL, 2},
		{"lc-ctype", required_argument, NULL, 3},
		{"lc-monetary", required_argument, NULL, 4},
		{"lc-numeric", required_argument, NULL, 5},
		{"lc-time", required_argument, NULL, 6},
		{"lc-messages", required_argument, NULL, 7},
		{"no-locale", no_argument, NULL, 8},
		{"text-search-config", required_argument, NULL, 'T'},
		{"auth", required_argument, NULL, 'A'},
		{"auth-local", required_argument, NULL, 10},
		{"auth-host", required_argument, NULL, 11},
		{"pwprompt", no_argument, NULL, 'W'},
		{"pwfile", required_argument, NULL, 9},
		{"username", required_argument, NULL, 'U'},
		{"help", no_argument, NULL, '?'},
		{"version", no_argument, NULL, 'V'},
		{"debug", no_argument, NULL, 'd'},
		{"show", no_argument, NULL, 's'},
		{"noclean", no_argument, NULL, 'n'},	/* for backwards compatibility */
		{"no-clean", no_argument, NULL, 'n'},
		{"nosync", no_argument, NULL, 'N'}, /* for backwards compatibility */
		{"no-sync", no_argument, NULL, 'N'},
		{"no-instructions", no_argument, NULL, 13},
		{"set", required_argument, NULL, 'c'},
		{"sync-only", no_argument, NULL, 'S'},
		{"waldir", required_argument, NULL, 'X'},
		{"wal-segsize", required_argument, NULL, 12},
		{"data-checksums", no_argument, NULL, 'k'},
		{"allow-group-access", no_argument, NULL, 'g'},
		{"discard-caches", no_argument, NULL, 14},
		{"locale-provider", required_argument, NULL, 15},
		{"icu-locale", required_argument, NULL, 16},
		{"icu-rules", required_argument, NULL, 17},
		{NULL, 0, NULL, 0}
	};

	/*
	 * options with no short version return a low integer, the rest return
	 * their short version value
	 */
	int			c;
	int			option_index;
	char	   *effective_user;
	PQExpBuffer start_db_cmd;
	char		pg_ctl_path[MAXPGPATH];

	/*
	 * Ensure that buffering behavior of stdout matches what it is in
	 * interactive usage (at least on most platforms).  This prevents
	 * unexpected output ordering when, eg, output is redirected to a file.
	 * POSIX says we must do this before any other usage of these files.
	 */
	setvbuf(stdout, NULL, PG_IOLBF, 0);

	pg_logging_init(argv[0]);
	progname = get_progname(argv[0]);
	set_pglocale_pgservice(argv[0], PG_TEXTDOMAIN("initdb"));

	if (argc > 1)
	{
		if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-?") == 0)
		{
			usage(progname);
			exit(0);
		}
		if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "-V") == 0)
		{
			puts("initdb (PostgreSQL) " PG_VERSION);
			exit(0);
		}
	}

	/* process command-line options */

	while ((c = getopt_long(argc, argv, "A:c:dD:E:gkL:nNsST:U:WX:",
							long_options, &option_index)) != -1)
	{
		switch (c)
		{
			case 'A':
				authmethodlocal = authmethodhost = pg_strdup(optarg);

				/*
				 * When ident is specified, use peer for local connections.
				 * Mirrored, when peer is specified, use ident for TCP/IP
				 * connections.
				 */
				if (strcmp(authmethodhost, "ident") == 0)
					authmethodlocal = "peer";
				else if (strcmp(authmethodlocal, "peer") == 0)
					authmethodhost = "ident";
				break;
			case 10:
				authmethodlocal = pg_strdup(optarg);
				break;
			case 11:
				authmethodhost = pg_strdup(optarg);
				break;
			case 'c':
				{
					char	   *buf = pg_strdup(optarg);
					char	   *equals = strchr(buf, '=');

					if (!equals)
					{
						pg_log_error("-c %s requires a value", buf);
						pg_log_error_hint("Try \"%s --help\" for more information.",
										  progname);
						exit(1);
					}
					*equals++ = '\0';	/* terminate variable name */
					add_stringlist_item(&extra_guc_names, buf);
					add_stringlist_item(&extra_guc_values, equals);
					pfree(buf);
				}
				break;
			case 'D':
				pg_data = pg_strdup(optarg);
				break;
			case 'E':
				encoding = pg_strdup(optarg);
				break;
			case 'W':
				pwprompt = true;
				break;
			case 'U':
				username = pg_strdup(optarg);
				break;
			case 'd':
				debug = true;
				printf(_("Running in debug mode.\n"));
				break;
			case 'n':
				noclean = true;
				printf(_("Running in no-clean mode.  Mistakes will not be cleaned up.\n"));
				break;
			case 'N':
				do_sync = false;
				break;
			case 'S':
				sync_only = true;
				break;
			case 'k':
				data_checksums = true;
				break;
			case 'L':
				share_path = pg_strdup(optarg);
				break;
			case 1:
				locale = pg_strdup(optarg);
				break;
			case 2:
				lc_collate = pg_strdup(optarg);
				break;
			case 3:
				lc_ctype = pg_strdup(optarg);
				break;
			case 4:
				lc_monetary = pg_strdup(optarg);
				break;
			case 5:
				lc_numeric = pg_strdup(optarg);
				break;
			case 6:
				lc_time = pg_strdup(optarg);
				break;
			case 7:
				lc_messages = pg_strdup(optarg);
				break;
			case 8:
				locale = "C";
				break;
			case 9:
				pwfilename = pg_strdup(optarg);
				break;
			case 's':
				show_setting = true;
				break;
			case 'T':
				default_text_search_config = pg_strdup(optarg);
				break;
			case 'X':
				xlog_dir = pg_strdup(optarg);
				break;
			case 12:
				str_wal_segment_size_mb = pg_strdup(optarg);
				break;
			case 13:
				noinstructions = true;
				break;
			case 'g':
				SetDataDirectoryCreatePerm(PG_DIR_MODE_GROUP);
				break;
			case 14:
				extra_options = psprintf("%s %s",
										 extra_options,
										 "-c debug_discard_caches=1");
				break;
			case 15:
				if (strcmp(optarg, "icu") == 0)
					locale_provider = COLLPROVIDER_ICU;
				else if (strcmp(optarg, "libc") == 0)
					locale_provider = COLLPROVIDER_LIBC;
				else
					pg_fatal("unrecognized locale provider: %s", optarg);
				break;
			case 16:
				icu_locale = pg_strdup(optarg);
				break;
			case 17:
				icu_rules = pg_strdup(optarg);
				break;
			default:
				/* getopt_long already emitted a complaint */
				pg_log_error_hint("Try \"%s --help\" for more information.", progname);
				exit(1);
		}
	}


	/*
	 * Non-option argument specifies data directory as long as it wasn't
	 * already specified with -D / --pgdata
	 */
	if (optind < argc && !pg_data)
	{
		pg_data = pg_strdup(argv[optind]);
		optind++;
	}

	if (optind < argc)
	{
		pg_log_error("too many command-line arguments (first is \"%s\")",
					 argv[optind]);
		pg_log_error_hint("Try \"%s --help\" for more information.", progname);
		exit(1);
	}

	if (icu_locale && locale_provider != COLLPROVIDER_ICU)
		pg_fatal("%s cannot be specified unless locale provider \"%s\" is chosen",
				 "--icu-locale", "icu");

	if (icu_rules && locale_provider != COLLPROVIDER_ICU)
		pg_fatal("%s cannot be specified unless locale provider \"%s\" is chosen",
				 "--icu-rules", "icu");

	atexit(cleanup_directories_atexit);

	/* If we only need to fsync, just do it and exit */
	if (sync_only)
	{
		setup_pgdata();

		/* must check that directory is readable */
		if (pg_check_dir(pg_data) <= 0)
			pg_fatal("could not access directory \"%s\": %m", pg_data);

		fputs(_("syncing data to disk ... "), stdout);
		fflush(stdout);
		fsync_pgdata(pg_data, PG_VERSION_NUM);
		check_ok();
		return 0;
	}

	if (pwprompt && pwfilename)
		pg_fatal("password prompt and password file cannot be specified together");

	check_authmethod_unspecified(&authmethodlocal);
	check_authmethod_unspecified(&authmethodhost);

	check_authmethod_valid(authmethodlocal, auth_methods_local, "local");
	check_authmethod_valid(authmethodhost, auth_methods_host, "host");

	check_need_password(authmethodlocal, authmethodhost);

	/* set wal segment size */
	if (str_wal_segment_size_mb == NULL)
		wal_segment_size_mb = (DEFAULT_XLOG_SEG_SIZE) / (1024 * 1024);
	else
	{
		char	   *endptr;

		/* check that the argument is a number */
		wal_segment_size_mb = strtol(str_wal_segment_size_mb, &endptr, 10);

		/* verify that wal segment size is valid */
		if (endptr == str_wal_segment_size_mb || *endptr != '\0')
			pg_fatal("argument of --wal-segsize must be a number");
		if (!IsValidWalSegSize(wal_segment_size_mb * 1024 * 1024))
			pg_fatal("argument of --wal-segsize must be a power of two between 1 and 1024");
	}

	get_restricted_token();

	setup_pgdata();

	setup_bin_paths(argv[0]);

	effective_user = get_id();
	if (!username)
		username = effective_user;

	if (strncmp(username, "pg_", 3) == 0)
		pg_fatal("superuser name \"%s\" is disallowed; role names cannot begin with \"pg_\"", username);

	printf(_("The files belonging to this database system will be owned "
			 "by user \"%s\".\n"
			 "This user must also own the server process.\n\n"),
		   effective_user);

	set_info_version();

	setup_data_file_paths();

	setup_locale_encoding();

	setup_text_search();

	printf("\n");

	if (data_checksums)
		printf(_("Data page checksums are enabled.\n"));
	else
		printf(_("Data page checksums are disabled.\n"));

	if (pwprompt || pwfilename)
		get_su_pwd();

	printf("\n");

	initialize_data_directory();

	if (do_sync)
	{
		fputs(_("syncing data to disk ... "), stdout);
		fflush(stdout);
		fsync_pgdata(pg_data, PG_VERSION_NUM);
		check_ok();
	}
	else
		printf(_("\nSync to disk skipped.\nThe data directory might become corrupt if the operating system crashes.\n"));

	if (authwarning)
	{
		printf("\n");
		pg_log_warning("enabling \"trust\" authentication for local connections");
		pg_log_warning_hint("You can change this by editing pg_hba.conf or using the option -A, or "
							"--auth-local and --auth-host, the next time you run initdb.");
	}

	if (!noinstructions)
	{
		/*
		 * Build up a shell command to tell the user how to start the server
		 */
		start_db_cmd = createPQExpBuffer();

		/* Get directory specification used to start initdb ... */
		strlcpy(pg_ctl_path, argv[0], sizeof(pg_ctl_path));
		canonicalize_path(pg_ctl_path);
		get_parent_directory(pg_ctl_path);
		/* ... and tag on pg_ctl instead */
		join_path_components(pg_ctl_path, pg_ctl_path, "pg_ctl");

		/* Convert the path to use native separators */
		make_native_path(pg_ctl_path);

		/* path to pg_ctl, properly quoted */
		appendShellString(start_db_cmd, pg_ctl_path);

		/* add -D switch, with properly quoted data directory */
		appendPQExpBufferStr(start_db_cmd, " -D ");
		appendShellString(start_db_cmd, pgdata_native);

		/* add suggested -l switch and "start" command */
		/* translator: This is a placeholder in a shell command. */
		appendPQExpBuffer(start_db_cmd, " -l %s start", _("logfile"));

		printf(_("\nSuccess. You can now start the database server using:\n\n"
				 "    %s\n\n"),
			   start_db_cmd->data);

		destroyPQExpBuffer(start_db_cmd);
	}


	success = true;
	return 0;
}
