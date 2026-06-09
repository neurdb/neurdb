#include "websocket.h"

#include <c.h>

#include "../cjson/cJSON.h"
#include "utils/elog.h"

/*  callback and handlers */
static int	callback(struct lws *wsi, enum lws_callback_reasons callback_reason,
					 void *user, void *input, size_t len);

static void handle_result(NrWebsocket * ws, const cJSON * json);

static void handle_ack_setup(NrWebsocket * ws, const cJSON * json);

static void handle_ack_disconnect(NrWebsocket * ws);

static void handle_ack_task(NrWebsocket * ws);

/*  message */
/*
 * Thread-safe outbound send. libwebsockets is single-threaded: lws_write must
 * only run on the service thread (the one calling lws_service). Producers on
 * the main backend thread therefore serialise the JSON, push it onto the
 * thread-safe queue, and wake the service thread (lws_cancel_service is the one
 * lws call documented as safe from another thread). The service thread drains
 * the queue from LWS_CALLBACK_CLIENT_WRITEABLE. Calling lws_write directly from
 * the main thread (the old behaviour) corrupted the connection once more than a
 * single batch was sent, which silently dropped the socket and hung the caller
 * in nws_wait_completion forever.
 */
static void queue_outbound(NrWebsocket * ws, const cJSON * json);

static void send_setup_signal(NrWebsocket * ws, size_t cache_size);

static void send_disconnect_signal(NrWebsocket * ws);

/*  websocket thread */
static void websocket_thread(void *arg);

/*  define the protocol in the websocket */
static const struct lws_protocols nws_protocol[] = {{
		"nws_protocol",
		callback,
		sizeof(NrWebsocket),
		0,
},
{NULL, NULL, 0, 0}};

/*  ****************************** Initialization, Connection, Disconnection */
/*  ****************************** */
NrWebsocket *
nws_initialize(const char *url, const int port, const char *path,
			   const size_t queue_max_size)
{
	NrWebsocket *websocket = (NrWebsocket *) malloc(sizeof(NrWebsocket));

	memset(websocket, 0, sizeof(NrWebsocket));

	init_batch_queue(&websocket->queue, queue_max_size);

	struct lws_context_creation_info info = {0};

	info.port = CONTEXT_PORT_NO_LISTEN;
	/* client side, no need to listen */
	info.protocols = nws_protocol;
	info.options = LWS_SERVER_OPTION_VALIDATE_UTF8;

	websocket->context = lws_create_context(&info);
	if (websocket->context == NULL)
	{
		elog(ERROR, "Websocket failure: context creation failed\n");
		lws_context_destroy(websocket->context);
		free(websocket);
		return NULL;
	}

	struct lws_client_connect_info connect_info = {0};

	connect_info.context = websocket->context;
	connect_info.address = url;
	connect_info.port = port;
	connect_info.path = path;
	connect_info.host = lws_canonical_hostname(websocket->context);
	connect_info.origin =
		lws_canonical_hostname(websocket->context);
	/* localhost connection */
	connect_info.protocol = nws_protocol[0].name;
	connect_info.pwsi = &websocket->instance;
	connect_info.userdata = websocket;

	websocket->instance = lws_client_connect_via_info(&connect_info);
	if (websocket->instance == NULL)
	{
		elog(ERROR, "Websocket failure: connection failed\n");
		lws_context_destroy(websocket->context);
		free(websocket);
		return NULL;
	}
	websocket->interrupted = 0;
	websocket->connnected = 0;
	websocket->setuped = 0;
	websocket->task_acknowledged = 0;
	return websocket;
}

int
nws_connect(NrWebsocket * ws)
{
	if (pthread_create(&ws->thread, NULL, (void *(*) (void *)) websocket_thread,
					   ws) != 0)
	{
		elog(ERROR, "Failed to create websocket thread\n");
		lws_context_destroy(ws->context);
		free(ws);
		return -1;
	}
	while (!ws->connnected && !ws->interrupted)
	{
		/* wait for the connection to be established */
		usleep(1000);
/* TODO: consider using a condition variable instead of busy */
		/* waiting */
	}
	if (ws->interrupted)
	{
		elog(ERROR, "AI engine connection failed before setup");
	}
	send_setup_signal(ws, ws->queue.max_size);
	while (!ws->setuped && !ws->interrupted)
	{
		/* wait for the setup to be acknowledged */
		usleep(1000);
/* TODO: consider using a condition variable instead of busy */
		/* waiting */
	}
	if (!ws->setuped && ws->interrupted)
	{
		elog(ERROR, "AI engine connection closed before setup was acknowledged");
	}
	return 0;
}

int
nws_disconnect(NrWebsocket * ws)
{
	send_disconnect_signal(ws);
	pthread_join(ws->thread, NULL);
	/* wait for the websocket thread to terminate */
	return 0;
}

void
nws_wait_completion(NrWebsocket * ws)
{
	while (!ws->completed && !ws->interrupted)
	{
		usleep(1000);
/* TODO: consider using a condition variable instead of busy */
		/* waiting */
	}
	if (!ws->completed && ws->interrupted)
	{
		/*
		 * The connection dropped before the server reported completion. Fail
		 * the query instead of spinning forever (the old loop only checked
		 * ``completed`` and would hang if the socket closed early).
		 */
		elog(ERROR, "AI engine connection closed before task completion");
	}
}

void
nws_free_websocket(NrWebsocket * ws)
{
	if (ws->thread)
	{
		pthread_join(ws->thread, NULL);
	}
	lws_context_destroy(ws->context);
	destroy_batch_queue(&ws->queue);
	free(ws);
}

static void
websocket_thread(void *arg)
{
	NrWebsocket *ws = (NrWebsocket *) arg;

	while (!ws->interrupted)
	{
		lws_service(ws->context, 50);
		/* 50 ms */

		/*
		 * Drain the outbound queue from this (service) thread only. Producers
		 * enqueue + lws_cancel_service; here we ask lws for a writable slot so
		 * the actual lws_write happens on the service thread.
		 */
		if (ws->instance && batch_queue_has_data(&ws->queue))
		{
			lws_callback_on_writable(ws->instance);
		}
	}
	pthread_exit(NULL);
}

/*  ****************************** Message ****************************** */
void
nws_send_batch_data(NrWebsocket * ws, const int batch_id,
					const MLStage ml_stage, const char *batch_data)
{
	cJSON	   *json = cJSON_CreateObject();

	cJSON_AddStringToObject(json, "version", "1");
	cJSON_AddStringToObject(json, "event", "batch_data");
	cJSON_AddStringToObject(json, "sessionId", ws->sid);
	cJSON_AddNumberToObject(json, "batchId", batch_id);
	cJSON_AddStringToObject(json, "stage", ML_STAGE[ml_stage]);
	cJSON_AddStringToObject(json, "byte", batch_data);

	/* hand off to the service thread (see queue_outbound) */
	queue_outbound(ws, json);

	cJSON_Delete(json);
}

void
nws_send_task(NrWebsocket * ws, MLTask ml_task, const char *table_name, void *task_spec)
{
	cJSON	   *json = cJSON_CreateObject();

	cJSON_AddStringToObject(json, "version", "1");
	cJSON_AddStringToObject(json, "event", "task");
	cJSON_AddStringToObject(json, "sessionId", ws->sid);
	cJSON_AddStringToObject(json, "type", ML_TASK[ml_task]);
	cJSON_AddStringToObject(json, "table", table_name);
	task_append_to_json(json, task_spec, ml_task);

	queue_outbound(ws, json);
	cJSON_Delete(json);
	while (!ws->task_acknowledged && !ws->interrupted)
	{
		/* wait for the task to be acknowledged */
		usleep(1000);
/* TODO: consider using a condition variable instead of busy */
		/* waiting */
	}
	if (!ws->task_acknowledged && ws->interrupted)
	{
		elog(ERROR, "AI engine connection closed before task was acknowledged");
	}
}

static void
queue_outbound(NrWebsocket * ws, const cJSON * json)
{
	char	   *data = cJSON_PrintUnformatted(json);

	if (data)
	{
		enqueue(&ws->queue, data);	/* enqueue strdups internally */
		free(data);
	}
	/* wake the service thread so it drains the queue promptly */
	if (ws->context)
	{
		lws_cancel_service(ws->context);
	}
}

static void
send_setup_signal(NrWebsocket * ws, const size_t cache_size)
{
	cJSON	   *json = cJSON_CreateObject();

	cJSON_AddStringToObject(json, "version", "1");
	cJSON_AddStringToObject(json, "event", "setup");
	cJSON_AddNumberToObject(json, "cacheSize", (double) cache_size);
	queue_outbound(ws, json);
	cJSON_Delete(json);
}

static void
send_disconnect_signal(NrWebsocket * ws)
{
	cJSON	   *json = cJSON_CreateObject();

	cJSON_AddStringToObject(json, "version", "1");
	cJSON_AddStringToObject(json, "event", "disconnect");
	cJSON_AddStringToObject(json, "sessionId", ws->sid);
	queue_outbound(ws, json);
	cJSON_Delete(json);
}

static void
buffer_data(NrWebsocket * websocket, const void *input, size_t len)
{
	lwsl_debug("Data is fragmented\n");

	if (websocket->buf == NULL)
	{
		websocket->buf = (char *) malloc(16384);
		websocket->buf_size = 16384;
		websocket->buf_used = 0;
		memset(websocket->buf, 0, websocket->buf_size);
	}

	if (websocket->buf_used + len >= websocket->buf_size)
	{
		lwsl_debug("Buffer overflow. Doubling the buffer size\n");
		char	   *old_buf = websocket->buf;

		websocket->buf = (char *) malloc(websocket->buf_size * 2);
		websocket->buf_size = websocket->buf_size * 2;
		memset(websocket->buf, 0, websocket->buf_size);
		memcpy(websocket->buf, old_buf, websocket->buf_used);
		free(old_buf);
	}

	memcpy(websocket->buf + websocket->buf_used, input, len);
	websocket->buf_used += len;

	return;
}

/*  ****************************** Callbacks and handler functions */
/*  ****************************** */
/**
 * Callback function for the websocket
 * @param wsi Websocket instance
 * @param callback_reason The reason of the callback
 * @param user User data
 * @param input Input data
 * @param len Length of the input data
 * @return int The status of the callback
 */
static int
callback(struct lws *wsi, enum lws_callback_reasons callback_reason,
		 void *user, void *input, size_t len)
{
	NrWebsocket *websocket = (NrWebsocket *) lws_wsi_user(wsi);

	switch (callback_reason)
	{
		case LWS_CALLBACK_CLIENT_RECEIVE:
			cJSON * json;
			if (websocket->buf != NULL)
			{
				buffer_data(websocket, input, len);
				json = cJSON_Parse((char *) websocket->buf);
			}
			else
			{
				json = cJSON_Parse((char *) input);
			}

			/* parse error */
			if (json == NULL)
			{
				if (websocket->buf == NULL)
				{
					buffer_data(websocket, input, len);
				}
				return 0;
			}

			/* get the "event" field from the JSON object */
			const		cJSON *event = cJSON_GetObjectItem(json, "event");

			if (event == NULL || !cJSON_IsString(event))
			{
				/*
				 * elog(ERROR, "Invalid JSON format: 'event' field missing or not
				 * a string\n");
				 */
				cJSON_Delete(json);

				if (websocket->buf == NULL)
				{
					buffer_data(websocket, input, len);
				}
				return 0;
			}

			/* parse success. Free the buffer if exists */
			if (websocket->buf != NULL)
			{
				free(websocket->buf);
				websocket->buf = NULL;
			}

			if (strcmp(event->valuestring, "request_data") == 0)
			{
				/* NEW: We should not use queue to block ourselves */
#if 0
				handle_request_data(websocket, json);
#endif
			}
			else if (strcmp(event->valuestring, "result") == 0)
			{
				handle_result(websocket, json);
			}
			else if (strcmp(event->valuestring, "ack_setup") == 0)
			{
				handle_ack_setup(websocket, json);
			}
			else if (strcmp(event->valuestring, "ack_disconnect") == 0)
			{
				handle_ack_disconnect(websocket);
			}
			else if (strcmp(event->valuestring, "ack_task") == 0)
			{
				handle_ack_task(websocket);
			}
			else
			{
				elog(ERROR, "Unknown event type: %s\n", event->valuestring);
			}
			cJSON_Delete(json);
			break;

		case LWS_CALLBACK_CLIENT_ESTABLISHED:
			/* connection established */
			websocket->connnected = 1;
			break;

		case LWS_CALLBACK_CLIENT_WRITEABLE:
		{
			/* write one queued message per writable slot (lws flow control) */
			char	   *msg = try_dequeue(&websocket->queue);

			if (msg != NULL)
			{
				const size_t msg_len = strlen(msg);
				unsigned char *buf = malloc(LWS_PRE + msg_len);

				if (buf)
				{
					memcpy(buf + LWS_PRE, msg, msg_len);
					lws_write(websocket->instance, buf + LWS_PRE, msg_len,
							  LWS_WRITE_TEXT);
					free(buf);
				}
				free(msg);

				/* more queued? ask for another writable slot */
				if (batch_queue_has_data(&websocket->queue))
				{
					lws_callback_on_writable(websocket->instance);
				}
			}
			break;
		}

		case LWS_CALLBACK_CLIENT_CLOSED:
			/* connection closed */
			websocket->interrupted = 1;
			break;

		default:
			break;
	}
	return 0;
}

static void
handle_result(NrWebsocket * ws, const cJSON * json)
{
	cJSON *result = cJSON_GetObjectItem(json, "byte");
	if (result == NULL || !cJSON_IsString(result))
	{
		elog(DEBUG2, "No 'byte' in response. Should be training/finetuning\n");
		int model_id = cJSON_GetObjectItem(json, "modelId")->valueint;
		ws->model_id = model_id;
		ws->result = NULL;
	} else {
		elog(DEBUG2, "'byte' found in response. Should be inference\n");
		ws->result = (char *) malloc(strlen(result->valuestring) + 1);
		strcpy(ws->result, result->valuestring);
	}
	ws->completed = 1;
}

static void
handle_ack_setup(NrWebsocket * ws, const cJSON * json)
{
	const		cJSON *session_id = cJSON_GetObjectItem(json, "sessionId");

	if (session_id == NULL || !cJSON_IsString(session_id))
	{
		elog(ERROR, "Invalid JSON format: 'sessionId' field missing or not a string\n");
		return;
	}
	strcpy(ws->sid, session_id->valuestring);
	/* set the session id */
	ws->setuped = 1;
}

static void
handle_ack_disconnect(NrWebsocket * ws)
{
	lws_close_reason(ws->instance, LWS_CLOSE_STATUS_NOSTATUS, NULL, 0);
	lws_set_timeout(ws->instance, PENDING_TIMEOUT_CLOSE_SEND, LWS_TO_KILL_SYNC);
}

static void
handle_ack_task(NrWebsocket * ws)
{
	ws->task_acknowledged = 1;
}
