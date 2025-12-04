#include <iostream>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <netinet/in.h>

#include "microservice/ns_micro_utils.h"
#include "microservice/ns_micro_proto.h"
#include "microservice/ns_micro_service.h"


static volatile sig_atomic_t is_service_running = 1;

static void on_sigint(int) {
    is_service_running = 0;
}

static void on_task_done(NSMicroTaskMsg *resp, void *ctx) {
    int fd = *static_cast<int *>(ctx);

    uint8_t *body = nullptr;
    uint32_t body_len = 0;
    if (ns_micro_msg_serialize(resp, &body, &body_len) != 0) {
        return;
    }
    (void) ns_micro_frame_write(fd, body, body_len);
    free(body);
}

static void *handle_client(void *arg) {
    int fd = *static_cast<int *>(arg);
    free(arg);

    constexpr uint32_t MAX_FRAME = 256U << 20; // 256MB
    while (is_service_running) {
        uint8_t *body = nullptr;
        uint32_t body_len = 0;
        int return_code = ns_micro_frame_read(fd, &body, &body_len, MAX_FRAME);
        if (return_code != NS_SOCKET_OK) break;

        NSMicroTaskMsg *msg = ns_micro_msg_deserialize(body, body_len);
        free(body);
        if (!msg) break;

        int result = ns_micro_service_submit(msg, on_task_done, &fd);
        if (result != 0) {
            break;
        }
    }
    close(fd);
    return nullptr;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <store_path> <num_threads> <omp_parallelism>\n";
        return 1;
    }
    const char *store_path = argv[1];
    int num_threads = atoi(argv[2]);
    int omp_parallelism = atoi(argv[3]);
    ns_micro_service_init(store_path, num_threads, omp_parallelism);

    signal(SIGINT, on_sigint);
    signal(SIGTERM, on_sigint);

    int lfd = ns_micro_listen_tcp("0.0.0.0", 9999, 512);
    if (lfd < 0) {
        perror("listen");
        return 1;
    }
    printf("NeurStore microservice listening on :9999\\n");

    while (is_service_running) {
        sockaddr_in client_addr;
        socklen_t len = sizeof(client_addr);
        auto cfd = static_cast<int *>(malloc(sizeof(int)));
        *cfd = accept(lfd, reinterpret_cast<sockaddr *>(&client_addr), &len);
        if (*cfd < 0) {
            free(cfd);
            if (!is_service_running) break;
            continue;
        }

        pthread_t th;
        pthread_create(&th, nullptr, handle_client, cfd);
        pthread_detach(th);
    }

    close(lfd);
    ns_micro_service_shutdown();
    return 0;
}
