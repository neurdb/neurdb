#include "neurstore/utils/timer.h"
#include "neurstore/utils/global.h"


/*********************************** C++ ***********************************/
void Timer::start(const std::string &label) {
    Time &time = times_[label];
    time.start = std::chrono::system_clock::now();
}

double Timer::stop(const std::string &label) {
    auto it = times_.find(label);
    if (it == times_.end() || it->second.start.time_since_epoch().count() == 0) {
        return -1.0;
    }
    Time &time = it->second;
    time.end = std::chrono::system_clock::now();
    time.duration += time.end - time.start;
    time.start = std::chrono::time_point<std::chrono::system_clock>();
    return time.duration.count();
}

const char *Timer::report() {
    report_.clear();
    for (const auto &pair: times_) {
        const Time &time = pair.second;
        report_ += pair.first + ": " + std::to_string(time.duration.count()) + "s\n";
    }
    return report_.c_str();
}

void Timer::reset() {
    times_.clear();
    report_.clear();
}

/************************************ C ************************************/
TimerC *ns_create_timer() {
    return new Timer();
}

TimerC *ns_get_timer() {
    return TIMER;
}

void ns_timer_start(TimerC *timer, const char *label) {
    if (timer) {
        timer->start(label);
    }
}

double ns_timer_stop(TimerC *timer, const char *label) {
    if (timer) {
        return timer->stop(label);
    }
    return -1.0;
}

void ns_timer_destroy(TimerC *timer) {
    delete timer;
}

const char *ns_timer_report(TimerC *timer) {
    if (timer) {
        return timer->report();
    }
    return nullptr;
}

void ns_timer_reset(TimerC *timer) {
    if (timer) {
        timer->reset();
    }
}
