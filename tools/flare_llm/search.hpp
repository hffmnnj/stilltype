#pragma once

// Independently implements FLARE's active stage-3 beam_search, not its
// inactive fairseq generator.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <utility>
#include <vector>

struct Candidate {
    float score;
    size_t parent;
    int token;
};

inline bool better(const Candidate & a, const Candidate & b) {
    if (a.score != b.score) return a.score > b.score;
    if (a.parent != b.parent) return a.parent < b.parent;
    return a.token < b.token;
}

struct Beam {
    float score = 0;
    std::vector<int> tokens;
};

struct Search {
    size_t remaining;
    int eos;
    std::vector<Beam> active{{}};
    std::vector<Beam> completed;
    explicit Search(size_t width, int eos_token) : remaining(width), eos(eos_token) {}

    std::vector<Candidate> select(const std::vector<std::vector<float>> & logits, float temp) const {
        if (logits.size() != active.size() || remaining == 0 || !(temp > 0))
            throw std::runtime_error("invalid search state");
        std::vector<Candidate> top;
        top.reserve(remaining);
        for (size_t p = 0; p < logits.size(); ++p) {
            if (logits[p].empty()) throw std::runtime_error("empty logits");
            float max_logit = *std::max_element(logits[p].begin(), logits[p].end());
            if (!std::isfinite(max_logit)) throw std::runtime_error("nonfinite logits");
            double sum = 0;
            for (float x : logits[p]) {
                if (!std::isfinite(x)) throw std::runtime_error("nonfinite logits");
                sum += std::exp(static_cast<double>(x - max_logit) / temp);
            }
            const float log_sum = static_cast<float>(std::log(sum));
            if (remaining == 1 && active.size() == 1) {
                const auto best = std::max_element(logits[p].begin(), logits[p].end());
                return {{active[p].score + (*best - max_logit) / temp - log_sum,
                         p, static_cast<int>(best - logits[p].begin())}};
            }
            for (size_t t = 0; t < logits[p].size(); ++t) {
                Candidate c{active[p].score + (logits[p][t] - max_logit) / temp - log_sum, p, static_cast<int>(t)};
                if (top.size() < remaining) {
                    top.push_back(c);
                    std::push_heap(top.begin(), top.end(), better);
                } else if (better(c, top.front())) {
                    std::pop_heap(top.begin(), top.end(), better);
                    top.back() = c;
                    std::push_heap(top.begin(), top.end(), better);
                }
            }
        }
        if (top.size() != remaining) throw std::runtime_error("beam exceeds candidate count");
        std::sort(top.begin(), top.end(), better);
        return top;
    }

    std::vector<Candidate> advance(const std::vector<Candidate> & selected, int step) {
        if (selected.size() != remaining || step < 1) throw std::runtime_error("invalid selection");
        std::vector<Beam> next;
        std::vector<Candidate> survivors;
        for (const auto & c : selected) {
            if (c.parent >= active.size()) throw std::runtime_error("invalid parent");
            Beam child = active[c.parent];
            child.tokens.push_back(c.token);
            child.score = c.score;
            if (c.token == eos) {
                child.score /= step;
                completed.push_back(std::move(child));
                --remaining;
            } else {
                next.push_back(std::move(child));
                survivors.push_back(c);
            }
        }
        active = std::move(next);
        return survivors;
    }

    const Beam & best_completed() const {
        if (completed.empty()) throw std::runtime_error("no completed sequence");
        return *std::max_element(completed.begin(), completed.end(),
            [](const Beam & a, const Beam & b) { return a.score < b.score; });
    }
};
