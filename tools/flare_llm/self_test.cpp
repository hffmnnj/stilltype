#include "search.hpp"
#include <cassert>
#include <cmath>
#include <iostream>

int main() {
    {
        Search s(3, 9);
        auto candidates = s.select({{1, 1, 1, 0}}, 1.2f);
        assert(candidates.size() == 3);
        assert(candidates[0].token == 0 && candidates[1].token == 1 && candidates[2].token == 2);
        s.advance(candidates, 1);
        auto tied = s.select({{0, 0}, {0, 0}, {0, 0}}, 1.2f);
        assert(tied[0].parent == 0 && tied[0].token == 0);
        assert(tied[1].parent == 0 && tied[1].token == 1);
        assert(tied[2].parent == 1 && tied[2].token == 0);
    }
    {
        Search s(3, 9);
        s.advance({{-0.1f, 0, 1}, {-0.2f, 0, 2}, {-0.3f, 0, 3}}, 1);
        auto mapping = s.advance({{-0.4f, 2, 4}, {-0.5f, 0, 5}, {-0.6f, 2, 6}}, 2);
        assert(mapping.size() == 3 && mapping[0].parent == 2 && mapping[1].parent == 0 && mapping[2].parent == 2);
        assert((s.active[0].tokens == std::vector<int>{3, 4}));
        assert((s.active[1].tokens == std::vector<int>{1, 5}));
        assert((s.active[2].tokens == std::vector<int>{3, 6}));
    }
    {
        Search s(3, 9);
        s.advance({{-0.1f, 0, 1}, {-0.2f, 0, 2}, {-0.3f, 0, 3}}, 1);
        auto survivors = s.advance({{-0.4f, 2, 4}, {-0.5f, 0, 9}, {-0.6f, 2, 5}}, 2);
        assert(survivors.size() == 2 && survivors[0].parent == 2 && survivors[1].parent == 2);
        assert((s.active[0].tokens == std::vector<int>{3, 4}));
        assert((s.active[1].tokens == std::vector<int>{3, 5}));
        assert(s.remaining == 2 && s.completed.size() == 1 && s.completed[0].score == -0.25f);
        s.advance({{-0.6f, 1, 9}, {-0.8f, 0, 9}}, 3);
        assert(s.remaining == 0 && s.completed.size() == 3);
        assert((s.best_completed().tokens == std::vector<int>{3, 5, 9}));
        assert(std::abs(s.best_completed().score - (-0.2f)) < 1e-6f);
    }
    {
        Search s(2, 9);
        s.advance({{-0.1f, 0, 1}, {-0.2f, 0, 2}}, 1);
        auto selected = s.select({{0, 0}, {3, -3}}, 1.2f);
        assert(selected.size() == 2);
        assert(selected[0].parent == 1 && selected[0].token == 0);
        assert(selected[1].parent == 0 && selected[1].token == 0);
    }
    std::cout << "selection self-test passed\n";
}
