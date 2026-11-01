#include "llama.h"
#include "search.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

using Clock = std::chrono::steady_clock;
constexpr int DIM = 3584, CAP = 128, EOS = 151643, LOGICAL_CTX = 512;
constexpr float TEMPERATURE = 1.2f;

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
void require(bool ok, const std::string & what) {
    if (!ok) throw std::runtime_error(what);
}
std::string json_string(const std::string & text) {
    std::string result = "\"";
    const char * hex = "0123456789abcdef";
    for (unsigned char c : text) {
        if (c == '"' || c == '\\') { result += '\\'; result += static_cast<char>(c); }
        else if (c < 32) { result += "\\u00"; result += hex[c >> 4]; result += hex[c & 15]; }
        else result += static_cast<char>(c);
    }
    return result + '"';
}

class FileDescriptor {
public:
    explicit FileDescriptor(int fd) : fd_(fd) {}
    ~FileDescriptor() { if (fd_ >= 0) close(fd_); }
    FileDescriptor(const FileDescriptor &) = delete;
    FileDescriptor & operator=(const FileDescriptor &) = delete;
    int get() const { return fd_; }
    void close_checked() {
        const int fd = fd_;
        fd_ = -1;
        if (fd >= 0 && close(fd) != 0) throw std::runtime_error("prediction output close failed");
    }
private:
    int fd_;
};

// Check the resolved parent, then create exactly once relative to its open
// directory. Never follow an existing final-component symlink.
int create_output(const std::filesystem::path & path) {
    require(!path.filename().empty() && path.filename() != "." && path.filename() != "..", "invalid output name");
    std::error_code ec;
    auto parent = std::filesystem::canonical(path.has_parent_path() ? path.parent_path() : ".", ec);
    require(!ec && std::filesystem::is_directory(parent, ec) && !ec, "output parent must be an existing directory");
    for (auto dir = parent;; dir = dir.parent_path()) {
        struct stat st{};
        const int result = lstat((dir / ".git").c_str(), &st);
        require(result != 0 && errno == ENOENT, "output parent is inside a Git checkout or cannot be inspected");
        if (dir == dir.root_path()) break;
    }
    FileDescriptor directory(open(parent.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    require(directory.get() >= 0, "cannot open output parent");
    int fd = openat(directory.get(), path.filename().c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
    require(fd >= 0, "cannot exclusively create output");
    // umask may remove permissions; ensure the promised owner-only mode.
    if (fchmod(fd, 0600) != 0) { close(fd); throw std::runtime_error("cannot secure output permissions"); }
    return fd;
}
void write_all(int fd, const std::string & text) {
    size_t offset = 0;
    while (offset < text.size()) {
        ssize_t n = write(fd, text.data() + offset, text.size() - offset);
        if (n < 0 && errno == EINTR) continue;
        require(n > 0, "prediction write failed");
        offset += static_cast<size_t>(n);
    }
}
uint64_t mem_available_kib() {
    std::ifstream file("/proc/meminfo");
    std::string key;
    while (file >> key) {
        if (key == "MemAvailable:") {
            uint64_t value = 0;
            require(static_cast<bool>(file >> value), "cannot measure available memory");
            return value;
        }
        std::string rest; std::getline(file, rest);
    }
    throw std::runtime_error("cannot measure available memory");
}
std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    std::vector<llama_token> tokens(text.size() + 16);
    int n = llama_tokenize(vocab, text.data(), static_cast<int32_t>(text.size()), tokens.data(),
                           static_cast<int32_t>(tokens.size()), false, false);
    require(n > 0 && n <= static_cast<int>(tokens.size()), "tokenization failed");
    tokens.resize(n);
    return tokens;
}
struct Batch {
    llama_batch value;
    Batch(int count, int embedding) : value(llama_batch_init(count, embedding, 1)) {}
    ~Batch() { llama_batch_free(value); }
    Batch(const Batch &) = delete;
    Batch & operator=(const Batch &) = delete;
};
void decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, int pos,
                   llama_seq_id seq, bool logits) {
    require(!tokens.empty() && pos >= 0 && pos + static_cast<int>(tokens.size()) <= LOGICAL_CTX, "token context limit");
    Batch owner(static_cast<int>(tokens.size()), 0);
    auto & batch = owner.value;
    require(batch.token && batch.pos && batch.n_seq_id && batch.seq_id && batch.logits, "token batch allocation failed");
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int i = 0; i < batch.n_tokens; ++i) {
        require(batch.seq_id[i] != nullptr, "token sequence allocation failed");
        batch.token[i] = tokens[i]; batch.pos[i] = pos + i;
        batch.n_seq_id[i] = 1; batch.seq_id[i][0] = seq;
        batch.logits[i] = logits && i == batch.n_tokens - 1;
    }
    int status = llama_decode(ctx, batch);
    require(status == 0, "token decode failed: " + std::to_string(status));
}
void decode_visual(llama_context * ctx, const std::vector<float> & data, int frames, int pos) {
    require(pos >= 0 && pos + frames <= LOGICAL_CTX, "visual context limit");
    Batch owner(frames, DIM);
    auto & batch = owner.value;
    require(batch.embd && batch.pos && batch.n_seq_id && batch.seq_id && batch.logits, "visual batch allocation failed");
    batch.n_tokens = frames;
    std::copy(data.begin(), data.end(), batch.embd);
    for (int i = 0; i < frames; ++i) {
        require(batch.seq_id[i] != nullptr, "visual sequence allocation failed");
        batch.pos[i] = pos + i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0; batch.logits[i] = false;
    }
    int status = llama_decode(ctx, batch);
    require(status == 0, "visual decode failed: " + std::to_string(status));
}
uint32_t little_u32(const unsigned char * p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
std::vector<float> read_visual(const std::filesystem::path & path, int & frames) {
    static_assert(sizeof(float) == 4 && std::numeric_limits<float>::is_iec559, "FP32 required");
    std::ifstream in(path, std::ios::binary);
    require(in.good(), "cannot read visual embeddings");
    unsigned char header[8] = {};
    in.read(reinterpret_cast<char *>(header), sizeof(header));
    uint32_t count = little_u32(header), dimension = little_u32(header + 4);
    require(in.good() && count >= 1 && count <= 200 && dimension == DIM, "invalid visual shape");
    frames = static_cast<int>(count);
    std::vector<unsigned char> bytes(static_cast<size_t>(frames) * DIM * 4);
    in.read(reinterpret_cast<char *>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    require(in.good() && in.peek() == std::char_traits<char>::eof(), "truncated or trailing visual data");
    std::vector<float> result(bytes.size() / 4);
    for (size_t i = 0; i < result.size(); ++i) {
        uint32_t bits = little_u32(bytes.data() + i * 4);
        std::memcpy(&result[i], &bits, sizeof(bits));
        require(std::isfinite(result[i]), "nonfinite visual values");
    }
    return result;
}
std::vector<float> copy_logits(llama_context * ctx, int count) {
    float * logits = llama_get_logits_ith(ctx, -1);
    require(logits != nullptr, "missing logits");
    return {logits, logits + count};
}
void remap_sequences(llama_context * ctx, int width, int old_count, const std::vector<Candidate> & survivors) {
    if (width == 1) return;
    auto mem = llama_get_memory(ctx);
    // Copy all parents to disjoint temporary IDs before reusing any live ID.
    for (size_t i = 0; i < survivors.size(); ++i)
        llama_memory_seq_cp(mem, static_cast<llama_seq_id>(survivors[i].parent), width + static_cast<int>(i), -1, -1);
    for (int i = 0; i < old_count; ++i)
        require(llama_memory_seq_rm(mem, i, -1, -1), "failed to remove old sequence");
    for (size_t i = 0; i < survivors.size(); ++i)
        llama_memory_seq_cp(mem, width + static_cast<int>(i), static_cast<llama_seq_id>(i), -1, -1);
    for (size_t i = 0; i < survivors.size(); ++i)
        require(llama_memory_seq_rm(mem, width + static_cast<int>(i), -1, -1), "failed to remove temporary sequence");
}
std::string detokenize(const llama_vocab * vocab, const Beam & beam) {
    std::string text;
    for (int token : beam.tokens) {
        if (token == EOS) break;
        char piece[1024];
        int n = llama_token_to_piece(vocab, token, piece, sizeof(piece), 0, false);
        require(n >= 0 && n <= static_cast<int>(sizeof(piece)), "token detokenization failed");
        text.append(piece, n);
    }
    return text;
}

int main(int argc, char ** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        std::cout << "usage: stilltype-flare-llm WIDTH BASE_GGUF ADAPTER_GGUF OUTPUT_JSONL VISUAL_BIN...\n";
        return 0;
    }
    if (argc < 6) {
        std::cerr << "usage: stilltype-flare-llm WIDTH BASE_GGUF ADAPTER_GGUF OUTPUT_JSONL VISUAL_BIN...\n";
        return 2;
    }
    int succeeded = 0, failed = 0;
    const int total = argc - 5;
    try {
        std::string w(argv[1]);
        require(w == "1" || w == "5" || w == "30", "beam width must be 1, 5, or 30");
        const int width = std::stoi(w);
        FileDescriptor out(create_output(argv[4]));
        // Qwen2.5-7B: 28 layers, 4 KV heads, 128 dimensions/head, F16 K and V.
        const uint64_t kv_estimate_bytes = uint64_t(LOGICAL_CTX) * width * 28 * 4 * 128 * 2 * 2;
        auto available = mem_available_kib();
        std::cerr << "available_kib=" << available << " estimated_kv_bytes=" << kv_estimate_bytes << "\n";
        require(available >= 8ULL * 1024 * 1024 + (kv_estimate_bytes + 1023) / 1024,
                "insufficient MemAvailable for 8 GiB baseline plus estimated KV cache; defer model load");
        llama_backend_init();
        struct Backend { ~Backend() { llama_backend_free(); } } backend;
        auto load_start = Clock::now();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[2], mp), llama_model_free);
        require(model != nullptr, "base model load failed");
        require(llama_model_n_embd(model.get()) == DIM && llama_model_n_layer(model.get()) == 28 &&
                llama_model_n_head(model.get()) == 28 && llama_model_n_head_kv(model.get()) == 4, "unexpected base architecture");
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        require(vocab && llama_vocab_n_tokens(vocab) == 152064 && llama_vocab_eos(vocab) == EOS, "unexpected vocabulary or EOS");
        std::unique_ptr<llama_adapter_lora, decltype(&llama_adapter_lora_free)> adapter(
            llama_adapter_lora_init(model.get(), argv[3]), llama_adapter_lora_free);
        require(adapter != nullptr, "adapter load failed");
        auto cp = llama_context_default_params();
        cp.n_ctx = LOGICAL_CTX * width; cp.n_batch = 256; cp.n_ubatch = 256;
        cp.n_seq_max = width == 1 ? 1 : width * 2;
        cp.kv_unified = width != 1;
        cp.n_threads = 4; cp.n_threads_batch = 4;
        cp.offload_kqv = false; cp.op_offload = false;
        require(cp.type_k == GGML_TYPE_F16 && cp.type_v == GGML_TYPE_F16, "unexpected KV cache type");
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        require(ctx && llama_n_ctx(ctx.get()) >= cp.n_ctx && llama_n_ctx_seq(ctx.get()) >= LOGICAL_CTX &&
                llama_n_seq_max(ctx.get()) >= cp.n_seq_max && llama_n_batch(ctx.get()) >= 200, "context initialization failed");
        llama_adapter_lora * adapters[] = {adapter.get()};
        float scales[] = {1.0f};
        require(llama_set_adapters_lora(ctx.get(), adapters, 1, scales) == 0, "adapter attach failed");
        auto a = tokenize(vocab, "A");
        auto prompt = tokenize(vocab, "transcribe the video: ");
        require(a == std::vector<llama_token>{32} && prompt == std::vector<llama_token>{1458, 3114, 279, 2766, 25, 220},
                "GGUF tokenizer differs from pinned official tokenizer on the prompt");
        prompt.push_back(EOS);
        std::cerr << "model_adapter_context_seconds=" << seconds(load_start) << "\n";
        for (int arg = 5; arg < argc; ++arg) {
            auto start = Clock::now();
            std::string id = std::filesystem::path(argv[arg]).stem().string(), text, error;
            int generated = 0, frames = 0, completed = 0, search_steps = 0;
            bool eos = false, search_complete = false;
            double prefill = 0;
            try {
                auto visual = read_visual(argv[arg], frames);
                require(a.size() + frames + prompt.size() + CAP <= LOGICAL_CTX, "context limit");
                llama_memory_clear(llama_get_memory(ctx.get()), true);
                int pos = 0;
                decode_tokens(ctx.get(), a, pos, 0, false); pos += static_cast<int>(a.size());
                decode_visual(ctx.get(), visual, frames, pos); pos += frames;
                decode_tokens(ctx.get(), prompt, pos, 0, true); pos += static_cast<int>(prompt.size());
                prefill = seconds(start);
                Search search(static_cast<size_t>(width), EOS);
                std::vector<std::vector<float>> logits{copy_logits(ctx.get(), llama_vocab_n_tokens(vocab))};
                for (int step = 1; step <= CAP; ++step) {
                    auto selected = search.select(logits, TEMPERATURE);
                    int old_count = static_cast<int>(search.active.size());
                    auto survivors = search.advance(selected, step);
                    completed = static_cast<int>(search.completed.size());
                    generated = step; search_steps = step;
                    if (search.remaining == 0) { search_complete = true; break; }
                    if (step == CAP) break;
                    remap_sequences(ctx.get(), width, old_count, survivors);
                    logits.clear();
                    logits.reserve(survivors.size());
                    for (size_t i = 0; i < survivors.size(); ++i) {
                        decode_tokens(ctx.get(), {survivors[i].token}, pos, static_cast<llama_seq_id>(i), true);
                        logits.push_back(copy_logits(ctx.get(), llama_vocab_n_tokens(vocab)));
                    }
                    ++pos;
                }
                if (search_complete) {
                    const auto & best = search.best_completed();
                    text = detokenize(vocab, best);
                    generated = static_cast<int>(best.tokens.size()); eos = true;
                } else {
                    if (!search.completed.empty()) {
                        const auto & best = search.best_completed();
                        text = detokenize(vocab, best); eos = true;
                        generated = static_cast<int>(best.tokens.size());
                    } else if (!search.active.empty()) {
                        text = detokenize(vocab, search.active.front());
                    }
                    error = width == 1 ? "output_cap_without_eos" : "output_cap_without_complete_search";
                }
            } catch (const std::exception & e) { error = e.what(); }
            std::ostringstream record;
            record << "{\"id\":" << json_string(id) << ",\"raw_text\":" << json_string(text)
                << ",\"error\":" << (error.empty() ? "null" : json_string(error))
                << ",\"generated_tokens\":" << generated << ",\"natural_eos\":" << (eos ? "true" : "false")
                << ",\"search_complete\":" << (search_complete ? "true" : "false")
                << ",\"search_steps\":" << search_steps << ",\"token_cap\":" << CAP
                << ",\"completed_beams\":" << completed << ",\"beam_width\":" << width
                << ",\"temperature\":1.2,\"logical_context\":512,\"kv_unified\":" << (cp.kv_unified ? "true" : "false")
                << ",\"native_context_slots\":" << llama_n_ctx(ctx.get()) << ",\"native_seq_max\":" << llama_n_seq_max(ctx.get())
                << ",\"estimated_kv_bytes\":" << kv_estimate_bytes
                << ",\"tie_order\":\"parent_then_token\",\"encoded_frames\":" << frames << ",\"prefill_seconds\":" << prefill
                << ",\"total_seconds\":" << seconds(start) << "}\n";
            write_all(out.get(), record.str());
            if (error.empty()) ++succeeded; else ++failed;
            std::cerr << "processed=" << succeeded + failed << " of=" << total << "\n";
        }
        out.close_checked();
    } catch (const std::exception & e) {
        std::cerr << "error: " << e.what() << "\n";
        std::cout << "{\"total\":" << total << ",\"succeeded\":" << succeeded << ",\"failed\":" << failed
                  << ",\"unprocessed\":" << total - succeeded - failed << ",\"output_complete\":false}\n";
        return 1;
    }
    std::cout << "{\"total\":" << total << ",\"succeeded\":" << succeeded << ",\"failed\":" << failed
              << ",\"unprocessed\":0,\"output_complete\":true}\n";
    return failed == 0 ? 0 : 1;
}
